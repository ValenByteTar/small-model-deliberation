"""
EXP-019: BitNet Relevance x Entailment Decomposition.

Hipotesis: BitNet puede resolver relevance y entailment por separado
cuando no puede resolver ambos en una clasificacion de 4 clases.

Arquitectura de 2 etapas + decision layer deterministico:

  EVIDENCE + CLAIM
       |
       v
  STAGE 1: RELEVANCE
  "Do CLAIM and EVIDENCE discuss the same specific subject and context?"
  YES / NO (con logprobs)
       |
       +-- NO --> UNRELATED (done)
       |
       +-- YES --> STAGE 2: ENTAILMENT
                   "Based on EVIDENCE, is CLAIM TRUE, FALSE, or PARTIALLY TRUE?"
                   TRUE / FALSE / PARTIALLY (con logprobs)
                        |
                        +-- TRUE --> SUPPORTS
                        +-- FALSE --> CONTRADICTS
                        +-- PARTIALLY --> PARTIAL

La autoridad esta en el sistema (decision layer deterministico),
no en el LLM. BitNet produce senales semanticas, no decisiones.

Metricas independientes:
  - Stage 1: relevance detection (binary, medida contra ground truth)
  - Stage 2: entailment accuracy (3-way, solo en casos relevantes)
  - Combined: sistema completo vs 50% target
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import requests

ROOT = Path(__file__).resolve().parent.parent

BITNET_ROOT = os.environ.get("BITNET_ROOT", os.path.expanduser("~/BitNet"))
MODEL_PATH = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
SERVER_EXE = "build/bin/Release/llama-server.exe"

BENCHMARK_PATH = ROOT / "benchmarks" / "semantic_assessment_v2.json"
OUTPUT_DIR = ROOT / "results" / "raw"

DEFAULT_PORT = 8096
SERVER_URL = f"http://127.0.0.1:{DEFAULT_PORT}"

# ----------------------------- Grammars -----------------------------

# Stage 1: Relevance binary
GRAMMAR_RELEVANCE = 'root ::= "YES" | "NO" | " YES" | " NO" | " Yes" | " No" | " yes" | " no"'

# Stage 2: Entailment 3-way
GRAMMAR_ENTAILMENT = 'root ::= "TRUE" | "FALSE" | "PARTIALLY" | " TRUE" | " FALSE" | " PARTIALLY" | " True" | " False" | " Partially" | " true" | " false" | " partially"'

# Stage 2b: Cascading para PARTIAL - si TRUE, preguntar FULLY/PARTIALLY
GRAMMAR_FULLY_PARTIAL = 'root ::= "FULLY" | "PARTIALLY" | " FULLY" | " PARTIALLY" | " Fully" | " Partially" | " fully" | " partially"'

# ----------------------------- Few-Shot: Relevance -----------------------------

# CRUCIAL: contradicciones son RELEVANTES (mismo sujeto, solo discrepan)
# wrong_subject y wrong_context son IRRELEVANTES
FEW_SHOT_RELEVANCE = '''Task: Do the CLAIM and EVIDENCE discuss the same specific subject (same product, standard, framework, technique) in the same context (same environment, sector, lifecycle phase)? Answer YES or NO. Note: if they discuss the same subject but disagree, that is still YES.

CLAIM: The NIST CSF has five core functions
EVIDENCE: The Framework Core consists of five Functions: Identify, Detect, Protect, Respond, and Recover.
Do the claim and evidence discuss the same specific subject and context? YES

CLAIM: The NIST CSF has three core functions
EVIDENCE: The Framework Core consists of five concurrent and continuous Functions: Identify, Detect, Protect, Respond, and Recover.
Do the claim and evidence discuss the same specific subject and context? YES

CLAIM: ISO 27001 requires risk assessment
EVIDENCE: SOC 2 requires organizations to perform periodic risk assessments of their systems.
Do the claim and evidence discuss the same specific subject and context? NO

CLAIM: Technique X is effective in cloud environments
EVIDENCE: Technique X is effective in traditional Windows domain environments for lateral movement.
Do the claim and evidence discuss the same specific subject and context? NO

CLAIM: Product A supports feature X
EVIDENCE: Product B supports feature X with advanced configuration options.
Do the claim and evidence discuss the same specific subject and context? NO

CLAIM: The tool automatically remediates vulnerabilities
EVIDENCE: The tool automatically identifies and categorizes vulnerabilities by severity.
Do the claim and evidence discuss the same specific subject and context? YES

'''

# ----------------------------- Few-Shot: Entailment -----------------------------

FEW_SHOT_ENTAILMENT = '''Task: The CLAIM and EVIDENCE discuss the same subject. Based on the EVIDENCE, is the CLAIM TRUE, FALSE, or PARTIALLY TRUE?

CLAIM: The NIST CSF has five core functions
EVIDENCE: The Framework Core consists of five Functions: Identify, Detect, Protect, Respond, and Recover.
Based on the evidence, the claim is: TRUE

CLAIM: The NIST CSF has three core functions
EVIDENCE: The Framework Core consists of five concurrent and continuous Functions: Identify, Detect, Protect, Respond, and Recover.
Based on the evidence, the claim is: FALSE

CLAIM: The framework includes risk assessment, mitigation, and recovery guidance
EVIDENCE: The framework provides risk assessment and mitigation guidance for organizations.
Based on the evidence, the claim is: PARTIALLY

CLAIM: Python 4.0 was released in 2023
EVIDENCE: Python 3.12 was released on October 2, 2023. There is no Python 4.0 release.
Based on the evidence, the claim is: FALSE

CLAIM: The framework mandates encryption at rest
EVIDENCE: The framework recommends encryption at rest as a best practice for sensitive data.
Based on the evidence, the claim is: PARTIALLY

'''

FEW_SHOT_FULLY_PARTIAL = '''Task: The claim is about the same subject as the evidence. Does the evidence fully or partially confirm the claim?

CLAIM: The NIST CSF has five core functions
EVIDENCE: The Framework Core consists of five Functions: Identify, Detect, Protect, Respond, and Recover.
Does the evidence fully or partially confirm the claim? FULLY

CLAIM: The framework includes risk assessment, mitigation, and recovery guidance
EVIDENCE: The framework provides risk assessment and mitigation guidance for organizations.
Does the evidence fully or partially confirm the claim? PARTIALLY

'''

# ----------------------------- Token Mapping -----------------------------

MAP_RELEVANCE = {
    "YES": "relevant", " YES": "relevant", "yes": "relevant", " Yes": "relevant", " yes": "relevant",
    "NO": "irrelevant", " NO": "irrelevant", "no": "irrelevant", " No": "irrelevant", " no": "irrelevant",
}

MAP_ENTAILMENT = {
    "TRUE": "SUPPORTS", " TRUE": "SUPPORTS", "true": "SUPPORTS", " True": "SUPPORTS", " true": "SUPPORTS",
    "FALSE": "CONTRADICTS", " FALSE": "CONTRADICTS", "false": "CONTRADICTS", " False": "CONTRADICTS", " false": "CONTRADICTS",
    "PARTIALLY": "PARTIAL", " PARTIALLY": "PARTIAL", "partially": "PARTIAL", " Partially": "PARTIAL", " partially": "PARTIAL",
    "PARTIAL": "PARTIAL", " PARTIAL": "PARTIAL", "partial": "PARTIAL", " Partial": "PARTIAL",
}

MAP_FULLY_PARTIAL = {
    "FULLY": "SUPPORTS", " FULLY": "SUPPORTS", "fully": "SUPPORTS", " Fully": "SUPPORTS",
    "PARTIALLY": "PARTIAL", " PARTIALLY": "PARTIAL", "partially": "PARTIAL", " Partially": "PARTIAL",
    "PARTIAL": "PARTIAL", " PARTIAL": "PARTIAL", "partial": "PARTIAL", " Partial": "PARTIAL",
}

# Ground truth mapping: expected -> relevant/irrelevant
EXPECTED_TO_RELEVANCE = {
    "SUPPORTS": "relevant",
    "CONTRADICTS": "relevant",
    "PARTIAL": "relevant",
    "UNRELATED": "irrelevant",
}

# ----------------------------- Server Control -----------------------------

_proc: subprocess.Popen | None = None


def start_server(port: int = DEFAULT_PORT) -> bool:
    global _proc
    exe = os.path.join(BITNET_ROOT, SERVER_EXE)
    model = os.path.join(BITNET_ROOT, MODEL_PATH)

    if not os.path.exists(exe) or not os.path.exists(model):
        print("ERROR: Binario o modelo de BitNet no encontrado", flush=True)
        return False

    try:
        r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
        if r.status_code == 200:
            print(f"  llama-server ya activo en puerto {port}", flush=True)
            return True
    except Exception:
        pass

    cmd = [
        exe, "-m", model, "--host", "127.0.0.1", "--port", str(port),
        "-t", "4", "-c", "2048", "-ngl", "0",
        "--override-kv", "tokenizer.ggml.pre=str:llama3",
    ]
    print(f"Iniciando llama-server en puerto {port}...", flush=True)
    _proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=0x08000000)

    t0 = time.time()
    while time.time() - t0 < 45:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if r.status_code == 200:
                print(f"llama-server listo en http://127.0.0.1:{port}", flush=True)
                return True
        except Exception:
            pass
        time.sleep(1)
    print("ERROR: Timeout iniciando llama-server", flush=True)
    return False


def stop_server():
    global _proc
    if _proc:
        try:
            _proc.terminate()
            _proc.wait(timeout=5)
        except Exception:
            try:
                _proc.kill()
            except Exception:
                pass
        _proc = None
        print("llama-server detenido.", flush=True)


# ----------------------------- LLM Calls with Logprobs -----------------------------

def call_with_logprobs(prompt: str, grammar: str, url: str, max_tokens: int = 6) -> dict:
    try:
        resp = requests.post(
            f"{url}/completion",
            json={
                "prompt": prompt, "stream": False, "temperature": 0.0,
                "max_tokens": max_tokens, "repeat_penalty": 1.0,
                "grammar": grammar, "n_probs": 8,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data.get("content", "").strip()
        probs = data.get("completion_probabilities", [])
        return {"content": content, "probs": probs}
    except Exception as exc:
        return {"content": f"ERROR: {exc}", "probs": []}


def get_token_logprobs(probs: list) -> List[dict]:
    if not probs:
        return []
    return probs[0].get("top_logprobs", [])


def aggregate_logprobs_by_label(token_logprobs: list, token_map: dict) -> Dict[str, float]:
    by_label = defaultdict(list)
    for tl in token_logprobs:
        tok = tl.get("token", "")
        lp = tl.get("logprob", -999)
        label = token_map.get(tok) or token_map.get(tok.strip()) or token_map.get(tok.lower()) or token_map.get(" " + tok.strip().lower())
        if label:
            by_label[label].append(lp)
    result = {}
    for label, lps in by_label.items():
        if lps:
            max_lp = max(lps)
            result[label] = max_lp + math.log(sum(math.exp(lp - max_lp) for lp in lps))
    return result


def map_raw_to_label(raw: str, token_map: dict) -> str:
    for key, val in token_map.items():
        if key in raw:
            return val
    return list(token_map.values())[0]  # fallback


# ----------------------------- Stage 1: Relevance -----------------------------

def predict_relevance(claim: str, evidence: str, url: str) -> dict:
    prompt = f"{FEW_SHOT_RELEVANCE}CLAIM: {claim.strip()}\nEVIDENCE: {evidence.strip()}\nDo the claim and evidence discuss the same specific subject and context?"
    result = call_with_logprobs(prompt, GRAMMAR_RELEVANCE, url, max_tokens=4)
    raw = result["content"]
    lps = get_token_logprobs(result["probs"])
    agg = aggregate_logprobs_by_label(lps, MAP_RELEVANCE)
    greedy = map_raw_to_label(raw, MAP_RELEVANCE)
    # Argmax over logprobs (alternative to greedy)
    argmax = max(agg, key=agg.get) if agg else greedy
    return {
        "raw": raw,
        "greedy": greedy,
        "argmax": argmax,
        "logprobs_by_label": agg,
        "raw_logprobs": [{"token": t.get("token", ""), "logprob": t.get("logprob", 0)} for t in lps],
    }


# ----------------------------- Stage 2: Entailment -----------------------------

def predict_entailment(claim: str, evidence: str, url: str) -> dict:
    prompt = f"{FEW_SHOT_ENTAILMENT}CLAIM: {claim.strip()}\nEVIDENCE: {evidence.strip()}\nBased on the evidence, the claim is:"
    result = call_with_logprobs(prompt, GRAMMAR_ENTAILMENT, url, max_tokens=6)
    raw = result["content"]
    lps = get_token_logprobs(result["probs"])
    agg = aggregate_logprobs_by_label(lps, MAP_ENTAILMENT)
    greedy = map_raw_to_label(raw, MAP_ENTAILMENT)
    argmax = max(agg, key=agg.get) if agg else greedy
    return {
        "raw": raw,
        "greedy": greedy,
        "argmax": argmax,
        "logprobs_by_label": agg,
        "raw_logprobs": [{"token": t.get("token", ""), "logprob": t.get("logprob", 0)} for t in lps],
    }


def predict_entailment_cascading(claim: str, evidence: str, url: str) -> dict:
    """Cascading: TRUE/FALSE first, then si TRUE -> FULLY/PARTIALLY."""
    # Paso 1: TRUE or FALSE?
    prompt1 = f"{FEW_SHOT_ENTAILMENT}CLAIM: {claim.strip()}\nEVIDENCE: {evidence.strip()}\nBased on the evidence, the claim is:"
    result1 = call_with_logprobs(prompt1, GRAMMAR_ENTAILMENT, url, max_tokens=6)
    raw1 = result1["content"]
    lps1 = get_token_logprobs(result1["probs"])
    agg1 = aggregate_logprobs_by_label(lps1, MAP_ENTAILMENT)
    greedy1 = map_raw_to_label(raw1, MAP_ENTAILMENT)

    # Si greedy dice SUPPORTS (TRUE), verificar FULLY/PARTIALLY
    if greedy1 == "SUPPORTS":
        prompt2 = f"{FEW_SHOT_FULLY_PARTIAL}CLAIM: {claim.strip()}\nEVIDENCE: {evidence.strip()}\nDoes the evidence fully or partially confirm the claim?"
        result2 = call_with_logprobs(prompt2, GRAMMAR_FULLY_PARTIAL, url, max_tokens=4)
        raw2 = result2["content"]
        lps2 = get_token_logprobs(result2["probs"])
        agg2 = aggregate_logprobs_by_label(lps2, MAP_FULLY_PARTIAL)
        greedy2 = map_raw_to_label(raw2, MAP_FULLY_PARTIAL)
        return {
            "raw": f"{raw1} -> {raw2}",
            "greedy": greedy2,  # SUPPORTS or PARTIAL
            "argmax": greedy2,
            "logprobs_by_label": agg1,  # keep stage 1 logprobs
            "logprobs_step2": agg2,
            "raw_logprobs": [{"token": t.get("token", ""), "logprob": t.get("logprob", 0)} for t in lps1],
            "raw_logprobs_step2": [{"token": t.get("token", ""), "logprob": t.get("logprob", 0)} for t in lps2],
        }
    else:
        return {
            "raw": raw1,
            "greedy": greedy1,
            "argmax": max(agg1, key=agg1.get) if agg1 else greedy1,
            "logprobs_by_label": agg1,
            "logprobs_step2": {},
            "raw_logprobs": [{"token": t.get("token", ""), "logprob": t.get("logprob", 0)} for t in lps1],
            "raw_logprobs_step2": [],
        }


# ----------------------------- Decision Layer (Deterministic) -----------------------------

def decision_layer(relevance: str, entailment: str) -> str:
    """Politica deterministica. Sin LLM authority."""
    if relevance == "irrelevant":
        return "UNRELATED"
    # relevant
    return entailment  # SUPPORTS, CONTRADICTS, or PARTIAL


# ----------------------------- Evaluation -----------------------------

def evaluate(cases: List[dict], url: str, use_cascading: bool = False) -> dict:
    mode_name = "cascading" if use_cascading else "direct"
    print("\n" + "=" * 75, flush=True)
    print(f"EXP-019: RELEVANCE x ENTAILMENT DECOMPOSITION ({mode_name})", flush=True)
    print("=" * 75, flush=True)

    # Metricas independientes
    # Stage 1: relevance (binary)
    s1_relevant_correct = 0
    s1_irrelevant_correct = 0
    s1_total_relevant = 0
    s1_total_irrelevant = 0
    s1_argmax_relevant_correct = 0
    s1_argmax_irrelevant_correct = 0

    # Stage 2: entailment (3-way, solo en casos relevantes)
    s2_correct_on_relevant = 0
    s2_total_on_relevant = 0  # casos donde stage 1 dijo relevant
    s2_correct_on_correct_relevant = 0  # casos donde stage 1 dijo relevant Y era correcto
    s2_total_on_correct_relevant = 0

    # Combined system
    combined_correct = 0
    combined_by_cat = defaultdict(lambda: {"total": 0, "correct": 0, "predictions": defaultdict(int)})

    # Also track argmax-based combined
    combined_argmax_correct = 0

    results = []
    t0 = time.time()

    for i, c in enumerate(cases):
        # Stage 1: relevance
        rel = predict_relevance(c["claim"], c["evidence"], url)
        rel_greedy = rel["greedy"]
        rel_argmax = rel["argmax"]

        # Ground truth relevance
        gt_rel = EXPECTED_TO_RELEVANCE[c["expected"]]
        s1_total_relevant += (gt_rel == "relevant")
        s1_total_irrelevant += (gt_rel == "irrelevant")
        if gt_rel == "relevant" and rel_greedy == "relevant":
            s1_relevant_correct += 1
        if gt_rel == "irrelevant" and rel_greedy == "irrelevant":
            s1_irrelevant_correct += 1
        if gt_rel == "relevant" and rel_argmax == "relevant":
            s1_argmax_relevant_correct += 1
        if gt_rel == "irrelevant" and rel_argmax == "irrelevant":
            s1_argmax_irrelevant_correct += 1

        # Stage 2: entailment (siempre correr, pero solo usar si relevant)
        if use_cascading:
            ent = predict_entailment_cascading(c["claim"], c["evidence"], url)
        else:
            ent = predict_entailment(c["claim"], c["evidence"], url)
        ent_greedy = ent["greedy"]
        ent_argmax = ent["argmax"]

        # Stage 2 metricas (solo en casos donde stage 1 dijo relevant)
        if rel_greedy == "relevant":
            s2_total_on_relevant += 1
            if ent_greedy == c["expected"]:
                s2_correct_on_relevant += 1
            # Solo en casos donde stage 1 acerto
            if gt_rel == "relevant":
                s2_total_on_correct_relevant += 1
                if ent_greedy == c["expected"]:
                    s2_correct_on_correct_relevant += 1

        # Decision layer (deterministic)
        final = decision_layer(rel_greedy, ent_greedy)
        final_argmax = decision_layer(rel_argmax, ent_argmax)

        is_ok = final == c["expected"]
        is_ok_argmax = final_argmax == c["expected"]
        if is_ok:
            combined_correct += 1
        if is_ok_argmax:
            combined_argmax_correct += 1

        cat = c["category"]
        combined_by_cat[cat]["total"] += 1
        combined_by_cat[cat]["predictions"][final] += 1
        if is_ok:
            combined_by_cat[cat]["correct"] += 1

        results.append({
            "id": c["id"],
            "category": cat,
            "expected": c["expected"],
            "gt_relevance": gt_rel,
            "stage1_greedy": rel_greedy,
            "stage1_argmax": rel_argmax,
            "stage1_raw": rel["raw"],
            "stage1_logprobs": rel["logprobs_by_label"],
            "stage1_raw_logprobs": rel["raw_logprobs"],
            "stage2_greedy": ent_greedy,
            "stage2_argmax": ent_argmax,
            "stage2_raw": ent["raw"],
            "stage2_logprobs": ent["logprobs_by_label"],
            "stage2_raw_logprobs": ent["raw_logprobs"],
            "final": final,
            "final_argmax": final_argmax,
            "correct": is_ok,
            "correct_argmax": is_ok_argmax,
        })

        status = "OK" if is_ok else f"X (got: {final})"
        print(f"  [{i+1:2d}/55] {c['id']:<8s} ({cat:<22s}) exp: {c['expected']:<11s} rel: {rel_greedy[:4]:<4s} ent: {ent_greedy[:4]:<4s} -> {status}", flush=True)

    wall = time.time() - t0
    n = len(cases)

    # Stage 1 report
    s1_acc = (s1_relevant_correct + s1_irrelevant_correct) / n
    s1_irr_rate = s1_irrelevant_correct / max(1, s1_total_irrelevant)
    s1_rel_rate = s1_relevant_correct / max(1, s1_total_relevant)
    s1_argmax_acc = (s1_argmax_relevant_correct + s1_argmax_irrelevant_correct) / n
    s1_argmax_irr_rate = s1_argmax_irrelevant_correct / max(1, s1_total_irrelevant)

    print("\n" + "=" * 75, flush=True)
    print("STAGE 1: RELEVANCE DETECTION (binary)", flush=True)
    print("-" * 75, flush=True)
    print(f"  Ground truth: {s1_total_relevant} relevant, {s1_total_irrelevant} irrelevant", flush=True)
    print(f"  Greedy:   relevant={s1_relevant_correct}/{s1_total_relevant}  irrelevant={s1_irrelevant_correct}/{s1_total_irrelevant}  acc={s1_acc:.1%}", flush=True)
    print(f"  Argmax:   relevant={s1_argmax_relevant_correct}/{s1_total_relevant}  irrelevant={s1_argmax_irrelevant_correct}/{s1_total_irrelevant}  acc={s1_argmax_acc:.1%}", flush=True)
    print(f"  Irrelevant detection rate (greedy): {s1_irr_rate:.1%} ({s1_irrelevant_correct}/{s1_total_irrelevant})", flush=True)
    print(f"  Irrelevant detection rate (argmax): {s1_argmax_irr_rate:.1%} ({s1_argmax_irrelevant_correct}/{s1_total_irrelevant})", flush=True)

    # Stage 2 report
    s2_acc_on_relevant = s2_correct_on_relevant / max(1, s2_total_on_relevant)
    s2_acc_on_correct = s2_correct_on_correct_relevant / max(1, s2_total_on_correct_relevant)
    print("\n" + "=" * 75, flush=True)
    print("STAGE 2: ENTAILMENT (3-way, solo en casos relevantes)", flush=True)
    print("-" * 75, flush=True)
    print(f"  Cases where Stage 1 said relevant: {s2_total_on_relevant}", flush=True)
    print(f"  Entailment accuracy (all relevant preds): {s2_acc_on_relevant:.1%} ({s2_correct_on_relevant}/{s2_total_on_relevant})", flush=True)
    print(f"  Cases where Stage 1 correctly said relevant: {s2_total_on_correct_relevant}", flush=True)
    print(f"  Entailment accuracy (correct relevant): {s2_acc_on_correct:.1%} ({s2_correct_on_correct_relevant}/{s2_total_on_correct_relevant})", flush=True)

    # Combined system
    combined_acc = combined_correct / n
    combined_argmax_acc = combined_argmax_correct / n
    print("\n" + "=" * 75, flush=True)
    print(f"COMBINED SYSTEM ({mode_name})", flush=True)
    print("-" * 75, flush=True)
    print(f"{'Categoria':<25} | {'Casos':<6} | {'Correctos':<10} | {'Accuracy':<9} | {'Predicciones'}", flush=True)
    print("-" * 75, flush=True)
    for cat, data in sorted(combined_by_cat.items()):
        tot = data["total"]
        corr = data["correct"]
        a = corr / tot if tot > 0 else 0.0
        preds_str = ", ".join(f"{k}:{v}" for k, v in sorted(data["predictions"].items()))
        print(f"{cat:<25} | {tot:<6} | {corr:<10} | {a:>8.1%} | {preds_str}", flush=True)
    print("-" * 75, flush=True)
    print(f"{'TOTAL GREEDY':<25} | {n:<6} | {combined_correct:<10} | {combined_acc:>8.1%} | Wall: {wall/60:.1f}m", flush=True)
    print(f"{'TOTAL ARGMAX':<25} | {n:<6} | {combined_argmax_correct:<10} | {combined_argmax_acc:>8.1%} |", flush=True)
    print("=" * 75, flush=True)

    target = 28
    print(f"\nTarget 50%: {target}/55 {'ALCANZADO' if combined_correct >= target else 'NO alcanzado'} (greedy)", flush=True)
    print(f"Target 50%: {target}/55 {'ALCANZADO' if combined_argmax_correct >= target else 'NO alcanzado'} (argmax)", flush=True)

    # Comparacion
    print(f"\nComparacion:", flush=True)
    print(f"  EXP-017 techo single:     29.1% (16/55)", flush=True)
    print(f"  EXP-018 logit ensemble:   40.0% (22/55)", flush=True)
    print(f"  EXP-019 combined greedy:  {combined_acc:.1%} ({combined_correct}/55)", flush=True)
    print(f"  EXP-019 combined argmax:  {combined_argmax_acc:.1%} ({combined_argmax_correct}/55)", flush=True)

    report = {
        "model": "BitNet-b1.58-2B-4T",
        "experiment": "EXP-019: Relevance x Entailment Decomposition",
        "mode": mode_name,
        "benchmark": "semantic_assessment_v2.json",
        "total_cases": n,
        "stage1": {
            "ground_truth_relevant": s1_total_relevant,
            "ground_truth_irrelevant": s1_total_irrelevant,
            "greedy": {
                "relevant_correct": s1_relevant_correct,
                "irrelevant_correct": s1_irrelevant_correct,
                "accuracy": round(s1_acc, 4),
                "irrelevant_detection_rate": round(s1_irr_rate, 4),
            },
            "argmax": {
                "relevant_correct": s1_argmax_relevant_correct,
                "irrelevant_correct": s1_argmax_irrelevant_correct,
                "accuracy": round(s1_argmax_acc, 4),
                "irrelevant_detection_rate": round(s1_argmax_irr_rate, 4),
            },
        },
        "stage2": {
            "cases_where_relevant": s2_total_on_relevant,
            "accuracy_on_relevant": round(s2_acc_on_relevant, 4),
            "cases_where_correctly_relevant": s2_total_on_correct_relevant,
            "accuracy_on_correctly_relevant": round(s2_acc_on_correct, 4),
        },
        "combined": {
            "greedy_accuracy": round(combined_acc, 4),
            "greedy_correct": combined_correct,
            "argmax_accuracy": round(combined_argmax_acc, 4),
            "argmax_correct": combined_argmax_correct,
        },
        "by_category": {
            cat: {
                "total": d["total"],
                "correct": d["correct"],
                "accuracy": round(d["correct"] / d["total"], 4) if d["total"] > 0 else 0.0,
                "predictions": dict(d["predictions"]),
            }
            for cat, d in combined_by_cat.items()
        },
        "wall_time_s": round(wall, 1),
        "cases": results,
    }
    return report


# ----------------------------- Main -----------------------------

def main():
    port = DEFAULT_PORT
    url = f"http://127.0.0.1:{port}"

    if not start_server(port):
        return 1

    try:
        with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
            cases = json.load(f)["cases"]

        # Modo 1: direct (TRUE/FALSE/PARTIALLY en una pasada)
        report_direct = evaluate(cases, url, use_cascading=False)
        out1 = OUTPUT_DIR / "bitnet_relevance_entailment_direct.json"
        with open(out1, "w", encoding="utf-8") as f:
            json.dump(report_direct, f, indent=2, ensure_ascii=False)
        print(f"\nGuardado: {out1}\n", flush=True)

        # Modo 2: cascading (TRUE/FALSE -> si TRUE, FULLY/PARTIALLY)
        report_cascading = evaluate(cases, url, use_cascading=True)
        out2 = OUTPUT_DIR / "bitnet_relevance_entailment_cascading.json"
        with open(out2, "w", encoding="utf-8") as f:
            json.dump(report_cascading, f, indent=2, ensure_ascii=False)
        print(f"\nGuardado: {out2}\n", flush=True)

        # Resumen final
        print("\n" + "=" * 75, flush=True)
        print("RESUMEN EXP-019", flush=True)
        print("=" * 75, flush=True)
        for name, rep in [("Direct", report_direct), ("Cascading", report_cascading)]:
            s1 = rep["stage1"]["greedy"]
            s1a = rep["stage1"]["argmax"]
            s2 = rep["stage2"]
            comb = rep["combined"]
            print(f"\n{name}:", flush=True)
            print(f"  Stage 1 (relevance): acc={s1['accuracy']:.1%}  irr_detection={s1['irrelevant_detection_rate']:.1%}", flush=True)
            print(f"  Stage 1 (argmax):    acc={s1a['accuracy']:.1%}  irr_detection={s1a['irrelevant_detection_rate']:.1%}", flush=True)
            print(f"  Stage 2 (entailment): acc_on_relevant={s2['accuracy_on_relevant']:.1%}  acc_on_correct={s2['accuracy_on_correctly_relevant']:.1%}", flush=True)
            print(f"  Combined greedy:      {comb['greedy_accuracy']:.1%} ({comb['greedy_correct']}/55)", flush=True)
            print(f"  Combined argmax:      {comb['argmax_accuracy']:.1%} ({comb['argmax_correct']}/55)", flush=True)
        print("=" * 75, flush=True)

    finally:
        stop_server()


if __name__ == "__main__":
    main()
