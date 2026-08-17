"""
EXP-020: BitNet Granularity Probe + Atomic Decomposition.

Objetivo: determinar si BitNet posee informacion suficiente para
distinguir soporte completo de soporte parcial, independientemente
del framing utilizado.

Fase 1: Granularity Probe directo
  - Casos minimos controlados (sin vocabulario complejo, sin relevance
    confounders, sin entidades ambiguas)
  - 3 regimenes: NLI 4-way, NLI cascading, directo FULL/PARTIAL
  - Mide si BitNet puede emitir PARTIAL cuando el evidence cubre
    parte del claim

Fase 2: Atomic Decomposition
  - Descomponer claims con conjunciones en proposiciones atomicas
  - Pedir a BitNet TRUE/FALSE para cada proposicion atomica
  - Agregar deterministamente:
      N/N TRUE -> SUPPORTS
      0/N TRUE -> CONTRADICTS (si hay overlap) o UNRELATED (si no)
      1..N-1 / N TRUE -> PARTIAL
  - Mide si BitNet puede resolver proposiciones atomicas cuando
    no puede resolver composicion

La autoridad esta en el aggregator deterministico, no en el LLM.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import requests

ROOT = Path(__file__).resolve().parent.parent

BITNET_ROOT = os.environ.get("BITNET_ROOT", os.path.expanduser("~/BitNet"))
MODEL_PATH = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
SERVER_EXE = "build/bin/Release/llama-server.exe"

PROBE_PATH = ROOT / "benchmarks" / "granularity_probe_v1.json"
V2_PATH = ROOT / "benchmarks" / "semantic_assessment_v2.json"
OUTPUT_DIR = ROOT / "results" / "raw"

DEFAULT_PORT = 8097
SERVER_URL = f"http://127.0.0.1:{DEFAULT_PORT}"

# ----------------------------- Grammars -----------------------------

GRAMMAR_NLI_4 = 'root ::= "YES" | "NO" | "PARTIALLY" | "NOT_MENTIONED" | " YES" | " NO" | " PARTIALLY" | " NOT_MENTIONED" | " Yes" | " No" | " yes" | " no"'
GRAMMAR_TRUE_FALSE = 'root ::= "TRUE" | "FALSE" | " TRUE" | " FALSE" | " True" | " False" | " true" | " false"'
GRAMMAR_FULL_PARTIAL = 'root ::= "FULL" | "PARTIAL" | " FULL" | " PARTIAL" | " Full" | " Partial" | " full" | " partial"'

# ----------------------------- Few-Shot -----------------------------

# Regimen 1: NLI 4-way (YES/NO/PARTIALLY/NOT_MENTIONED)
FEW_SHOT_NLI4 = '''Task: Based on the EVIDENCE, answer if the CLAIM is YES (fully supported), NO (contradicted), PARTIALLY (partially supported), or NOT_MENTIONED (evidence does not address the claim).

CLAIM: The system supports encryption at rest, encryption in transit, and access logging.
EVIDENCE: The system supports encryption at rest, encryption in transit, and access logging.
Is the claim supported by the evidence? YES

CLAIM: The system supports encryption at rest, encryption in transit, and access logging.
EVIDENCE: The system supports encryption at rest and encryption in transit.
Is the claim supported by the evidence? PARTIALLY

CLAIM: The system supports encryption at rest, encryption in transit, and access logging.
EVIDENCE: The system supports quantum key distribution.
Is the claim supported by the evidence? NOT_MENTIONED

CLAIM: The system supports encryption at rest only.
EVIDENCE: The system supports encryption at rest, encryption in transit, and access logging.
Is the claim supported by the evidence? NO

'''

# Regimen 2: NLI cascading (TRUE/FALSE -> si TRUE, FULL/PARTIAL)
FEW_SHOT_CASCADING_STEP1 = '''Task: Based on the EVIDENCE, is the CLAIM TRUE or FALSE?

CLAIM: The system supports encryption at rest, encryption in transit, and access logging.
EVIDENCE: The system supports encryption at rest, encryption in transit, and access logging.
Based on the evidence, the claim is: TRUE

CLAIM: The system supports encryption at rest only.
EVIDENCE: The system supports encryption at rest, encryption in transit, and access logging.
Based on the evidence, the claim is: FALSE

'''

FEW_SHOT_CASCADING_STEP2 = '''Task: The claim is partially supported by the evidence. Does the evidence fully or partially confirm the claim?

CLAIM: The system supports encryption at rest, encryption in transit, and access logging.
EVIDENCE: The system supports encryption at rest and encryption in transit.
Does the evidence fully or partially confirm the claim? PARTIAL

CLAIM: The system supports encryption at rest, encryption in transit, and access logging.
EVIDENCE: The system supports encryption at rest, encryption in transit, and access logging.
Does the evidence fully or partially confirm the claim? FULL

'''

# Regimen 3: Directo FULL/PARTIAL/CONTRADICTS/UNRELATED
FEW_SHOT_DIRECT = '''Task: Classify the relationship between CLAIM and EVIDENCE as FULL, PARTIAL, CONTRADICTS, or UNRELATED.

CLAIM: The system supports encryption at rest, encryption in transit, and access logging.
EVIDENCE: The system supports encryption at rest, encryption in transit, and access logging.
RELATION: FULL

CLAIM: The system supports encryption at rest, encryption in transit, and access logging.
EVIDENCE: The system supports encryption at rest and encryption in transit.
RELATION: PARTIAL

CLAIM: The system supports encryption at rest only.
EVIDENCE: The system supports encryption at rest, encryption in transit, and access logging.
RELATION: CONTRADICTS

CLAIM: The system supports quantum encryption.
EVIDENCE: The system supports encryption at rest, encryption in transit, and access logging.
RELATION: UNRELATED

'''

# Fase 2: Atomic decomposition - TRUE/FALSE per proposition
FEW_SHOT_ATOMIC = '''Task: Based on the EVIDENCE, is the specific CLAIM TRUE or FALSE? Answer TRUE if the evidence explicitly confirms the claim, FALSE if it does not.

CLAIM: The system supports encryption at rest.
EVIDENCE: The system supports encryption at rest, encryption in transit, and access logging.
Based on the evidence, the claim is: TRUE

CLAIM: The system supports access logging.
EVIDENCE: The system supports encryption at rest and encryption in transit.
Based on the evidence, the claim is: FALSE

CLAIM: The protocol provides authentication.
EVIDENCE: The protocol provides authentication, authorization, and auditing.
Based on the evidence, the claim is: TRUE

CLAIM: The protocol provides auditing.
EVIDENCE: The protocol provides authentication and authorization.
Based on the evidence, the claim is: FALSE

'''

# ----------------------------- Token Mapping -----------------------------

MAP_NLI4 = {
    "YES": "SUPPORTS", " YES": "SUPPORTS", "yes": "SUPPORTS", " Yes": "SUPPORTS", " yes": "SUPPORTS",
    "NO": "CONTRADICTS", " NO": "CONTRADICTS", "no": "CONTRADICTS", " No": "CONTRADICTS", " no": "CONTRADICTS",
    "PARTIALLY": "PARTIAL", " PARTIALLY": "PARTIAL", "partially": "PARTIAL", " Partially": "PARTIAL",
    "NOT_MENTIONED": "UNRELATED", " NOT_MENTIONED": "UNRELATED",
}

MAP_TRUE_FALSE = {
    "TRUE": "TRUE", " TRUE": "TRUE", "true": "TRUE", " True": "TRUE", " true": "TRUE",
    "FALSE": "FALSE", " FALSE": "FALSE", "false": "FALSE", " False": "FALSE", " false": "FALSE",
}

MAP_FULL_PARTIAL = {
    "FULL": "SUPPORTS", " FULL": "SUPPORTS", "full": "SUPPORTS", " Full": "SUPPORTS",
    "PARTIAL": "PARTIAL", " PARTIAL": "PARTIAL", "partial": "PARTIAL", " Partial": "PARTIAL",
}

MAP_DIRECT = {
    "FULL": "SUPPORTS", " FULL": "SUPPORTS", "full": "SUPPORTS", " Full": "SUPPORTS",
    "PARTIAL": "PARTIAL", " PARTIAL": "PARTIAL", "partial": "PARTIAL", " Partial": "PARTIAL",
    "CONTRADICTS": "CONTRADICTS", " CONTRADICTS": "CONTRADICTS", "contradicts": "CONTRADICTS",
    "UNRELATED": "UNRELATED", " UNRELATED": "UNRELATED", "unrelated": "UNRELATED",
}

# ----------------------------- Server Control -----------------------------

_proc: subprocess.Popen | None = None


def start_server(port: int = DEFAULT_PORT) -> bool:
    global _proc
    exe = os.path.join(BITNET_ROOT, SERVER_EXE)
    model = os.path.join(BITNET_ROOT, MODEL_PATH)
    if not os.path.exists(exe) or not os.path.exists(model):
        print("ERROR: Binario o modelo no encontrado", flush=True)
        return False
    try:
        r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
        if r.status_code == 200:
            print(f"  llama-server ya activo en puerto {port}", flush=True)
            return True
    except Exception:
        pass
    cmd = [exe, "-m", model, "--host", "127.0.0.1", "--port", str(port),
           "-t", "4", "-c", "2048", "-ngl", "0",
           "--override-kv", "tokenizer.ggml.pre=str:llama3"]
    print(f"Iniciando llama-server en puerto {port}...", flush=True)
    _proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=0x08000000)
    t0 = time.time()
    while time.time() - t0 < 45:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if r.status_code == 200:
                print(f"llama-server listo", flush=True)
                return True
        except Exception:
            pass
        time.sleep(1)
    print("ERROR: Timeout", flush=True)
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


# ----------------------------- LLM Calls -----------------------------

def call_with_logprobs(prompt: str, grammar: str, url: str, max_tokens: int = 6) -> dict:
    try:
        resp = requests.post(
            f"{url}/completion",
            json={"prompt": prompt, "stream": False, "temperature": 0.0,
                  "max_tokens": max_tokens, "repeat_penalty": 1.0,
                  "grammar": grammar, "n_probs": 8},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return {"content": data.get("content", "").strip(), "probs": data.get("completion_probabilities", [])}
    except Exception as exc:
        return {"content": f"ERROR: {exc}", "probs": []}


def get_token_logprobs(probs: list) -> List[dict]:
    if not probs:
        return []
    return probs[0].get("top_logprobs", [])


def aggregate_logprobs(token_logprobs: list, token_map: dict) -> Dict[str, float]:
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


def map_raw(raw: str, token_map: dict) -> str:
    raw_lower = raw.strip().lower()
    for key, val in token_map.items():
        if key.strip().lower() in raw_lower:
            return val
    return list(set(token_map.values()))[0]


# ----------------------------- Fase 1: Granularity Probe -----------------------------

def predict_nli4(claim: str, evidence: str, url: str) -> dict:
    prompt = f"{FEW_SHOT_NLI4}CLAIM: {claim.strip()}\nEVIDENCE: {evidence.strip()}\nIs the claim supported by the evidence?"
    result = call_with_logprobs(prompt, GRAMMAR_NLI_4, url, max_tokens=6)
    raw = result["content"]
    lps = get_token_logprobs(result["probs"])
    agg = aggregate_logprobs(lps, MAP_NLI4)
    greedy = map_raw(raw, MAP_NLI4)
    argmax = max(agg, key=agg.get) if agg else greedy
    return {"raw": raw, "greedy": greedy, "argmax": argmax, "logprobs_by_label": agg,
            "raw_logprobs": [{"token": t.get("token", ""), "logprob": t.get("logprob", 0)} for t in lps]}


def predict_cascading(claim: str, evidence: str, url: str) -> dict:
    prompt1 = f"{FEW_SHOT_CASCADING_STEP1}CLAIM: {claim.strip()}\nEVIDENCE: {evidence.strip()}\nBased on the evidence, the claim is:"
    result1 = call_with_logprobs(prompt1, GRAMMAR_TRUE_FALSE, url, max_tokens=4)
    raw1 = result1["content"]
    lps1 = get_token_logprobs(result1["probs"])
    agg1 = aggregate_logprobs(lps1, MAP_TRUE_FALSE)
    greedy1 = map_raw(raw1, MAP_TRUE_FALSE)

    if greedy1 == "TRUE":
        prompt2 = f"{FEW_SHOT_CASCADING_STEP2}CLAIM: {claim.strip()}\nEVIDENCE: {evidence.strip()}\nDoes the evidence fully or partially confirm the claim?"
        result2 = call_with_logprobs(prompt2, GRAMMAR_FULL_PARTIAL, url, max_tokens=4)
        raw2 = result2["content"]
        lps2 = get_token_logprobs(result2["probs"])
        agg2 = aggregate_logprobs(lps2, MAP_FULL_PARTIAL)
        greedy2 = map_raw(raw2, MAP_FULL_PARTIAL)
        return {"raw": f"{raw1} -> {raw2}", "greedy": greedy2, "argmax": greedy2,
                "logprobs_by_label": agg1, "logprobs_step2": agg2,
                "raw_logprobs": [{"token": t.get("token", ""), "logprob": t.get("logprob", 0)} for t in lps1]}
    else:
        # FALSE -> CONTRADICTS (simplificado para probe)
        return {"raw": raw1, "greedy": "CONTRADICTS", "argmax": "CONTRADICTS",
                "logprobs_by_label": agg1, "logprobs_step2": {},
                "raw_logprobs": [{"token": t.get("token", ""), "logprob": t.get("logprob", 0)} for t in lps1]}


def predict_direct(claim: str, evidence: str, url: str) -> dict:
    prompt = f"{FEW_SHOT_DIRECT}CLAIM: {claim.strip()}\nEVIDENCE: {evidence.strip()}\nRELATION:"
    result = call_with_logprobs(prompt, 'root ::= "FULL" | "PARTIAL" | "CONTRADICTS" | "UNRELATED" | " FULL" | " PARTIAL" | " CONTRADICTS" | " UNRELATED"', url, max_tokens=6)
    raw = result["content"]
    lps = get_token_logprobs(result["probs"])
    agg = aggregate_logprobs(lps, MAP_DIRECT)
    greedy = map_raw(raw, MAP_DIRECT)
    argmax = max(agg, key=agg.get) if agg else greedy
    return {"raw": raw, "greedy": greedy, "argmax": argmax, "logprobs_by_label": agg,
            "raw_logprobs": [{"token": t.get("token", ""), "logprob": t.get("logprob", 0)} for t in lps]}


def run_fase1(cases: list, url: str) -> list:
    """Fase 1: Granularity probe directo con 3 regimenes."""
    regimes = [
        ("NLI 4-way", predict_nli4),
        ("NLI Cascading", predict_cascading),
        ("Direct FULL/PARTIAL", predict_direct),
    ]
    all_reports = []
    for name, fn in regimes:
        print(f"\n{'='*70}", flush=True)
        print(f"FASE 1: {name}", flush=True)
        print(f"{'='*70}", flush=True)
        by_cat = defaultdict(lambda: {"total": 0, "correct": 0, "predictions": defaultdict(int)})
        total_correct = 0
        results = []
        for i, c in enumerate(cases):
            pred_data = fn(c["claim"], c["evidence"], url)
            pred = pred_data["greedy"]
            is_ok = pred == c["expected"]
            if is_ok:
                total_correct += 1
            cat = c["category"]
            by_cat[cat]["total"] += 1
            by_cat[cat]["predictions"][pred] += 1
            if is_ok:
                by_cat[cat]["correct"] += 1
            results.append({
                "id": c["id"], "category": cat, "expected": c["expected"],
                "predicted": pred, "raw": pred_data["raw"], "correct": is_ok,
                "logprobs_by_label": pred_data["logprobs_by_label"],
                "argmax": pred_data["argmax"],
            })
            status = "OK" if is_ok else f"X (got: {pred}, raw: {pred_data['raw'][:20]})"
            print(f"  [{i+1:2d}/{len(cases)}] {c['id']:<8s} ({cat:<24s}) exp: {c['expected']:<11s} -> {status}", flush=True)

        acc = total_correct / len(cases)
        print(f"\n{'-'*75}", flush=True)
        print(f"{'Categoria':<27} | {'Casos':<6} | {'Correctos':<10} | {'Accuracy':<9} | {'Predicciones'}", flush=True)
        print(f"{'-'*75}", flush=True)
        for cat, data in sorted(by_cat.items()):
            a = data["correct"] / data["total"] if data["total"] > 0 else 0.0
            preds_str = ", ".join(f"{k}:{v}" for k, v in sorted(data["predictions"].items()))
            print(f"{cat:<27} | {data['total']:<6} | {data['correct']:<10} | {a:>8.1%} | {preds_str}", flush=True)
        print(f"{'-'*75}", flush=True)
        print(f"{'TOTAL ' + name:<27} | {len(cases):<6} | {total_correct:<10} | {acc:>8.1%} |", flush=True)

        # Tambien argmax
        argmax_correct = sum(1 for r in results if r["argmax"] == r["expected"])
        print(f"{'TOTAL ' + name + ' (argmax)':<27} | {len(cases):<6} | {argmax_correct:<10} | {argmax_correct/len(cases):>8.1%} |", flush=True)

        all_reports.append({
            "regime": name,
            "greedy_accuracy": round(acc, 4),
            "greedy_correct": total_correct,
            "argmax_accuracy": round(argmax_correct / len(cases), 4),
            "argmax_correct": argmax_correct,
            "by_category": {cat: {"total": d["total"], "correct": d["correct"],
                                  "accuracy": round(d["correct"] / d["total"], 4) if d["total"] > 0 else 0.0,
                                  "predictions": dict(d["predictions"])}
                            for cat, d in by_cat.items()},
            "cases": results,
        })
    return all_reports


# ----------------------------- Fase 2: Atomic Decomposition -----------------------------

def decompose_claim(claim: str) -> List[str]:
    """Descompone un claim con conjunciones en proposiciones atomicas.
    Estrategia: split por ', ' y ' and ', inferir prefijo comun, limpiar.
    """
    claim_clean = claim.strip().rstrip(".")
    # Split por ', ' y ' and '
    parts = [p.strip() for p in re.split(r',\s+|\s+and\s+', claim_clean) if p.strip()]
    if len(parts) <= 1:
        return [claim_clean]

    # Inferir prefijo comun: "The system supports", "The protocol provides", etc.
    # El prefijo es el sujeto + verbo principal. Heuristica: buscar el verbo principal
    # en la primera parte y usar todo hasta despues del verbo como prefijo.
    first = parts[0]
    first_words = first.split()

    # Verbos comunes que indican donde termina el prefijo
    VERBS = {"supports", "provides", "performs", "covers", "applies", "encrypts",
             "includes", "requires", "uses", "has", "is", "are", "can", "cannot"}

    # Encontrar el indice del verbo principal
    verb_idx = None
    for i, w in enumerate(first_words):
        if w.lower().rstrip("s") in VERBS or w.lower() in VERBS:
            verb_idx = i
            break

    if verb_idx is not None and verb_idx < len(first_words) - 1:
        # Prefijo = sujeto + verbo (ej: "The system supports")
        prefix = " ".join(first_words[:verb_idx + 1])
    elif len(first_words) > 4:
        # Fallback: primeras 3 palabras
        prefix = " ".join(first_words[:3])
    else:
        # No hay prefijo claro, devolver partes sin prefijo
        return parts

    # Construir atoms: primera parte completa, resto con prefijo
    atoms = [first]
    for p in parts[1:]:
        # Limpiar "and" al inicio
        p = re.sub(r'^and\s+', '', p, flags=re.IGNORECASE).strip()
        if not p:
            continue
        # Si la parte no empieza con el prefijo, agregarlo
        if not p.lower().startswith(prefix.lower()):
            atoms.append(f"{prefix} {p}")
        else:
            atoms.append(p)
    return atoms


def predict_atomic(proposition: str, evidence: str, url: str) -> dict:
    """Predice TRUE/FALSE para una proposicion atomica vs evidence."""
    prompt = f"{FEW_SHOT_ATOMIC}CLAIM: {proposition.strip()}\nEVIDENCE: {evidence.strip()}\nBased on the evidence, the claim is:"
    result = call_with_logprobs(prompt, GRAMMAR_TRUE_FALSE, url, max_tokens=4)
    raw = result["content"]
    lps = get_token_logprobs(result["probs"])
    agg = aggregate_logprobs(lps, MAP_TRUE_FALSE)
    greedy = map_raw(raw, MAP_TRUE_FALSE)
    argmax = max(agg, key=agg.get) if agg else greedy
    return {"raw": raw, "greedy": greedy, "argmax": argmax, "logprobs_by_label": agg,
            "raw_logprobs": [{"token": t.get("token", ""), "logprob": t.get("logprob", 0)} for t in lps]}


def aggregate_atomic_results(atomic_results: list, expected: str) -> dict:
    """Agregador deterministico: N/N TRUE -> SUPPORTS, 0/N -> CONTRADICTS/UNRELATED, mixto -> PARTIAL."""
    n = len(atomic_results)
    if n == 0:
        return {"final": "UNRELATED", "reason": "no_atoms"}

    true_count = sum(1 for r in atomic_results if r["greedy"] == "TRUE")
    false_count = sum(1 for r in atomic_results if r["greedy"] == "FALSE")

    if true_count == n:
        final = "SUPPORTS"
        reason = f"{true_count}/{n} TRUE"
    elif true_count == 0:
        # Todos FALSE -> CONTRADICTS o UNRELATED dependiendo del overlap
        # Para el probe, asumimos CONTRADICTS si hay overlap topico
        final = "CONTRADICTS"
        reason = f"0/{n} TRUE"
    else:
        final = "PARTIAL"
        reason = f"{true_count}/{n} TRUE"

    return {"final": final, "reason": reason, "true_count": true_count, "false_count": false_count, "total": n}


def aggregate_atomic_results_argmax(atomic_results: list, expected: str) -> dict:
    """Agregador con argmax (logprobs) en lugar de greedy."""
    n = len(atomic_results)
    if n == 0:
        return {"final": "UNRELATED", "reason": "no_atoms"}

    true_count = sum(1 for r in atomic_results if r["argmax"] == "TRUE")
    false_count = sum(1 for r in atomic_results if r["argmax"] == "FALSE")

    if true_count == n:
        final = "SUPPORTS"
        reason = f"{true_count}/{n} TRUE (argmax)"
    elif true_count == 0:
        final = "CONTRADICTS"
        reason = f"0/{n} TRUE (argmax)"
    else:
        final = "PARTIAL"
        reason = f"{true_count}/{n} TRUE (argmax)"

    return {"final": final, "reason": reason, "true_count": true_count, "false_count": false_count, "total": n}


def run_fase2(cases: list, url: str) -> dict:
    """Fase 2: Atomic decomposition para casos con conjunciones."""
    print(f"\n{'='*70}", flush=True)
    print("FASE 2: ATOMIC DECOMPOSITION", flush=True)
    print(f"{'='*70}", flush=True)

    # Solo casos que tienen conjunciones (partial_support, full_support, minimal_difference, single_dimension_missing)
    decomposable_cats = {"full_support", "partial_support", "minimal_difference", "single_dimension_missing"}
    decomposable = [c for c in cases if c["category"] in decomposable_cats]

    by_cat = defaultdict(lambda: {"total": 0, "correct": 0, "predictions": defaultdict(int)})
    total_correct_greedy = 0
    total_correct_argmax = 0
    results = []

    for i, c in enumerate(decomposable):
        atoms = decompose_claim(c["claim"])
        print(f"\n  [{i+1}/{len(decomposable)}] {c['id']} ({c['category']})", flush=True)
        print(f"    Claim: {c['claim']}", flush=True)
        print(f"    Evidence: {c['evidence']}", flush=True)
        print(f"    Atoms ({len(atoms)}):", flush=True)

        atomic_results = []
        for atom in atoms:
            pred = predict_atomic(atom, c["evidence"], url)
            atomic_results.append(pred)
            print(f"      {atom[:60]:<60s} -> {pred['greedy']:<6s} (argmax: {pred['argmax']:<6s})", flush=True)

        agg_greedy = aggregate_atomic_results(atomic_results, c["expected"])
        agg_argmax = aggregate_atomic_results_argmax(atomic_results, c["expected"])

        final_g = agg_greedy["final"]
        final_a = agg_argmax["final"]
        is_ok_g = final_g == c["expected"]
        is_ok_a = final_a == c["expected"]
        if is_ok_g:
            total_correct_greedy += 1
        if is_ok_a:
            total_correct_argmax += 1

        cat = c["category"]
        by_cat[cat]["total"] += 1
        by_cat[cat]["predictions"][final_g] += 1
        if is_ok_g:
            by_cat[cat]["correct"] += 1

        status_g = "OK" if is_ok_g else f"X (got: {final_g})"
        status_a = "OK" if is_ok_a else f"X (got: {final_a})"
        print(f"    Aggregator greedy: {agg_greedy['reason']} -> {final_g} {status_g}", flush=True)
        print(f"    Aggregator argmax: {agg_argmax['reason']} -> {final_a} {status_a}", flush=True)

        results.append({
            "id": c["id"], "category": cat, "expected": c["expected"],
            "claim": c["claim"], "evidence": c["evidence"],
            "atoms": atoms,
            "atomic_results": [{"atom": a, "greedy": r["greedy"], "argmax": r["argmax"],
                                "logprobs": r["logprobs_by_label"]} for a, r in zip(atoms, atomic_results)],
            "aggregator_greedy": agg_greedy, "aggregator_argmax": agg_argmax,
            "final_greedy": final_g, "final_argmax": final_a,
            "correct_greedy": is_ok_g, "correct_argmax": is_ok_a,
        })

    n = len(decomposable)
    acc_g = total_correct_greedy / n if n > 0 else 0
    acc_a = total_correct_argmax / n if n > 0 else 0

    print(f"\n{'='*75}", flush=True)
    print(f"{'Categoria':<27} | {'Casos':<6} | {'Correctos':<10} | {'Accuracy':<9} | {'Predicciones'}", flush=True)
    print(f"{'-'*75}", flush=True)
    for cat, data in sorted(by_cat.items()):
        a = data["correct"] / data["total"] if data["total"] > 0 else 0.0
        preds_str = ", ".join(f"{k}:{v}" for k, v in sorted(data["predictions"].items()))
        print(f"{cat:<27} | {data['total']:<6} | {data['correct']:<10} | {a:>8.1%} | {preds_str}", flush=True)
    print(f"{'-'*75}", flush=True)
    print(f"{'TOTAL ATOMIC (greedy)':<27} | {n:<6} | {total_correct_greedy:<10} | {acc_g:>8.1%} |", flush=True)
    print(f"{'TOTAL ATOMIC (argmax)':<27} | {n:<6} | {total_correct_argmax:<10} | {acc_a:>8.1%} |", flush=True)
    print(f"{'='*75}", flush=True)

    # Tambien medir atomic accuracy individual
    atomic_true_correct = 0
    atomic_true_total = 0
    atomic_false_correct = 0
    atomic_false_total = 0
    for r in results:
        for ar in r["atomic_results"]:
            # Determinar ground truth: si el atom esta en el evidence -> TRUE, sino FALSE
            # Heuristica simple: buscar el atom en el evidence
            atom_lower = ar["atom"].lower()
            evidence_lower = r["evidence"].lower()
            # Normalizar para comparacion
            gt = "TRUE" if any(w in evidence_lower for w in atom_lower.split()[-3:]) else "FALSE"
            if gt == "TRUE":
                atomic_true_total += 1
                if ar["greedy"] == "TRUE":
                    atomic_true_correct += 1
            else:
                atomic_false_total += 1
                if ar["greedy"] == "FALSE":
                    atomic_false_correct += 1

    print(f"\nAtomic proposition accuracy (greedy):", flush=True)
    print(f"  TRUE (present in evidence):  {atomic_true_correct}/{atomic_true_total}", flush=True)
    print(f"  FALSE (absent from evidence): {atomic_false_correct}/{atomic_false_total}", flush=True)

    return {
        "regime": "Atomic Decomposition",
        "decomposable_cases": n,
        "greedy_accuracy": round(acc_g, 4),
        "greedy_correct": total_correct_greedy,
        "argmax_accuracy": round(acc_a, 4),
        "argmax_correct": total_correct_argmax,
        "atomic_true_accuracy": round(atomic_true_correct / max(1, atomic_true_total), 4),
        "atomic_true_correct": atomic_true_correct,
        "atomic_true_total": atomic_true_total,
        "atomic_false_accuracy": round(atomic_false_correct / max(1, atomic_false_total), 4),
        "atomic_false_correct": atomic_false_correct,
        "atomic_false_total": atomic_false_total,
        "by_category": {cat: {"total": d["total"], "correct": d["correct"],
                              "accuracy": round(d["correct"] / d["total"], 4) if d["total"] > 0 else 0.0,
                              "predictions": dict(d["predictions"])}
                        for cat, d in by_cat.items()},
        "cases": results,
    }


# ----------------------------- Main -----------------------------

def main():
    port = DEFAULT_PORT
    url = f"http://127.0.0.1:{port}"

    if not start_server(port):
        return 1

    try:
        with open(PROBE_PATH, "r", encoding="utf-8") as f:
            cases = json.load(f)["cases"]

        # Fase 1: Granularity probe directo
        fase1_reports = run_fase1(cases, url)

        # Fase 2: Atomic decomposition
        fase2_report = run_fase2(cases, url)

        # Resumen
        print(f"\n{'='*75}", flush=True)
        print("RESUMEN EXP-020: GRANULARITY PROBE", flush=True)
        print(f"{'='*75}", flush=True)
        print(f"\nFase 1: Granularity Probe directo ({len(cases)} casos)", flush=True)
        for r in fase1_reports:
            print(f"  {r['regime']:<25} greedy: {r['greedy_accuracy']:.1%} ({r['greedy_correct']}/{len(cases)})  argmax: {r['argmax_accuracy']:.1%} ({r['argmax_correct']}/{len(cases)})", flush=True)
        print(f"\nFase 2: Atomic Decomposition ({fase2_report['decomposable_cases']} casos descomponibles)", flush=True)
        print(f"  Aggregator greedy:  {fase2_report['greedy_accuracy']:.1%} ({fase2_report['greedy_correct']}/{fase2_report['decomposable_cases']})", flush=True)
        print(f"  Aggregator argmax:  {fase2_report['argmax_accuracy']:.1%} ({fase2_report['argmax_correct']}/{fase2_report['decomposable_cases']})", flush=True)
        print(f"  Atomic TRUE accuracy:  {fase2_report['atomic_true_accuracy']:.1%} ({fase2_report['atomic_true_correct']}/{fase2_report['atomic_true_total']})", flush=True)
        print(f"  Atomic FALSE accuracy: {fase2_report['atomic_false_accuracy']:.1%} ({fase2_report['atomic_false_correct']}/{fase2_report['atomic_false_total']})", flush=True)

        # Guardar
        out = OUTPUT_DIR / "bitnet_granularity_probe.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"fase1": fase1_reports, "fase2": fase2_report}, f, indent=2, ensure_ascii=False)
        print(f"\nReporte: {out}", flush=True)

    finally:
        stop_server()


if __name__ == "__main__":
    main()
