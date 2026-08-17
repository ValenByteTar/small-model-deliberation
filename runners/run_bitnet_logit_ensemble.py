"""
EXP-018 Fase 3: Logit Ensemble para BitNet b1.58-2B-4T.

Suma logprobs por etiqueta canonica (SUPPORTS/CONTRADICTS/UNRELATED/PARTIAL)
across multiples regimenes usando logsumexp. El vote fallaba porque majority
washes out; logit sum preserva la confianza.

Regimenes combinados:
  - fs0: Few-Shot GBNF (SUPPORTS/CONTRADICTS/UNRELATED/PARTIAL)
  - bin: Binary Cascading (Relevancia -> Polaridad)
  - nli3c: NLI 4-way (YES/NO/PARTIALLY/NOT_MENTIONED)

Todos con n_probs=4 para capturar top-4 tokens y sus logprobs.

Tambien prueba un fix del grammar NLI con espacios leading.
"""

from __future__ import annotations

import json
import math
import os
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

BENCHMARK_PATH = ROOT / "benchmarks" / "semantic_assessment_v2.json"
OUTPUT_DIR = ROOT / "results" / "raw"

DEFAULT_PORT = 8095
SERVER_URL = f"http://127.0.0.1:{DEFAULT_PORT}"

# ----------------------------- Grammars -----------------------------

GRAMMAR_RELATION = 'root ::= relation\nrelation ::= "SUPPORTS" | "CONTRADICTS" | "UNRELATED" | "PARTIAL"'
GRAMMAR_YES_NO = 'root ::= "YES" | "NO"'
GRAMMAR_POLARITY = 'root ::= "SUPPORTS" | "CONTRADICTS" | "PARTIAL"'
# Fix: incluir variantes con espacio leading (tokens reales del modelo)
GRAMMAR_NLI_4_FIXED = 'root ::= "YES" | "NO" | "PARTIALLY" | "NOT_MENTIONED" | " YES" | " NO" | " PARTIALLY" | " NOT_MENTIONED" | " Yes" | " No" | " yes" | " no"'

# ----------------------------- Few-Shot -----------------------------

FEW_SHOT_BASE = '''Task: Classify the relationship between CLAIM and EVIDENCE as SUPPORTS, CONTRADICTS, UNRELATED, or PARTIAL.

CLAIM: The NIST CSF has five core functions
EVIDENCE: The Framework Core consists of five Functions: Identify, Detect, Protect, Respond, and Recover.
RELATION: SUPPORTS

CLAIM: Python 4.0 was released in 2023
EVIDENCE: Python 3.12 was released on October 2, 2023. There is no Python 4.0.
RELATION: CONTRADICTS

CLAIM: The sky is blue
EVIDENCE: The NIST Cybersecurity Framework provides guidance for managing cybersecurity risk.
RELATION: UNRELATED

CLAIM: ISO 27001 requires a specific risk assessment methodology
EVIDENCE: The standard requires a risk assessment process but does not specify a particular methodology.
RELATION: PARTIAL

'''

FEW_SHOT_NLI_4 = '''Task: Based on the EVIDENCE, answer if the CLAIM is YES, NO, PARTIALLY, or NOT_MENTIONED.

CLAIM: The NIST CSF has five core functions
EVIDENCE: The Framework Core consists of five Functions: Identify, Detect, Protect, Respond, and Recover.
Is the claim supported by the evidence? YES

CLAIM: Python 4.0 was released in 2023
EVIDENCE: Python 3.12 was released on October 2, 2023. There is no Python 4.0.
Is the claim supported by the evidence? NO

CLAIM: The sky is blue
EVIDENCE: The NIST Cybersecurity Framework provides guidance for managing cybersecurity risk.
Is the claim supported by the evidence? NOT_MENTIONED

CLAIM: ISO 27001 requires a specific risk assessment methodology
EVIDENCE: The standard requires a risk assessment process but does not specify a particular methodology.
Is the claim supported by the evidence? PARTIALLY

'''

# ----------------------------- Token -> Label Mapping -----------------------------

# Mapear tokens (con variantes de espacio/case) a etiquetas canonicas
TOKEN_MAP_RELATION = {
    "SUPPORTS": "SUPPORTS", " SUPPORTS": "SUPPORTS", "supports": "SUPPORTS", " supports": "SUPPORTS",
    "CONTRADICTS": "CONTRADICTS", " CONTRADICTS": "CONTRADICTS", "contradicts": "CONTRADICTS", " contradicts": "CONTRADICTS",
    "UNRELATED": "UNRELATED", " UNRELATED": "UNRELATED", "unrelated": "UNRELATED", " unrelated": "UNRELATED",
    "PARTIAL": "PARTIAL", " PARTIAL": "PARTIAL", "partial": "PARTIAL", " partial": "PARTIAL",
}

TOKEN_MAP_NLI4 = {
    "YES": "SUPPORTS", " YES": "SUPPORTS", "yes": "SUPPORTS", " Yes": "SUPPORTS", " yes": "SUPPORTS",
    "NO": "CONTRADICTS", " NO": "CONTRADICTS", "no": "CONTRADICTS", " No": "CONTRADICTS", " no": "CONTRADICTS",
    "PARTIALLY": "PARTIAL", " PARTIALLY": "PARTIAL", "partially": "PARTIAL", " Partially": "PARTIAL",
    "NOT_MENTIONED": "UNRELATED", " NOT_MENTIONED": "UNRELATED", "not_mentioned": "UNRELATED", " Not_mentioned": "UNRELATED",
    "NOT": "UNRELATED", " NOT": "UNRELATED", "not": "UNRELATED", " Not": "UNRELATED",
}

TOKEN_MAP_POLARITY = {
    "SUPPORTS": "SUPPORTS", " SUPPORTS": "SUPPORTS",
    "CONTRADICTS": "CONTRADICTS", " CONTRADICTS": "CONTRADICTS",
    "PARTIAL": "PARTIAL", " PARTIAL": "PARTIAL",
}

TOKEN_MAP_YES_NO = {
    "YES": "relevant", " YES": "relevant", "yes": "relevant", " Yes": "relevant",
    "NO": "irrelevant", " NO": "irrelevant", "no": "irrelevant", " No": "irrelevant",
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


# ----------------------------- Logprob Extraction -----------------------------

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
    """Extrae top_logprobs del primer token generado."""
    if not probs:
        return []
    return probs[0].get("top_logprobs", [])


def aggregate_logprobs_by_label(token_logprobs: list, token_map: dict) -> Dict[str, float]:
    """Agrega logprobs por etiqueta canonica usando logsumexp.
    Si un token no mapea a ninguna etiqueta, se ignora.
    """
    by_label = defaultdict(list)
    for tl in token_logprobs:
        tok = tl.get("token", "")
        lp = tl.get("logprob", -999)
        label = token_map.get(tok) or token_map.get(tok.strip()) or token_map.get(tok.lower()) or token_map.get(" " + tok.strip().lower())
        if label:
            by_label[label].append(lp)

    # Logsumexp por etiqueta
    result = {}
    for label, lps in by_label.items():
        if lps:
            max_lp = max(lps)
            result[label] = max_lp + math.log(sum(math.exp(lp - max_lp) for lp in lps))
    return result


# ----------------------------- Regime Predictors (with logprobs) -----------------------------

def predict_fs0(claim: str, evidence: str, url: str) -> dict:
    prompt = f"{FEW_SHOT_BASE}CLAIM: {claim.strip()}\nEVIDENCE: {evidence.strip()}\nRELATION:"
    result = call_with_logprobs(prompt, GRAMMAR_RELATION, url, max_tokens=6)
    raw = result["content"]
    lps = get_token_logprobs(result["probs"])
    agg = aggregate_logprobs_by_label(lps, TOKEN_MAP_RELATION)
    greedy = TOKEN_MAP_RELATION.get(raw, TOKEN_MAP_RELATION.get(raw.strip(), "UNRELATED"))
    return {"raw": raw, "greedy": greedy, "logprobs_by_label": agg,
            "raw_logprobs": [{"token": t.get("token", ""), "logprob": t.get("logprob", 0)} for t in lps]}


def predict_nli4(claim: str, evidence: str, url: str) -> dict:
    prompt = f"{FEW_SHOT_NLI_4}CLAIM: {claim.strip()}\nEVIDENCE: {evidence.strip()}\nIs the claim supported by the evidence?"
    result = call_with_logprobs(prompt, GRAMMAR_NLI_4_FIXED, url, max_tokens=6)
    raw = result["content"]
    lps = get_token_logprobs(result["probs"])
    agg = aggregate_logprobs_by_label(lps, TOKEN_MAP_NLI4)
    # greedy: mapear raw a etiqueta
    greedy = "UNRELATED"
    for key, val in TOKEN_MAP_NLI4.items():
        if key in raw:
            greedy = val
            break
    return {"raw": raw, "greedy": greedy, "logprobs_by_label": agg,
            "raw_logprobs": [{"token": t.get("token", ""), "logprob": t.get("logprob", 0)} for t in lps]}


def predict_bin_cascading(claim: str, evidence: str, url: str) -> dict:
    """Binary cascading: Relevancia YES/NO -> si YES, SUPPORTS/CONTRADICTS/PARTIAL."""
    # Paso 1: Relevancia
    prompt_p1 = f'''Task: Are the CLAIM and EVIDENCE discussing the same subject or domain? Answer YES or NO.

CLAIM: The NIST CSF has five core functions
EVIDENCE: The Framework Core consists of five Functions: Identify, Detect, Protect, Respond, and Recover.
ANSWER: YES

CLAIM: The sky is blue
EVIDENCE: The NIST Cybersecurity Framework provides guidance for managing cybersecurity risk.
ANSWER: NO

CLAIM: {claim.strip()}
EVIDENCE: {evidence.strip()}
ANSWER:'''
    result1 = call_with_logprobs(prompt_p1, GRAMMAR_YES_NO, url, max_tokens=4)
    raw1 = result1["content"]
    lps1 = get_token_logprobs(result1["probs"])
    agg1 = aggregate_logprobs_by_label(lps1, TOKEN_MAP_YES_NO)

    # Si NO es relevante -> UNRELATED
    # Pero en logit ensemble, contribuimos al pool de logprobs
    # Para el ensemble, mapeamos: si relevante -> contribuimos a SUPPORTS/CONTRADICTS/PARTIAL
    # si no relevante -> contribuimos a UNRELATED

    # Paso 2: si YES, polaridad
    prompt_p2 = f'''Task: Since CLAIM and EVIDENCE are related, classify the relationship as SUPPORTS, CONTRADICTS, or PARTIAL.

CLAIM: The NIST CSF has five core functions
EVIDENCE: The Framework Core consists of five Functions: Identify, Detect, Protect, Respond, and Recover.
RELATION: SUPPORTS

CLAIM: Python 4.0 was released in 2023
EVIDENCE: Python 3.12 was released on October 2, 2023. There is no Python 4.0.
RELATION: CONTRADICTS

CLAIM: ISO 27001 requires a specific risk assessment methodology
EVIDENCE: The standard requires a risk assessment process but does not specify a particular methodology.
RELATION: PARTIAL

CLAIM: {claim.strip()}
EVIDENCE: {evidence.strip()}
RELATION:'''
    result2 = call_with_logprobs(prompt_p2, GRAMMAR_POLARITY, url, max_tokens=6)
    raw2 = result2["content"]
    lps2 = get_token_logprobs(result2["probs"])
    agg2 = aggregate_logprobs_by_label(lps2, TOKEN_MAP_POLARITY)

    # Logprobs combinados: relevancia * polaridad
    # Si relevancia dice YES (logprob alto), contribuimos agg2 al pool
    # Si relevancia dice NO, contribuimos UNRELATED con la confianza de NO
    relevant_lp = agg1.get("relevant", -999)
    irrelevant_lp = agg1.get("irrelevant", -999)

    combined = {}
    # UNRELATED recibe el logprob de "irrelevant"
    if irrelevant_lp > -999:
        combined["UNRELATED"] = irrelevant_lp
    # SUPPORTS/CONTRADICTS/PARTIAL reciben logprob de polaridad + logprob de "relevant"
    if relevant_lp > -999:
        for label, lp in agg2.items():
            combined[label] = lp + relevant_lp

    # Greedy: si YES -> polaridad, si NO -> UNRELATED
    if "YES" in raw1 or "yes" in raw1 or " Yes" in raw1:
        greedy = TOKEN_MAP_POLARITY.get(raw2, TOKEN_MAP_POLARITY.get(raw2.strip(), "SUPPORTS"))
    else:
        greedy = "UNRELATED"

    return {"raw": f"{raw1} -> {raw2}", "greedy": greedy, "logprobs_by_label": combined,
            "raw_logprobs": [{"token": t.get("token", ""), "logprob": t.get("logprob", 0)} for t in lps1],
            "raw_logprobs_step2": [{"token": t.get("token", ""), "logprob": t.get("logprob", 0)} for t in lps2]}


# ----------------------------- Logit Ensemble -----------------------------

def logit_ensemble(claim: str, evidence: str, url: str) -> dict:
    """Corre 3 regimenes y suma logprobs por etiqueta canonica."""
    r1 = predict_fs0(claim, evidence, url)
    r2 = predict_nli4(claim, evidence, url)
    r3 = predict_bin_cascading(claim, evidence, url)

    # Sumar logprobs por etiqueta (logsumexp across regimenes)
    all_labels = set()
    for r in [r1, r2, r3]:
        all_labels.update(r["logprobs_by_label"].keys())

    ensemble_logprobs = {}
    for label in all_labels:
        lps = []
        for r in [r1, r2, r3]:
            if label in r["logprobs_by_label"]:
                lps.append(r["logprobs_by_label"][label])
        if lps:
            max_lp = max(lps)
            ensemble_logprobs[label] = max_lp + math.log(sum(math.exp(lp - max_lp) for lp in lps))

    # Argmax
    if ensemble_logprobs:
        final = max(ensemble_logprobs, key=ensemble_logprobs.get)
    else:
        final = "UNRELATED"

    return {
        "final": final,
        "ensemble_logprobs": ensemble_logprobs,
        "regimes": {
            "fs0": {"greedy": r1["greedy"], "logprobs_by_label": r1["logprobs_by_label"]},
            "nli4": {"greedy": r2["greedy"], "logprobs_by_label": r2["logprobs_by_label"]},
            "bin": {"greedy": r3["greedy"], "logprobs_by_label": r3["logprobs_by_label"]},
        },
    }


# ----------------------------- Main -----------------------------

def main():
    port = DEFAULT_PORT
    url = f"http://127.0.0.1:{port}"

    if not start_server(port):
        return 1

    try:
        with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
            cases = json.load(f)["cases"]

        n = len(cases)
        print("=" * 70, flush=True)
        print(f"LOGIT ENSEMBLE - BITNET b1.58-2B-4T ({n} casos)", flush=True)
        print("Regimenes: fs0 + nli4 + bin_cascading | Agregador: logsumexp", flush=True)
        print("=" * 70, flush=True)

        by_category = defaultdict(lambda: {"total": 0, "correct": 0, "predictions": defaultdict(int)})
        total_correct = 0
        t0 = time.time()
        results = []

        for i, c in enumerate(cases):
            ens = logit_ensemble(c["claim"], c["evidence"], url)
            pred = ens["final"]
            is_ok = pred == c["expected"]
            if is_ok:
                total_correct += 1

            cat = c["category"]
            by_category[cat]["total"] += 1
            by_category[cat]["predictions"][pred] += 1
            if is_ok:
                by_category[cat]["correct"] += 1

            results.append({
                "id": c["id"],
                "category": cat,
                "expected": c["expected"],
                "predicted": pred,
                "correct": is_ok,
                "ensemble_logprobs": ens["ensemble_logprobs"],
                "regime_greedys": {k: v["greedy"] for k, v in ens["regimes"].items()},
                "regime_logprobs": {k: v["logprobs_by_label"] for k, v in ens["regimes"].items()},
            })

            status = "OK" if is_ok else f"X (got: {pred})"
            greedys = "/".join(f"{k}={v['greedy'][:4]}" for k, v in ens["regimes"].items())
            print(f"  [{i+1:2d}/{n}] {c['id']:<8s} ({cat:<22s}) exp: {c['expected']:<11s} -> {status} [{greedys}]", flush=True)

        wall = time.time() - t0
        acc = total_correct / n

        print("\n" + "=" * 75, flush=True)
        print(f"{'Categoria':<25} | {'Casos':<6} | {'Correctos':<10} | {'Accuracy':<9} | {'Predicciones'}", flush=True)
        print("-" * 75, flush=True)
        for cat, data in sorted(by_category.items()):
            tot = data["total"]
            corr = data["correct"]
            a = corr / tot if tot > 0 else 0.0
            preds_str = ", ".join(f"{k}:{v}" for k, v in sorted(data["predictions"].items()))
            print(f"{cat:<25} | {tot:<6} | {corr:<10} | {a:>8.1%} | {preds_str}", flush=True)
        print("-" * 75, flush=True)
        print(f"{'TOTAL LOGIT ENSEMBLE':<25} | {n:<6} | {total_correct:<10} | {acc:>8.1%} | Wall: {wall/60:.1f}m", flush=True)
        print("=" * 75, flush=True)

        # Comparacion con regimenes individuales
        print("\nComparacion:", flush=True)
        print(f"  fs0 single (EXP-017):       29.1% (16/55)", flush=True)
        print(f"  nli4 single (EXP-018):      29.1% (16/55)", flush=True)
        print(f"  bin cascade single (EXP-017): 27.3% (15/55)", flush=True)
        print(f"  LOGIT ENSEMBLE:              {acc:.1%} ({total_correct}/55)", flush=True)
        target = 28
        print(f"  Target 50%:                  {target}/55 {'ALCANZADO' if total_correct >= target else 'NO alcanzado'}", flush=True)

        # Guardar
        report = {
            "model": "BitNet-b1.58-2B-4T",
            "regime": "Logit Ensemble (fs0 + nli4 + bin_cascading, logsumexp aggregation)",
            "benchmark": "semantic_assessment_v2.json",
            "total_cases": n,
            "overall_accuracy": round(acc, 4),
            "correct_count": total_correct,
            "wall_time_s": round(wall, 1),
            "by_category": {
                cat: {
                    "total": d["total"],
                    "correct": d["correct"],
                    "accuracy": round(d["correct"] / d["total"], 4) if d["total"] > 0 else 0.0,
                    "predictions": dict(d["predictions"]),
                }
                for cat, d in by_category.items()
            },
            "cases": results,
        }

        out = OUTPUT_DIR / "bitnet_logit_ensemble.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nReporte: {out}", flush=True)

    finally:
        stop_server()


if __name__ == "__main__":
    main()
