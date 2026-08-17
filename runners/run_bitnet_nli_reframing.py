"""
EXP-018 Fase 1+2: NLI Reframing + Logprobs Diagnostic para BitNet b1.58-2B-4T.

Hipotesis: BitNet tiene sesgo contra el token "SUPPORTS" como etiqueta de
clasificacion. Reformulando a NLI (TRUE/FALSE/CANNOT_TELL), tokens naturales
que el modelo maneja con confianza, se rompe la SUPPORTS wall.

Ademas: medir n_probs=4 para ver si SUPPORTS tiene masa oculta bajo el greedy.

3 regimenes NLI:
  3a: NLI 3-way (TRUE/FALSE/CANNOT_TELL)
  3b: NLI 4-way con PARTIAL (TRUE -> FULLY/PARTIALLY)
  3c: Pregunta directa (YES/NO/PARTIALLY/NOT_MENTIONED)

Todos con n_probs=4 para capturar logprobs de top-4 tokens.
"""

from __future__ import annotations

import json
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

DEFAULT_PORT = 8094
SERVER_URL = f"http://127.0.0.1:{DEFAULT_PORT}"

# ----------------------------- Grammars -----------------------------

GRAMMAR_NLI_3 = 'root ::= "TRUE" | "FALSE" | "CANNOT_TELL"'
GRAMMAR_FULLY_PARTIAL = 'root ::= "FULLY" | "PARTIALLY"'
GRAMMA_NLI_4 = 'root ::= "YES" | "NO" | "PARTIALLY" | "NOT_MENTIONED"'

# ----------------------------- Few-Shot -----------------------------

FEW_SHOT_NLI_3 = '''Task: Based on the EVIDENCE, determine if the CLAIM is TRUE, FALSE, or CANNOT_TELL.

CLAIM: The NIST CSF has five core functions
EVIDENCE: The Framework Core consists of five Functions: Identify, Detect, Protect, Respond, and Recover.
Based on the evidence, the claim is: TRUE

CLAIM: Python 4.0 was released in 2023
EVIDENCE: Python 3.12 was released on October 2, 2023. There is no Python 4.0.
Based on the evidence, the claim is: FALSE

CLAIM: The sky is blue
EVIDENCE: The NIST Cybersecurity Framework provides guidance for managing cybersecurity risk.
Based on the evidence, the claim is: CANNOT_TELL

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

FEW_SHOT_FULLY_PARTIAL = '''Task: The claim is partially supported by the evidence. Determine if it is FULLY or PARTIALLY supported.

CLAIM: The framework includes risk assessment, mitigation, and recovery guidance
EVIDENCE: The framework provides risk assessment and mitigation guidance for organizations.
Is the claim fully or partially supported? PARTIALLY

CLAIM: The NIST CSF has five core functions
EVIDENCE: The Framework Core consists of five Functions: Identify, Detect, Protect, Respond, and Recover.
Is the claim fully or partially supported? FULLY

'''

# ----------------------------- Mapping -----------------------------

MAP_NLI_3 = {"TRUE": "SUPPORTS", "FALSE": "CONTRADICTS", "CANNOT_TELL": "UNRELATED"}
MAP_NLI_4 = {"YES": "SUPPORTS", "NO": "CONTRADICTS", "PARTIALLY": "PARTIAL", "NOT_MENTIONED": "UNRELATED"}
MAP_FULLY_PARTIAL = {"FULLY": "SUPPORTS", "PARTIALLY": "PARTIAL"}

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
        exe,
        "-m", model,
        "--host", "127.0.0.1",
        "--port", str(port),
        "-t", "4",
        "-c", "2048",
        "-ngl", "0",
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


# ----------------------------- Predictors with logprobs -----------------------------

def call_with_logprobs(prompt: str, grammar: str, url: str, max_tokens: int = 6) -> dict:
    """Llamada a /completion con n_probs=4 para capturar top-4 tokens."""
    try:
        resp = requests.post(
            f"{url}/completion",
            json={
                "prompt": prompt,
                "stream": False,
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "repeat_penalty": 1.0,
                "grammar": grammar,
                "n_probs": 4,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data.get("content", "").strip()
        # completion_probabilities es una lista de posiciones, cada una con top_logprobs
        probs = data.get("completion_probabilities", [])
        return {"content": content, "probs": probs}
    except Exception as exc:
        return {"content": f"ERROR: {exc}", "probs": []}


def extract_first_token_logprobs(probs: list) -> List[dict]:
    """Extrae los top_logprobs del primer token generado."""
    if not probs:
        return []
    first = probs[0]
    return first.get("top_logprobs", [])


def predict_nli_3(claim: str, evidence: str, url: str = SERVER_URL) -> dict:
    prompt = f"{FEW_SHOT_NLI_3}CLAIM: {claim.strip()}\nEVIDENCE: {evidence.strip()}\nBased on the evidence, the claim is:"
    result = call_with_logprobs(prompt, GRAMMAR_NLI_3, url, max_tokens=4)
    raw = result["content"]
    # Normalizar: puede generar "TRUE" o " TRUE" o "TRUE\n"
    for key in MAP_NLI_3:
        if key in raw:
            mapped = MAP_NLI_3[key]
            break
    else:
        mapped = "UNRELATED"
    logprobs = extract_first_token_logprobs(result["probs"])
    return {
        "raw": raw,
        "mapped": mapped,
        "logprobs": [{"token": t.get("token", ""), "logprob": t.get("logprob", 0)} for t in logprobs],
    }


def predict_nli_4(claim: str, evidence: str, url: str = SERVER_URL) -> dict:
    prompt = f"{FEW_SHOT_NLI_4}CLAIM: {claim.strip()}\nEVIDENCE: {evidence.strip()}\nIs the claim supported by the evidence?"
    result = call_with_logprobs(prompt, GRAMMA_NLI_4, url, max_tokens=6)
    raw = result["content"]
    for key in MAP_NLI_4:
        if key in raw:
            mapped = MAP_NLI_4[key]
            break
    else:
        mapped = "UNRELATED"
    logprobs = extract_first_token_logprobs(result["probs"])
    return {
        "raw": raw,
        "mapped": mapped,
        "logprobs": [{"token": t.get("token", ""), "logprob": t.get("logprob", 0)} for t in logprobs],
    }


def predict_nli_cascading(claim: str, evidence: str, url: str = SERVER_URL) -> dict:
    """NLI 3-way + si TRUE, segunda pasada FULLY/PARTIALLY."""
    # Paso 1
    prompt1 = f"{FEW_SHOT_NLI_3}CLAIM: {claim.strip()}\nEVIDENCE: {evidence.strip()}\nBased on the evidence, the claim is:"
    result1 = call_with_logprobs(prompt1, GRAMMAR_NLI_3, url, max_tokens=4)
    raw1 = result1["content"]
    logprobs1 = extract_first_token_logprobs(result1["probs"])

    for key in MAP_NLI_3:
        if key in raw1:
            step1 = key
            break
    else:
        step1 = "CANNOT_TELL"

    if step1 == "TRUE":
        # Paso 2: FULLY o PARTIALLY
        prompt2 = f"{FEW_SHOT_FULLY_PARTIAL}CLAIM: {claim.strip()}\nEVIDENCE: {evidence.strip()}\nIs the claim fully or partially supported?"
        result2 = call_with_logprobs(prompt2, GRAMMAR_FULLY_PARTIAL, url, max_tokens=4)
        raw2 = result2["content"]
        logprobs2 = extract_first_token_logprobs(result2["probs"])
        for key in MAP_FULLY_PARTIAL:
            if key in raw2:
                mapped = MAP_FULLY_PARTIAL[key]
                break
        else:
            mapped = "SUPPORTS"
        return {
            "raw": f"{raw1} -> {raw2}",
            "mapped": mapped,
            "logprobs": [{"token": t.get("token", ""), "logprob": t.get("logprob", 0)} for t in logprobs1],
            "logprobs_step2": [{"token": t.get("token", ""), "logprob": t.get("logprob", 0)} for t in logprobs2],
        }
    else:
        return {
            "raw": raw1,
            "mapped": MAP_NLI_3[step1],
            "logprobs": [{"token": t.get("token", ""), "logprob": t.get("logprob", 0)} for t in logprobs1],
            "logprobs_step2": [],
        }


# ----------------------------- Evaluation -----------------------------

def evaluate_regime(name: str, predictor_fn, cases: List[dict], url: str) -> dict:
    print("\n" + "=" * 70, flush=True)
    print(f"EVALUANDO: {name}", flush=True)
    print("=" * 70, flush=True)

    by_category = defaultdict(lambda: {"total": 0, "correct": 0, "predictions": defaultdict(int)})
    total_correct = 0
    t0 = time.time()
    results = []

    for i, c in enumerate(cases):
        pred_data = predictor_fn(c["claim"], c["evidence"], url=url)
        pred = pred_data["mapped"]
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
            "raw": pred_data["raw"],
            "correct": is_ok,
            "logprobs": pred_data["logprobs"],
            "logprobs_step2": pred_data.get("logprobs_step2", []),
        })
        status = "OK" if is_ok else f"X (got: {pred}, raw: {pred_data['raw'][:20]})"
        print(f"  [{i+1:2d}/55] {c['id']:<8s} ({cat:<22s}) exp: {c['expected']:<11s} -> {status}", flush=True)

    wall = time.time() - t0
    acc = total_correct / len(cases)

    print("-" * 75, flush=True)
    print(f"{'Categoria':<25} | {'Casos':<6} | {'Correctos':<10} | {'Accuracy':<9} | {'Predicciones'}", flush=True)
    print("-" * 75, flush=True)
    for cat, data in sorted(by_category.items()):
        tot = data["total"]
        corr = data["correct"]
        a = corr / tot if tot > 0 else 0.0
        preds_str = ", ".join(f"{k}:{v}" for k, v in sorted(data["predictions"].items()))
        print(f"{cat:<25} | {tot:<6} | {corr:<10} | {a:>8.1%} | {preds_str}", flush=True)
    print("-" * 75, flush=True)
    print(f"{'TOTAL ' + name[:25]:<25} | {len(cases):<6} | {total_correct:<10} | {acc:>8.1%} | Wall: {wall/60:.1f}m", flush=True)
    print("=" * 70, flush=True)

    return {
        "regime_name": name,
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


# ----------------------------- Main -----------------------------

def main():
    port = DEFAULT_PORT
    url = f"http://127.0.0.1:{port}"

    if not start_server(port):
        return 1

    try:
        with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
            cases = json.load(f)["cases"]

        experiments = [
            ("Regimen 3a: NLI 3-way (TRUE/FALSE/CANNOT_TELL)", predict_nli_3),
            ("Regimen 3b: NLI Cascading (TRUE -> FULLY/PARTIALLY)", predict_nli_cascading),
            ("Regimen 3c: NLI 4-way (YES/NO/PARTIALLY/NOT_MENTIONED)", predict_nli_4),
        ]

        all_reports = []
        for name, fn in experiments:
            rep = evaluate_regime(name, fn, cases, url)
            all_reports.append(rep)
            out = OUTPUT_DIR / f"bitnet_nli_{name.split(':')[0].strip().replace(' ', '_').lower()}.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(rep, f, indent=2, ensure_ascii=False)
            print(f"  Guardado: {out}\n", flush=True)

        # Resumen
        print("\n" + "=" * 75, flush=True)
        print("RESUMEN NLI REFRAMING - BITNET b1.58-2B-4T", flush=True)
        print("=" * 75, flush=True)
        print(f"{'Regimen':<50} | {'Accuracy':<10} | {'Correctos':<10}", flush=True)
        print("-" * 75, flush=True)
        for r in all_reports:
            print(f"{r['regime_name']:<50} | {r['overall_accuracy']:>8.1%} | {r['correct_count']:>5}/55", flush=True)
        print("=" * 75, flush=True)

        # Guardar todo junto
        out_all = OUTPUT_DIR / "bitnet_nli_reframing_all.json"
        with open(out_all, "w", encoding="utf-8") as f:
            json.dump(all_reports, f, indent=2, ensure_ascii=False)
        print(f"\nReporte completo: {out_all}", flush=True)

    finally:
        stop_server()


if __name__ == "__main__":
    main()
