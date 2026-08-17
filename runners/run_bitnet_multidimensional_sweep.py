"""
EXP-017b: Barrido Multidimensional Exhaustivo para BitNet b1.58-2B-4T.

Evalúa sistemáticamente:
1. Regimen 0: Zero-Shot GBNF puro (temp=0.0)
2. Regimen 0 + Temp: Zero-Shot GBNF (temp=0.2, top_k=20)
3. Regimen 1 + Temp: Few-Shot GBNF (temp=0.2, top_k=20)
4. Regimen 2: Binary Cascading / One-vs-All (YES/NO Relevance -> Polarity)

Genera el perfil cognitivo de cada regimen y lo contrasta contra los modelos de referencia.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

ROOT = Path(__file__).resolve().parent.parent

BITNET_ROOT = os.environ.get("BITNET_ROOT", os.path.expanduser("~/BitNet"))
MODEL_PATH = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
SERVER_EXE = "build/bin/Release/llama-server.exe"

BENCHMARK_PATH = ROOT / "benchmarks" / "semantic_assessment_v2.json"
OUTPUT_DIR = ROOT / "results" / "raw"

DEFAULT_PORT = 8093
SERVER_URL = f"http://127.0.0.1:{DEFAULT_PORT}"

GRAMMAR_RELATION = '''root ::= relation
relation ::= "SUPPORTS" | "CONTRADICTS" | "UNRELATED" | "PARTIAL"'''

GRAMMAR_YES_NO = 'root ::= "YES" | "NO"'
GRAMMAR_POLARITY = 'root ::= "SUPPORTS" | "CONTRADICTS" | "PARTIAL"'

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


# ----------------------------- Predictors -----------------------------

def predict_zero_shot(claim: str, evidence: str, temp: float = 0.0, top_k: int = 1, url: str = SERVER_URL) -> str:
    prompt = f"Task: Classify relationship between CLAIM and EVIDENCE as SUPPORTS, CONTRADICTS, UNRELATED, or PARTIAL.\n\nCLAIM: {claim.strip()}\nEVIDENCE: {evidence.strip()}\nRELATION:"
    payload = {
        "prompt": prompt,
        "stream": False,
        "temperature": temp,
        "top_k": top_k,
        "max_tokens": 6,
        "repeat_penalty": 1.0,
        "grammar": GRAMMAR_RELATION,
    }
    r = requests.post(f"{url}/completion", json=payload, timeout=60)
    return r.json().get("content", "").strip()


def predict_few_shot_temp(claim: str, evidence: str, temp: float = 0.2, top_k: int = 20, url: str = SERVER_URL) -> str:
    prompt = f"{FEW_SHOT_BASE}CLAIM: {claim.strip()}\nEVIDENCE: {evidence.strip()}\nRELATION:"
    payload = {
        "prompt": prompt,
        "stream": False,
        "temperature": temp,
        "top_k": top_k,
        "max_tokens": 6,
        "repeat_penalty": 1.0,
        "grammar": GRAMMAR_RELATION,
    }
    r = requests.post(f"{url}/completion", json=payload, timeout=60)
    return r.json().get("content", "").strip()


def predict_binary_cascading(claim: str, evidence: str, url: str = SERVER_URL) -> str:
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
    r1 = requests.post(f"{url}/completion", json={
        "prompt": prompt_p1, "stream": False, "temperature": 0.0, "max_tokens": 4, "grammar": GRAMMAR_YES_NO, "repeat_penalty": 1.0
    }, timeout=60)
    ans1 = r1.json().get("content", "").strip()

    if ans1 == "NO":
        return "UNRELATED"

    # Paso 2: Polaridad
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
    r2 = requests.post(f"{url}/completion", json={
        "prompt": prompt_p2, "stream": False, "temperature": 0.0, "max_tokens": 6, "grammar": GRAMMAR_POLARITY, "repeat_penalty": 1.0
    }, timeout=60)
    return r2.json().get("content", "").strip()


# ----------------------------- Evaluation Runner -----------------------------

def evaluate_regime(name: str, predictor_fn, cases: List[dict], url: str) -> dict:
    print("\n" + "=" * 70, flush=True)
    print(f"EVALUANDO: {name}", flush=True)
    print("=" * 70, flush=True)

    by_category = defaultdict(lambda: {"total": 0, "correct": 0, "predictions": defaultdict(int)})
    total_correct = 0
    t0 = time.time()

    results = []
    for i, c in enumerate(cases):
        pred = predictor_fn(c["claim"], c["evidence"], url=url)
        is_ok = (pred == c["expected"])
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
        })
        status = "OK" if is_ok else f"X (got: {pred})"
        print(f"  [{i+1:2d}/55] {c['id']:<8s} ({cat:<22s}) exp: {c['expected']:<11s} -> {status}", flush=True)

    wall = time.time() - t0
    acc = total_correct / len(cases)

    print("-" * 75, flush=True)
    print(f"Total {name}: {total_correct}/55 ({acc:.1%}) in {wall/60:.1f}m", flush=True)
    print("=" * 75, flush=True)

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


def main():
    parser = argparse.ArgumentParser(description="BitNet Multidimensional Sweep")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    port = args.port
    url = f"http://127.0.0.1:{port}"

    if not start_server(port):
        return 1

    try:
        with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
            cases = json.load(f)["cases"]

        experiments = [
            ("Regimen 0: Zero-Shot GBNF (temp=0.0)", lambda cl, ev, url: predict_zero_shot(cl, ev, temp=0.0, top_k=1, url=url)),
            ("Regimen 0: Zero-Shot GBNF (temp=0.2, top_k=20)", lambda cl, ev, url: predict_zero_shot(cl, ev, temp=0.2, top_k=20, url=url)),
            ("Regimen 1: Few-Shot GBNF (temp=0.2, top_k=20)", lambda cl, ev, url: predict_few_shot_temp(cl, ev, temp=0.2, top_k=20, url=url)),
            ("Regimen 2: Binary Cascading / One-vs-All", lambda cl, ev, url: predict_binary_cascading(cl, ev, url=url)),
        ]

        all_reports = []
        for name, fn in experiments:
            rep = evaluate_regime(name, fn, cases, url)
            all_reports.append(rep)

        out_path = OUTPUT_DIR / "bitnet_multidimensional_sweep.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_reports, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 75, flush=True)
        print("RESUMEN GENERAL DEL BARRIDO MULTIDIMENSIONAL (BITNET 2B)", flush=True)
        print("=" * 75, flush=True)
        print(f"{'Regimen / Configuracion':<45} | {'Accuracy':<10} | {'Correctos':<10}")
        print("-" * 75, flush=True)
        for r in all_reports:
            print(f"{r['regime_name']:<45} | {r['overall_accuracy']:>8.1%} | {r['correct_count']:>5}/55", flush=True)
        print("=" * 75, flush=True)

    finally:
        stop_server()


if __name__ == "__main__":
    main()
