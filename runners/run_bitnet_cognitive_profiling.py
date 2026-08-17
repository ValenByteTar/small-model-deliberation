"""
EXP-017: Perfil Cognitivo y Barrido de Regimenes de Inferencia para BitNet b1.58-2B-4T.

Objetivo:
  Determinar si existe algun regimen de inferencia (minima carga cognitiva,
  GBNF constrained decoding a nivel de 1 token, few-shot calibrado, sin JSON)
  en el que BitNet 2B conserve capacidad semantica util o especializada.

Evaluacion:
  - single: 1 worker (neutral)
  - ensemble_2: 2 workers (entailment, skeptical) + majority vote
  - ensemble_4: 4 workers (entailment, skeptical, contradiction, neutral) + majority vote

Genera una matriz comparativa detallada por las 10 categorias diagnosticas.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

ROOT = Path(__file__).resolve().parent.parent

BITNET_ROOT = os.environ.get("BITNET_ROOT", os.path.expanduser("~/BitNet"))
MODEL_PATH = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
SERVER_EXE = "build/bin/Release/llama-server.exe"

BENCHMARK_PATH = ROOT / "benchmarks" / "semantic_assessment_v2.json"
OUTPUT_DIR = ROOT / "results" / "raw"

DEFAULT_PORT = 8092
SERVER_URL = f"http://127.0.0.1:{DEFAULT_PORT}"

GRAMMAR_RELATION = '''root ::= relation
relation ::= "SUPPORTS" | "CONTRADICTS" | "UNRELATED" | "PARTIAL"'''

# ----------------------------- Worker Prompts -----------------------------
# Cada worker tiene un rol especializado pero comparte el mismo few-shot base.
# Los prompts estan disenados para BitNet: minimalistas, sin JSON, sin explicacion.

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

# Roles especializados (prefijo breve antes del few-shot base)
WORKER_ROLES = {
    "neutral": "You are a neutral analyst. Classify objectively.\n\n",
    "entailment": "You are an entailment analyst. Focus on logical support.\n\n",
    "skeptical": "You are a skeptical analyst. Be conservative: only SUPPORTS if clearly proven.\n\n",
    "contradiction": "You are a contradiction analyst. Actively look for conflicts.\n\n",
}

WORKER_ORDER_2 = ["entailment", "skeptical"]
WORKER_ORDER_4 = ["entailment", "skeptical", "contradiction", "neutral"]

# ----------------------------- llama-server control -----------------------------

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
        "--temp", "0.0",
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

def _call_llama_server(prompt: str, url: str, max_tokens: int = 6) -> Tuple[str, float]:
    """Llamadaunica a llama-server con GBNF grammar constraint."""
    t0 = time.time()
    try:
        resp = requests.post(
            f"{url}/completion",
            json={
                "prompt": prompt,
                "stream": False,
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "repeat_penalty": 1.0,
                "grammar": GRAMMAR_RELATION,
            },
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json().get("content", "").strip()
        return raw, time.time() - t0
    except Exception as exc:
        return f"ERROR: {exc}", time.time() - t0


def predict_single(claim: str, evidence: str, url: str = SERVER_URL) -> Tuple[str, float, List[str]]:
    """Single worker (neutral). Returns (final_relation, total_latency, [worker_pred])."""
    prompt = f"{WORKER_ROLES['neutral']}{FEW_SHOT_BASE}CLAIM: {claim.strip()}\nEVIDENCE: {evidence.strip()}\nRELATION:"
    pred, lat = _call_llama_server(prompt, url)
    return pred, lat, [pred]


def predict_ensemble(
    claim: str, evidence: str, workers: List[str], url: str = SERVER_URL
) -> Tuple[str, float, List[str]]:
    """Ensemble con N workers secuenciales + majority vote.
    Returns (final_relation, total_latency, [worker_preds]).
    """
    preds = []
    total_lat = 0.0
    for role in workers:
        prompt = f"{WORKER_ROLES[role]}{FEW_SHOT_BASE}CLAIM: {claim.strip()}\nEVIDENCE: {evidence.strip()}\nRELATION:"
        pred, lat = _call_llama_server(prompt, url)
        preds.append(pred)
        total_lat += lat

    # Majority vote (desempate: primer voto en orden de workers)
    valid_preds = [p for p in preds if p in ("SUPPORTS", "CONTRADICTS", "UNRELATED", "PARTIAL")]
    if not valid_preds:
        return "UNRELATED", total_lat, preds

    counts = Counter(valid_preds)
    max_count = max(counts.values())
    winners = [r for r, c in counts.items() if c == max_count]
    if len(winners) == 1:
        final = winners[0]
    else:
        # Desempate: preferir el primer worker valido
        for p in valid_preds:
            if p in winners:
                final = p
                break
        else:
            final = winners[0]

    return final, total_lat, preds


# ----------------------------- Run Profile -----------------------------

def run_profile(
    cases: List[dict],
    mode: str,
    workers: List[str],
    url: str,
) -> dict:
    """Ejecuta el perfil cognitivo completo para un modo dado."""
    n = len(cases)
    mode_label = f"{'_'.join(workers)}" if workers else "single"

    print("\n" + "=" * 70, flush=True)
    print(f"PERFIL COGNITIVO - BITNET b1.58-2B-4T ({mode})", flush=True)
    print(f"Workers: {len(workers) if workers else 1} | Regimen: GBNF + Few-Shot + repeat_penalty=1.0", flush=True)
    print("=" * 70, flush=True)

    results = []
    by_category = defaultdict(lambda: {"total": 0, "correct": 0, "predictions": defaultdict(int)})
    total_correct = 0

    t_start = time.time()
    for i, case in enumerate(cases):
        cid = case["id"]
        cat = case["category"]
        exp = case["expected"]
        cl = case["claim"]
        ev = case["evidence"]

        if mode == "single":
            pred, lat, worker_preds = predict_single(cl, ev, url=url)
        else:
            pred, lat, worker_preds = predict_ensemble(cl, ev, workers, url=url)

        is_ok = (pred == exp)
        if is_ok:
            total_correct += 1

        by_category[cat]["total"] += 1
        by_category[cat]["predictions"][pred] += 1
        if is_ok:
            by_category[cat]["correct"] += 1

        results.append({
            "id": cid,
            "category": cat,
            "expected": exp,
            "predicted": pred,
            "worker_predictions": worker_preds,
            "correct": is_ok,
            "latency_s": round(lat, 2),
        })

        workers_str = ",".join(worker_preds)
        status = "OK" if is_ok else f"X (pred: {pred})"
        print(f"  [{i+1:2d}/{n}] {cid:<8s} ({cat:<22s}) exp: {exp:<11s} -> {status} [{lat:.1f}s] workers: [{workers_str}]", flush=True)

    wall = time.time() - t_start
    overall_acc = total_correct / n

    print("\n" + "-" * 75, flush=True)
    print(f"{'Categoria':<25} | {'Casos':<6} | {'Correctos':<10} | {'Accuracy':<9} | {'Predicciones'}", flush=True)
    print("-" * 75, flush=True)

    for cat, data in sorted(by_category.items()):
        tot = data["total"]
        corr = data["correct"]
        acc = corr / tot if tot > 0 else 0.0
        preds_str = ", ".join(f"{k}:{v}" for k, v in sorted(data["predictions"].items()))
        print(f"{cat:<25} | {tot:<6} | {corr:<10} | {acc:>8.1%} | {preds_str}", flush=True)

    print("-" * 75, flush=True)
    print(f"{'TOTAL ' + mode:<25} | {n:<6} | {total_correct:<10} | {overall_acc:>8.1%} | Wall: {wall/60:.1f} min", flush=True)
    print("=" * 70, flush=True)

    return {
        "mode": mode,
        "workers": workers if workers else ["neutral"],
        "model": "BitNet-b1.58-2B-4T",
        "regime": "Minimal Cognitive Overhead (GBNF 1-token constraint, Few-Shot minimal, repeat_penalty=1.0)",
        "benchmark": "semantic_assessment_v2.json",
        "total_cases": n,
        "overall_accuracy": round(overall_acc, 4),
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
    parser = argparse.ArgumentParser(description="BitNet Cognitive Profiler (single + ensemble)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--modes", nargs="+", default=["single", "ensemble_2", "ensemble_4"],
                        choices=["single", "ensemble_2", "ensemble_4"],
                        help="Modes to run (default: all three)")
    args = parser.parse_args()

    port = args.port
    url = f"http://127.0.0.1:{port}"

    if not start_server(port):
        return 1

    try:
        with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
            cases = json.load(f)["cases"]

        all_reports = []

        if "single" in args.modes:
            report = run_profile(cases, "single", [], url)
            all_reports.append(report)
            out = OUTPUT_DIR / "bitnet_cognitive_profile_single.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"  Guardado: {out}\n", flush=True)

        if "ensemble_2" in args.modes:
            report = run_profile(cases, "ensemble_2", WORKER_ORDER_2, url)
            all_reports.append(report)
            out = OUTPUT_DIR / "bitnet_cognitive_profile_ensemble_2.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"  Guardado: {out}\n", flush=True)

        if "ensemble_4" in args.modes:
            report = run_profile(cases, "ensemble_4", WORKER_ORDER_4, url)
            all_reports.append(report)
            out = OUTPUT_DIR / "bitnet_cognitive_profile_ensemble_4.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"  Guardado: {out}\n", flush=True)

        # Resumen comparativo
        print("\n" + "=" * 70, flush=True)
        print("RESUMEN COMPARATIVO - BITNET b1.58-2B-4T", flush=True)
        print("=" * 70, flush=True)
        print(f"{'Modo':<15} | {'Accuracy':<10} | {'Correctos':<10} | {'Wall':<8}", flush=True)
        print("-" * 50, flush=True)
        for r in all_reports:
            print(f"{r['mode']:<15} | {r['overall_accuracy']:>8.1%} | {r['correct_count']:>5}/{r['total_cases']:<3} | {r['wall_time_s']/60:>6.1f}m", flush=True)
        print("=" * 70, flush=True)

    finally:
        stop_server()


if __name__ == "__main__":
    main()
