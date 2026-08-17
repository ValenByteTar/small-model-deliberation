"""
EXP-021: BitNet Absence Detection Falsation Probe.

Ultimo experimento de falsacion. Responde una unica pregunta:
¿Puede BitNet usar ausencia de evidencia como condicion negativa
para una inferencia composicional?

Tres condiciones minimales:
  A) implicit_absence:  Evidence A+B, Claim A+B+C    -> FALSE (C ausente)
  B) explicit_negation: Evidence A+B, Claim A+B+NOT-C -> TRUE  (NOT-C consistente)
  C) total_absence:     Evidence A+B, Claim C          -> FALSE (C totalmente ausente)

Mide exclusivamente:
  - greedy TRUE/FALSE
  - logprob(TRUE), logprob(FALSE)
  - margen logP(TRUE) - logP(FALSE)

Si BitNet dice TRUE sistematicamente para lo no-soportado,
cerramos la rama: incapacidad operacional de usar ausencia
como condicion negativa.
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

BENCH_PATH = ROOT / "benchmarks" / "absence_detection_v1.json"
OUTPUT_DIR = ROOT / "results" / "raw"

DEFAULT_PORT = 8098
SERVER_URL = f"http://127.0.0.1:{DEFAULT_PORT}"

GRAMMAR_TRUE_FALSE = 'root ::= "TRUE" | "FALSE" | " TRUE" | " FALSE" | " True" | " False" | " true" | " false"'

# Few-shot: 2 ejemplos (1 TRUE, 1 FALSE), minimalistas
FEW_SHOT = '''Task: Based on the EVIDENCE, is the CLAIM TRUE or FALSE? Answer TRUE only if the evidence explicitly confirms every part of the claim. Answer FALSE if any part of the claim is not confirmed by the evidence.

CLAIM: The system supports encryption at rest and encryption in transit.
EVIDENCE: The system supports encryption at rest, encryption in transit, and access logging.
Based on the evidence, the claim is: TRUE

CLAIM: The system supports encryption at rest, encryption in transit, and access logging.
EVIDENCE: The system supports encryption at rest and encryption in transit.
Based on the evidence, the claim is: FALSE

'''

MAP_TRUE_FALSE = {
    "TRUE": "TRUE", " TRUE": "TRUE", "true": "TRUE", " True": "TRUE", " true": "TRUE",
    "FALSE": "FALSE", " FALSE": "FALSE", "false": "FALSE", " False": "FALSE", " false": "FALSE",
}

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
                print("llama-server listo", flush=True)
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


def call_with_logprobs(prompt: str, grammar: str, url: str, max_tokens: int = 4) -> dict:
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


def predict_true_false(claim: str, evidence: str, url: str) -> dict:
    prompt = f"{FEW_SHOT}CLAIM: {claim.strip()}\nEVIDENCE: {evidence.strip()}\nBased on the evidence, the claim is:"
    result = call_with_logprobs(prompt, GRAMMAR_TRUE_FALSE, url, max_tokens=4)
    raw = result["content"]
    lps = get_token_logprobs(result["probs"])
    agg = aggregate_logprobs(lps, MAP_TRUE_FALSE)
    greedy = map_raw(raw, MAP_TRUE_FALSE)
    argmax = max(agg, key=agg.get) if agg else greedy
    return {
        "raw": raw, "greedy": greedy, "argmax": argmax,
        "logprob_true": agg.get("TRUE", None),
        "logprob_false": agg.get("FALSE", None),
        "margin": (agg.get("TRUE", 0) - agg.get("FALSE", 0)) if "TRUE" in agg and "FALSE" in agg else None,
        "raw_logprobs": [{"token": t.get("token", ""), "logprob": t.get("logprob", 0)} for t in lps],
    }


def main():
    port = DEFAULT_PORT
    url = f"http://127.0.0.1:{port}"

    if not start_server(port):
        return 1

    try:
        with open(BENCH_PATH, "r", encoding="utf-8") as f:
            cases = json.load(f)["cases"]

        print(f"\n{'='*90}", flush=True)
        print("EXP-021: ABSENCE DETECTION FALSATION PROBE", flush=True)
        print(f"{'='*90}", flush=True)
        print(f"\n3 condiciones x 6 casos = {len(cases)} casos", flush=True)
        print(f"  A) implicit_absence:  Evidence A+B, Claim A+B+C    -> FALSE (C ausente)", flush=True)
        print(f"  B) explicit_negation: Evidence A+B, Claim A+B+NOT-C -> TRUE  (NOT-C consistente)", flush=True)
        print(f"  C) total_absence:     Evidence A+B, Claim C          -> FALSE (C totalmente ausente)", flush=True)
        print(f"\nMetrica: greedy TRUE/FALSE, logprob(TRUE), logprob(FALSE), margen", flush=True)

        results = []
        by_cond = defaultdict(lambda: {"total": 0, "correct": 0, "true_preds": 0, "false_preds": 0,
                                        "margins": [], "lp_true": [], "lp_false": []})

        print(f"\n{'ID':<8} {'Cond':<20} {'Exp':<6} {'Greedy':<7} {'Argmax':<7} {'lp(TRUE)':<10} {'lp(FALSE)':<10} {'Margen':<10} {'OK?'}", flush=True)
        print("-" * 95, flush=True)

        for i, c in enumerate(cases):
            pred = predict_true_false(c["claim"], c["evidence"], url)
            is_ok = pred["greedy"] == c["expected"]
            cond = c["condition"]

            by_cond[cond]["total"] += 1
            if is_ok:
                by_cond[cond]["correct"] += 1
            if pred["greedy"] == "TRUE":
                by_cond[cond]["true_preds"] += 1
            else:
                by_cond[cond]["false_preds"] += 1
            if pred["margin"] is not None:
                by_cond[cond]["margins"].append(pred["margin"])
            if pred["logprob_true"] is not None:
                by_cond[cond]["lp_true"].append(pred["logprob_true"])
            if pred["logprob_false"] is not None:
                by_cond[cond]["lp_false"].append(pred["logprob_false"])

            lp_t = f"{pred['logprob_true']:.3f}" if pred['logprob_true'] is not None else "N/A"
            lp_f = f"{pred['logprob_false']:.3f}" if pred['logprob_false'] is not None else "N/A"
            margin = f"{pred['margin']:.3f}" if pred['margin'] is not None else "N/A"
            ok_str = "OK" if is_ok else "X"

            print(f"{c['id']:<8} {cond:<20} {c['expected']:<6} {pred['greedy']:<7} {pred['argmax']:<7} {lp_t:<10} {lp_f:<10} {margin:<10} {ok_str}", flush=True)

            results.append({
                "id": c["id"], "condition": cond, "expected": c["expected"],
                "claim": c["claim"], "evidence": c["evidence"],
                "greedy": pred["greedy"], "argmax": pred["argmax"],
                "logprob_true": pred["logprob_true"], "logprob_false": pred["logprob_false"],
                "margin": pred["margin"], "raw": pred["raw"],
                "raw_logprobs": pred["raw_logprobs"],
                "correct": is_ok,
            })

        # Resumen por condicion
        print(f"\n{'='*95}", flush=True)
        print(f"{'Condicion':<22} | {'Casos':<6} | {'Correctos':<10} | {'Accuracy':<9} | {'TRUE':<5} | {'FALSE':<6} | {'Margen medio':<13} | {'Margen min':<11}", flush=True)
        print(f"{'-'*95}", flush=True)

        total_correct = 0
        total_cases = len(cases)
        all_margins_false_expected = []  # margins de casos donde expected=FALSE

        for cond, data in sorted(by_cond.items()):
            acc = data["correct"] / data["total"] if data["total"] > 0 else 0.0
            mean_margin = sum(data["margins"]) / len(data["margins"]) if data["margins"] else 0
            min_margin = min(data["margins"]) if data["margins"] else 0
            print(f"{cond:<22} | {data['total']:<6} | {data['correct']:<10} | {acc:>8.1%} | {data['true_preds']:<5} | {data['false_preds']:<6} | {mean_margin:>13.3f} | {min_margin:>11.3f}", flush=True)
            total_correct += data["correct"]

        print(f"{'-'*95}", flush=True)
        overall_acc = total_correct / total_cases
        print(f"{'TOTAL':<22} | {total_cases:<6} | {total_correct:<10} | {overall_acc:>8.1%} |", flush=True)
        print(f"{'='*95}", flush=True)

        # Analisis critico: casos donde expected=FALSE
        false_cases = [r for r in results if r["expected"] == "FALSE"]
        false_correct = sum(1 for r in false_cases if r["greedy"] == "FALSE")
        false_true_preds = sum(1 for r in false_cases if r["greedy"] == "TRUE")
        false_margins = [r["margin"] for r in false_cases if r["margin"] is not None]

        print(f"\n{'='*70}", flush=True)
        print("ANALISIS DE FALSACION: casos donde expected=FALSE", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"  Total casos expected=FALSE: {len(false_cases)}", flush=True)
        print(f"  Correctos (greedy=FALSE):   {false_correct}/{len(false_cases)} ({false_correct/len(false_cases):.1%})", flush=True)
        print(f"  Incorrectos (greedy=TRUE):  {false_true_preds}/{len(false_cases)} ({false_true_preds/len(false_cases):.1%})", flush=True)
        if false_margins:
            mean_m = sum(false_margins) / len(false_margins)
            min_m = min(false_margins)
            max_m = max(false_margins)
            print(f"  Margen logP(TRUE)-logP(FALSE):", flush=True)
            print(f"    medio: {mean_m:.3f}", flush=True)
            print(f"    min:   {min_m:.3f}", flush=True)
            print(f"    max:   {max_m:.3f}", flush=True)
            positive_margins = sum(1 for m in false_margins if m > 0)
            print(f"    Casos con margen > 0 (TRUE > FALSE): {positive_margins}/{len(false_margins)}", flush=True)

        # Casos donde expected=TRUE (explicit_negation)
        true_cases = [r for r in results if r["expected"] == "TRUE"]
        true_correct = sum(1 for r in true_cases if r["greedy"] == "TRUE")
        true_margins = [r["margin"] for r in true_cases if r["margin"] is not None]

        print(f"\n  Total casos expected=TRUE:  {len(true_cases)}", flush=True)
        print(f"  Correctos (greedy=TRUE):    {true_correct}/{len(true_cases)} ({true_correct/len(true_cases):.1%})", flush=True)
        if true_margins:
            mean_m = sum(true_margins) / len(true_margins)
            print(f"  Margen medio: {mean_m:.3f}", flush=True)

        # Veredicto
        print(f"\n{'='*70}", flush=True)
        print("VEREDICTO", flush=True)
        print(f"{'='*70}", flush=True)
        false_acc = false_correct / len(false_cases) if false_cases else 0
        if false_acc == 0:
            print(f"  FALSE accuracy: 0% ({false_correct}/{len(false_cases)})", flush=True)
            print(f"  BitNet dice TRUE sistematicamente para claims no-soportados.", flush=True)
            print(f"  CONCLUSION: BitNet no puede usar ausencia de evidencia como", flush=True)
            print(f"  condicion negativa. Incapacidad operacional confirmada.", flush=True)
            print(f"  Cerrar rama BitNet como semantic assessor generalista.", flush=True)
        elif false_acc < 0.5:
            print(f"  FALSE accuracy: {false_acc:.1%} ({false_correct}/{len(false_cases)})", flush=True)
            print(f"  BitNet falla la mayoria de casos de ausencia. Incapacidad", flush=True)
            print(f"  parcial confirmada. Cerrar rama con matiz.", flush=True)
        else:
            print(f"  FALSE accuracy: {false_acc:.1%} ({false_correct}/{len(false_cases)})", flush=True)
            print(f"  BitNet puede detectar ausencia en casos minimos.", flush=True)
            print(f"  El problema de EXP-020 era el framing, no la capacidad.", flush=True)
            print(f"  Reabrir investigacion con framing optimizado.", flush=True)

        # Guardar
        out = OUTPUT_DIR / "bitnet_absence_detection.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump({
                "cases": results,
                "summary": {
                    "total": total_cases,
                    "correct": total_correct,
                    "accuracy": round(overall_acc, 4),
                    "false_expected": len(false_cases),
                    "false_correct": false_correct,
                    "false_accuracy": round(false_acc, 4),
                    "true_expected": len(true_cases),
                    "true_correct": true_correct,
                    "true_accuracy": round(true_correct / max(1, len(true_cases)), 4),
                    "false_margins_mean": round(sum(false_margins) / len(false_margins), 4) if false_margins else None,
                    "false_margins_min": round(min(false_margins), 4) if false_margins else None,
                    "false_margins_max": round(max(false_margins), 4) if false_margins else None,
                    "false_positive_margins": sum(1 for m in false_margins if m > 0),
                },
                "by_condition": {cond: {"total": d["total"], "correct": d["correct"],
                                        "accuracy": round(d["correct"] / d["total"], 4) if d["total"] > 0 else 0.0,
                                        "true_preds": d["true_preds"], "false_preds": d["false_preds"],
                                        "mean_margin": round(sum(d["margins"]) / len(d["margins"]), 4) if d["margins"] else None,
                                        "mean_lp_true": round(sum(d["lp_true"]) / len(d["lp_true"]), 4) if d["lp_true"] else None,
                                        "mean_lp_false": round(sum(d["lp_false"]) / len(d["lp_false"]), 4) if d["lp_false"] else None}
                                 for cond, d in by_cond.items()},
            }, f, indent=2, ensure_ascii=False)
        print(f"\nReporte: {out}", flush=True)

    finally:
        stop_server()


if __name__ == "__main__":
    main()
