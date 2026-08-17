"""
Coliseo v1 - Repeticion controlada para Llama3.2 (CPU).

Motivacion:
  El resultado historico de Llama3.2 en Coliseo v1 (16.4% single, 29.1% ens2)
  esta contaminado por el protocolo: num_predict=10 + salida libre + parser
  leniento que defaultea a UNRELATED. Las pruebas aisladas mostraron que
  Llama3.2 se trunca antes de emitir la clasificacion con num_predict=10.

  Este runner repite Coliseo v1 para Llama3.2 unicamente, en CPU, con un
  protocolo controlado:
    - num_predict=64        (presupuesto ampliado)
    - think=false           (evita razonamiento interno que consume tokens)
    - format=json (schema)  (salida estructurada)
    - parser estricto       (no defaultea a UNRELATED; marca PROTOCOL_ERROR)
    - raw conservado        (respuesta cruda + eval_count + done_reason)

  Mismo benchmark (semantic_assessment_v2.json, 55 casos), mismos roles y
  configs que Coliseo v1, para permitir comparacion directa.

  No sobrescribe resultados crudos previos: escribe a
  results/raw/coliseo_v1_llama32_cpu_controlled.json.

Uso:
    python runners/run_coliseo_v1_llama32_cpu_controlled.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hybrid_rag.kernel.semantic_ensemble import WORKER_PROMPTS
from hybrid_rag.kernel.state import SEMANTIC_RELATIONS


# ----------------------------- Configuracion -----------------------------

MODEL_NAME = "llama3.2:3b"
MODEL_LABEL = "llama32-3b"

BENCHMARK = ROOT / "benchmarks" / "semantic_assessment_v2.json"
OUTPUT = ROOT / "results" / "raw" / "coliseo_v1_llama32_cpu_controlled.json"

BASE_URL = "http://localhost:11434"
DEFAULT_PORT = 11434
NUM_PREDICT = 64
TEMPERATURE = 0.0
NUM_THREAD = 4
NUM_GPU = 0  # CPU
TIMEOUT_S = 120.0

CONFIGS = [
    {"label": "single",     "roles": ["neutral"]},
    {"label": "ensemble_2", "roles": ["entailment", "skeptical"]},
    {"label": "ensemble_4", "roles": ["entailment", "skeptical", "contradiction", "neutral"]},
]

# Schema JSON estricto: solo relation + confidence.
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "relation": {
            "type": "string",
            "enum": ["SUPPORTS", "CONTRADICTS", "UNRELATED", "PARTIAL"],
        },
        "confidence": {"type": "number"},
    },
    "required": ["relation", "confidence"],
    "additionalProperties": False,
}


# ----------------------------- Provider -----------------------------

def generate_structured(prompt: str, base_url: str = BASE_URL) -> Dict[str, Any]:
    """Llama a Ollama con think=false, format=json schema, presupuesto ampliado.

    Devuelve un dict con: raw, eval_count, done_reason, latency_s, ok.
    No lanza; captura errores y los devuelve como ok=False.
    """
    t0 = time.time()
    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "format": JSON_SCHEMA,
                "options": {
                    "num_predict": NUM_PREDICT,
                    "temperature": TEMPERATURE,
                    "num_thread": NUM_THREAD,
                    "num_gpu": NUM_GPU,
                },
                "keep_alive": "10m",
            },
            timeout=TIMEOUT_S,
        )
        resp.raise_for_status()
        data = resp.json()
        latency = time.time() - t0
        return {
            "raw": (data.get("response") or "").strip(),
            "eval_count": data.get("eval_count", 0),
            "done_reason": data.get("done_reason", ""),
            "latency_s": latency,
            "ok": True,
            "error": None,
        }
    except Exception as exc:
        return {
            "raw": "",
            "eval_count": 0,
            "done_reason": "",
            "latency_s": time.time() - t0,
            "ok": False,
            "error": str(exc),
        }


# ----------------------------- Parser estricto -----------------------------

def parse_strict(gen: Dict[str, Any]) -> Tuple[str, float, bool, str]:
    """Parser estricto. No defaultea a UNRELATED.

    Returns: (relation, confidence, valid_json, parse_note)
      relation       -> SEMANTIC_RELATIONS o "PROTOCOL_ERROR"
      confidence     -> float (0.0 si invalido)
      valid_json     -> True si el JSON parseo y relation es valida
      parse_note     -> diagnostico breve
    """
    if not gen["ok"]:
        return "PROTOCOL_ERROR", 0.0, False, f"request_error: {gen['error']}"

    raw = gen["raw"]
    if not raw:
        return "PROTOCOL_ERROR", 0.0, False, "empty_response"

    try:
        obj = json.loads(raw)
    except Exception as exc:
        # A veces el JSON queda truncado; intento recuperacion heuristica
        # solo si contiene una relation valida literal.
        up = raw.upper()
        for rel in SEMANTIC_RELATIONS:
            if rel in up:
                return rel, 0.3, False, f"json_parse_failed_recovered: {exc}"
        return "PROTOCOL_ERROR", 0.0, False, f"json_parse_failed: {exc}"

    relation = str(obj.get("relation", "")).strip().upper()
    confidence = obj.get("confidence", 0.5)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.5

    if relation not in SEMANTIC_RELATIONS:
        return "PROTOCOL_ERROR", 0.0, False, f"invalid_relation: {relation!r}"

    return relation, confidence, True, "ok"


# ----------------------------- Ejecucion -----------------------------

def run_single(cases: List[dict], base_url: str = BASE_URL) -> List[dict]:
    """Config single: un worker neutral por caso."""
    results = []
    for case in cases:
        prompt = WORKER_PROMPTS["neutral"](case["claim"], case["evidence"])
        gen = generate_structured(prompt, base_url=base_url)
        relation, confidence, valid_json, note = parse_strict(gen)
        results.append({
            "id": case["id"],
            "category": case["category"],
            "expected": case["expected"],
            "produced": relation,
            "correct": relation == case["expected"],
            "valid_json": valid_json,
            "confidence": round(confidence, 3),
            "latency_s": round(gen["latency_s"], 2),
            "eval_count": gen["eval_count"],
            "done_reason": gen["done_reason"],
            "raw": gen["raw"],
            "parse_note": note,
        })
    return results


def run_ensemble(cases: List[dict], roles: List[str], base_url: str = BASE_URL) -> List[dict]:
    """Config ensemble: un worker por rol, agregacion por mayoria ponderada
    por confidence (replica ConfidenceWeightedMajorityVote)."""
    results = []
    for case in cases:
        votes: List[Dict[str, Any]] = []
        for role in roles:
            prompt = WORKER_PROMPTS[role](case["claim"], case["evidence"])
            gen = generate_structured(prompt, base_url=base_url)
            relation, confidence, valid_json, note = parse_strict(gen)
            votes.append({
                "role": role,
                "relation": relation,
                "confidence": round(confidence, 3),
                "valid_json": valid_json,
                "latency_s": round(gen["latency_s"], 2),
                "eval_count": gen["eval_count"],
                "done_reason": gen["done_reason"],
                "raw": gen["raw"],
                "parse_note": note,
            })

        # Agregacion: solo votos con relation valida (no PROTOCOL_ERROR)
        valid_votes = [v for v in votes if v["relation"] in SEMANTIC_RELATIONS]
        if not valid_votes:
            produced = "PROTOCOL_ERROR"
            agreement = 0.0
        else:
            weights: Dict[str, float] = {}
            counts: Dict[str, int] = {}
            for v in valid_votes:
                weights[v["relation"]] = weights.get(v["relation"], 0.0) + v["confidence"]
                counts[v["relation"]] = counts.get(v["relation"], 0) + 1
            produced = max(weights, key=weights.get)
            agreement = counts[produced] / len(valid_votes)

        total_lat = sum(v["latency_s"] for v in votes)
        results.append({
            "id": case["id"],
            "category": case["category"],
            "expected": case["expected"],
            "produced": produced,
            "correct": produced == case["expected"],
            "agreement": round(agreement, 3),
            "latency_s": round(total_lat, 2),
            "votes": votes,
        })
    return results


def compute_metrics(results: List[dict]) -> dict:
    n = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    protocol_ok = sum(1 for r in results if r.get("produced") in SEMANTIC_RELATIONS)
    latencies = [r.get("latency_s", 0.0) for r in results]
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0

    by_cat = defaultdict(lambda: {"total": 0, "correct": 0})
    by_rel = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        by_cat[r["category"]]["total"] += 1
        by_rel[r["expected"]]["total"] += 1
        if r["correct"]:
            by_cat[r["category"]]["correct"] += 1
            by_rel[r["expected"]]["correct"] += 1

    errors = [r for r in results if not r["correct"]]

    return {
        "n": n,
        "accuracy": round(correct / n, 4) if n > 0 else 0.0,
        "correct": correct,
        "protocol_validity": round(protocol_ok / n, 4) if n > 0 else 0.0,
        "avg_latency_s": round(avg_lat, 2),
        "by_category": {
            cat: {"total": s["total"], "correct": s["correct"],
                  "accuracy": round(s["correct"] / s["total"], 4) if s["total"] > 0 else 0.0}
            for cat, s in sorted(by_cat.items())
        },
        "by_relation": {
            rel: {"total": s["total"], "correct": s["correct"],
                  "accuracy": round(s["correct"] / s["total"], 4) if s["total"] > 0 else 0.0}
            for rel, s in sorted(by_rel.items())
        },
        "errors": [{"id": e["id"], "category": e["category"],
                    "expected": e["expected"], "produced": e["produced"]}
                   for e in errors],
    }


def unload_model(base_url: str = BASE_URL):
    try:
        requests.post(
            f"{base_url}/api/generate",
            json={"model": MODEL_NAME, "keep_alive": 0},
            timeout=10,
        )
        print(f"  unloaded: {MODEL_NAME}", flush=True)
    except Exception as e:
        print(f"  unload failed: {MODEL_NAME} ({e})", flush=True)


# ----------------------------- Main -----------------------------

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Coliseo v1 - Repeticion controlada Llama3.2 (CPU)"
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="Ollama instance port (default: 11434)")
    args = parser.parse_args()
    base_url = f"http://localhost:{args.port}"

    print("=" * 70, flush=True)
    print("COLOSEO v1 - REPETICION CONTROLADA - Llama3.2 (CPU)", flush=True)
    print("Protocolo: num_predict=64, think=false, format=json, parser estricto", flush=True)
    print("=" * 70, flush=True)

    with BENCHMARK.open("r", encoding="utf-8") as f:
        cases = json.load(f)["cases"]
    n = len(cases)
    print(f"Cases: {n} | Model: {MODEL_NAME} | CPU (num_gpu=0) | Ollama: {base_url}", flush=True)
    print(f"Configs: {[c['label'] for c in CONFIGS]}", flush=True)
    print(flush=True)

    # Verificar disponibilidad del modelo
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=10)
        r.raise_for_status()
        tags = {m["name"] for m in r.json().get("models", [])}
        if MODEL_NAME not in tags:
            print(f"SKIP: {MODEL_NAME} no disponible en Ollama ({base_url})", flush=True)
            return 1
        print(f"  OK: {MODEL_NAME} ({base_url})", flush=True)
    except Exception as e:
        print(f"SKIP: Ollama no responde en {base_url} ({e})", flush=True)
        return 1
    print(flush=True)

    all_configs: Dict[str, Any] = {}

    for ci, cfg in enumerate(CONFIGS):
        clabel = cfg["label"]
        roles = cfg["roles"]
        print(f"{'='*70}", flush=True)
        print(f"Config {ci+1}/{len(CONFIGS)}: {clabel} (roles={roles})", flush=True)
        print(f"{'='*70}", flush=True)

        t0 = time.time()
        if len(roles) == 1:
            results = run_single(cases, base_url=base_url)
        else:
            results = run_ensemble(cases, roles, base_url=base_url)
        wall = time.time() - t0
        metrics = compute_metrics(results)

        all_configs[clabel] = {
            "roles": roles,
            "metrics": metrics,
            "wall_time_s": round(wall, 1),
            "per_case": results,
        }

        print(f"  accuracy={metrics['accuracy']:.1%} ({metrics['correct']}/{n}) "
              f"protocol={metrics['protocol_validity']:.0%} "
              f"avg_lat={metrics['avg_latency_s']:.1f}s wall={wall:.0f}s", flush=True)

        if metrics["correct"] < n:
            print(f"  Errores por categoria:", flush=True)
            for cat, s in sorted(metrics["by_category"].items()):
                if s["correct"] < s["total"]:
                    print(f"    {cat:25s}: {s['correct']}/{s['total']}", flush=True)
        print(flush=True)

    # Tabla comparativa vs historico
    print("=" * 70, flush=True)
    print("COMPARACION vs COLOSEO v1 HISTORICO (CPU, num_predict=10, libre)", flush=True)
    print("=" * 70, flush=True)
    print(f"  {'config':<12s} | {'historico':>10s} | {'controlado':>11s} | {'delta':>8s}", flush=True)
    print("  " + "-" * 52, flush=True)
    cpu_baselines = {"single": 0.164, "ensemble_2": 0.291, "ensemble_4": 0.200}
    for cfg in CONFIGS:
        cl = cfg["label"]
        hist = cpu_baselines.get(cl, 0.0)
        ctrl = all_configs[cl]["metrics"]["accuracy"]
        delta = ctrl - hist
        print(f"  {cl:<12s} | {hist:>9.1%} | {ctrl:>10.1%} | {delta:>+7.1%}", flush=True)

    print(flush=True)
    print("Validez de protocolo (controlado):", flush=True)
    for cfg in CONFIGS:
        cl = cfg["label"]
        pv = all_configs[cl]["metrics"]["protocol_validity"]
        print(f"  {cl:<12s}: {pv:.0%}", flush=True)

    # Reporte JSON
    report = {
        "pilot": "coliseo_v1_llama32_cpu_controlled",
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmark": str(BENCHMARK.name),
        "case_count": n,
        "model": {"label": MODEL_LABEL, "name": MODEL_NAME},
        "execution": "CPU (num_gpu=0)",
        "protocol": {
            "num_predict": NUM_PREDICT,
            "temperature": TEMPERATURE,
            "num_thread": NUM_THREAD,
            "num_gpu": NUM_GPU,
            "think": False,
            "format": "json_schema",
            "parser": "strict_no_default",
            "schema": JSON_SCHEMA,
        },
        "configs": [{"label": c["label"], "roles": c["roles"]} for c in CONFIGS],
        "results": {cl: all_configs[cl] for cl in all_configs},
        "historical_baseline": cpu_baselines,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(flush=True)
    print(f"Report: {OUTPUT}", flush=True)

    print(flush=True)
    print("Descargando modelo de Ollama...", flush=True)
    unload_model(base_url=base_url)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
