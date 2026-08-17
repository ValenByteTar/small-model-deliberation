"""
Coliseo: 4 modelos Ollama (CPU) vs SemanticAssessment Benchmark v2.

Modelos:
  - ibm/granite4.1:3b-q4_K_M  (3.4B, Q4)
  - llama3.2:3b                 (3B, Meta)
  - qwen3-4b-rag:latest         (4B, RAG-tuned)
  - llama3.1:latest             (8B, Meta)

Por cada modelo:
  - Single worker (neutral prompt)
  - Ensemble 2 workers (A+B, entailment + skeptical)
  - Ensemble 4 workers (A+B+C+D, todos los roles)

Reporta accuracy por modelo, por configuracion, por categoria
diagnostica y por relacion. Identifica el mejor modelo y si el
ensemble aporta valor sobre single.

Todo corre en CPU (num_gpu=0) para no competir con el pipeline.

Uso:
    python scripts/run_coliseo_benchmark_v2.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hybrid_rag.kernel.semantic_ensemble import (
    ConfidenceWeightedMajorityVote,
    SemanticEnsemble,
    SemanticWorker,
    WorkerResult,
    WORKER_PROMPTS,
    WORKER_ROLES,
)
from hybrid_rag.kernel.state import SEMANTIC_RELATIONS
from hybrid_rag.providers.ollama_provider import OllamaModelProvider

BENCHMARK = ROOT / "tests" / "eval" / "canonical" / "semantic_assessment_benchmark_v2.json"
OUTPUT = ROOT / "tests" / "eval" / "canonical" / "reports" / "coliseo_benchmark_v2.json"

MODELS = [
    {"name": "ibm/granite4.1:3b-q4_K_M", "label": "granite-3b-q4", "params": "3.4B"},
    {"name": "llama3.2:3b",              "label": "llama32-3b",     "params": "3B"},
    {"name": "qwen3-4b-rag:latest",      "label": "qwen3-4b-rag",   "params": "4B"},
]

# Configuraciones a probar por modelo
CONFIGS = [
    {"label": "single",       "roles": ["neutral"]},
    {"label": "ensemble_2",   "roles": ["entailment", "skeptical"]},
    {"label": "ensemble_4",   "roles": ["entailment", "skeptical", "contradiction", "neutral"]},
]


def make_provider(model_name: str, base_url: str = "http://localhost:11434") -> OllamaModelProvider:
    return OllamaModelProvider(
        model=model_name,
        base_url=base_url,
        num_gpu=0,  # CPU forzado
        default_options={"num_predict": 10, "temperature": 0.0, "num_thread": 4},
    )


def make_workers(model_name: str, roles: List[str], base_url: str = "http://localhost:11434") -> List[SemanticWorker]:
    workers = []
    for i, role in enumerate(roles):
        provider = make_provider(model_name, base_url=base_url)
        w = SemanticWorker(
            worker_id=f"w-{role[0]}",
            role=role,
            model_provider=provider,
            prompt_fn=WORKER_PROMPTS[role],
        )
        workers.append(w)
    return workers


def run_config(model_name: str, roles: List[str], cases: List[dict], base_url: str = "http://localhost:11434") -> Tuple[List[dict], float]:
    """Corre una configuracion (single o ensemble) sobre todos los casos."""
    workers = make_workers(model_name, roles, base_url=base_url)
    total_time = 0.0
    results = []

    if len(workers) == 1:
        # Single: secuencial
        w = workers[0]
        for case in cases:
            wr = w.assess(
                claim=case["claim"], evidence_text=case["evidence"],
                evidence_id=case["id"], run_id="coliseo", timeout=90.0,
            )
            total_time += wr.latency_s
            produced = wr.relation if wr.assessment else "ERROR"
            results.append({
                "id": case["id"], "category": case["category"],
                "expected": case["expected"], "produced": produced,
                "correct": produced == case["expected"],
                "latency_s": round(wr.latency_s, 2),
            })
    else:
        # Ensemble: workers en paralelo por caso
        ensemble = SemanticEnsemble(workers=workers, strategy=ConfidenceWeightedMajorityVote())
        for case in cases:
            assessment, worker_results, meta = ensemble.assess(
                claim=case["claim"], evidence_text=case["evidence"],
                evidence_id=case["id"], run_id="coliseo", timeout=90.0,
            )
            lat = meta.get("total_latency_s", 0.0)
            total_time += lat
            produced = assessment.relation if assessment else "ERROR"
            results.append({
                "id": case["id"], "category": case["category"],
                "expected": case["expected"], "produced": produced,
                "correct": produced == case["expected"],
                "latency_s": round(lat, 2),
                "agreement": meta.get("agreement", 0.0),
            })
    return results, total_time


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


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Coliseo benchmark v2 (CPU)")
    parser.add_argument("--port", type=int, default=11434,
                        help="Ollama instance port (default: 11434)")
    args = parser.parse_args()
    base_url = f"http://localhost:{args.port}"

    print("=" * 70, flush=True)
    print("COLOSEO: 4 modelos vs SemanticAssessment Benchmark v2", flush=True)
    print("CPU-only (num_gpu=0) | 55 casos | 10 categorias diagnosticas", flush=True)
    print(f"Ollama: {base_url}", flush=True)
    print("=" * 70, flush=True)

    with BENCHMARK.open("r", encoding="utf-8") as f:
        bench = json.load(f)
    cases = bench["cases"]
    n = len(cases)
    print(f"Cases: {n}", flush=True)
    print(f"Models: {[m['label'] for m in MODELS]}", flush=True)
    print(f"Configs: {[c['label'] for c in CONFIGS]}", flush=True)
    print(f"Total runs: {len(MODELS) * len(CONFIGS)} configs x {n} cases", flush=True)
    print(flush=True)

    # Verificar modelos disponibles
    for m in MODELS:
        p = make_provider(m["name"], base_url=base_url)
        if not p.is_available():
            print(f"SKIP: {m['name']} no disponible en {base_url}", flush=True)
            return 1
        print(f"  OK: {m['label']} ({m['name']})", flush=True)
    print(flush=True)

    all_results = {}

    for mi, model in enumerate(MODELS):
        mlabel = model["label"]
        mname = model["name"]
        print(f"{'='*70}", flush=True)
        print(f"MODELO {mi+1}/{len(MODELS)}: {mlabel} ({mname})", flush=True)
        print(f"{'='*70}", flush=True)

        all_results[mlabel] = {"model": mname, "configs": {}}

        for ci, cfg in enumerate(CONFIGS):
            clabel = cfg["label"]
            roles = cfg["roles"]
            print(f"\n  Config {ci+1}/{len(CONFIGS)}: {clabel} (roles={roles})", flush=True)

            t0 = time.time()
            results, total_time = run_config(mname, roles, cases, base_url=base_url)
            wall_time = time.time() - t0
            metrics = compute_metrics(results)

            all_results[mlabel]["configs"][clabel] = {
                "roles": roles,
                "metrics": metrics,
                "wall_time_s": round(wall_time, 1),
                "per_case": results,
            }

            print(f"    accuracy={metrics['accuracy']:.1%} ({metrics['correct']}/{n}) "
                  f"protocol={metrics['protocol_validity']:.0%} "
                  f"avg_lat={metrics['avg_latency_s']:.1f}s "
                  f"wall={wall_time:.0f}s", flush=True)

            # Print por categoria solo si hay errores
            if metrics["correct"] < n:
                print(f"    Errores por categoria:", flush=True)
                for cat, s in sorted(metrics["by_category"].items()):
                    if s["correct"] < s["total"]:
                        print(f"      {cat:25s}: {s['correct']}/{s['total']}", flush=True)

        print(flush=True)

    # ==================== Tabla comparativa final ====================
    print("=" * 70, flush=True)
    print("TABLA COMPARATIVA FINAL", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)

    # Header
    header = f"{'Model':<18s}"
    for cfg in CONFIGS:
        header += f" | {cfg['label']:>12s}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    # Filas: accuracy por modelo
    for model in MODELS:
        mlabel = model["label"]
        row = f"{mlabel:<18s}"
        for cfg in CONFIGS:
            clabel = cfg["label"]
            if mlabel in all_results and clabel in all_results[mlabel]["configs"]:
                acc = all_results[mlabel]["configs"][clabel]["metrics"]["accuracy"]
                row += f" | {acc:>11.1%}"
            else:
                row += f" | {'N/A':>12s}"
        print(row, flush=True)

    print(flush=True)

    # Latencia
    print("Latencia promedio (s):", flush=True)
    header = f"{'Model':<18s}"
    for cfg in CONFIGS:
        header += f" | {cfg['label']:>12s}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for model in MODELS:
        mlabel = model["label"]
        row = f"{mlabel:<18s}"
        for cfg in CONFIGS:
            clabel = cfg["label"]
            if mlabel in all_results and clabel in all_results[mlabel]["configs"]:
                lat = all_results[mlabel]["configs"][clabel]["metrics"]["avg_latency_s"]
                row += f" | {lat:>11.1f}s"
            else:
                row += f" | {'N/A':>12s}"
        print(row, flush=True)

    print(flush=True)

    # Mejor modelo
    best_model = None
    best_acc = 0.0
    best_cfg = None
    for model in MODELS:
        mlabel = model["label"]
        for cfg in CONFIGS:
            clabel = cfg["label"]
            if mlabel in all_results and clabel in all_results[mlabel]["configs"]:
                acc = all_results[mlabel]["configs"][clabel]["metrics"]["accuracy"]
                if acc > best_acc:
                    best_acc = acc
                    best_model = mlabel
                    best_cfg = clabel

    print(f"Mejor configuracion: {best_model} / {best_cfg} = {best_acc:.1%}", flush=True)

    # Ensemble vs single por modelo
    print(flush=True)
    print("Ensemble vs Single:", flush=True)
    for model in MODELS:
        mlabel = model["label"]
        if mlabel not in all_results:
            continue
        configs = all_results[mlabel]["configs"]
        single_acc = configs.get("single", {}).get("metrics", {}).get("accuracy", 0.0)
        ens2_acc = configs.get("ensemble_2", {}).get("metrics", {}).get("accuracy", 0.0)
        ens4_acc = configs.get("ensemble_4", {}).get("metrics", {}).get("accuracy", 0.0)
        delta2 = ens2_acc - single_acc
        delta4 = ens4_acc - single_acc
        print(f"  {mlabel:<18s}: single={single_acc:.1%} ens2={ens2_acc:.1%} ({delta2:+.1%}) "
              f"ens4={ens4_acc:.1%} ({delta4:+.1%})", flush=True)

    # Por categoria del mejor modelo
    print(flush=True)
    print(f"Categorias diagnosticas — {best_model} / {best_cfg}:", flush=True)
    if best_model and best_cfg:
        cats = all_results[best_model]["configs"][best_cfg]["metrics"]["by_category"]
        for cat, s in sorted(cats.items()):
            acc = s["accuracy"]
            bar = "█" * int(acc * 20)
            print(f"  {cat:25s}: {s['correct']:2d}/{s['total']:2d} = {acc:5.1%} {bar}", flush=True)

    # Reporte JSON
    report = {
        "pilot": "coliseo_benchmark_v2",
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmark": str(BENCHMARK.name),
        "case_count": n,
        "models": [{"label": m["label"], "name": m["name"], "params": m["params"]} for m in MODELS],
        "configs": [{"label": c["label"], "roles": c["roles"]} for c in CONFIGS],
        "results": all_results,
        "best": {"model": best_model, "config": best_cfg, "accuracy": best_acc},
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(flush=True)
    print(f"Report: {OUTPUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
