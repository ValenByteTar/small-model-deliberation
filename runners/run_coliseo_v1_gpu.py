"""
Coliseo v1 GPU: mismos 3 modelos del coliseo v1 pero en GPU.

Aisla la variable CPU vs GPU. Mismo dataset, mismos modelos, mismos
prompts, misma logica. Solo cambia num_gpu=0 -> num_gpu=99.

Modelos:
  - ibm/granite4.1:3b-q4_K_M  (3.4B, Q4)
  - llama3.2:3b                 (3B, Meta)
  - qwen3-4b-rag:latest         (4B, RAG-tuned)

Uso:
    python scripts/run_coliseo_v1_gpu.py
"""

from __future__ import annotations
import json, sys, time
from collections import defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hybrid_rag.kernel.semantic_ensemble import (
    ConfidenceWeightedMajorityVote, SemanticEnsemble, SemanticWorker,
    WORKER_PROMPTS, WORKER_ROLES,
)
from hybrid_rag.kernel.state import SEMANTIC_RELATIONS
from hybrid_rag.providers.ollama_provider import OllamaModelProvider


def unload_models(model_names):
    """Descarga los modelos de Ollama (keep_alive=0) para liberar RAM."""
    for name in model_names:
        try:
            requests.post(
                "http://localhost:11434/api/generate",
                json={"model": name, "keep_alive": 0},
                timeout=10,
            )
            print(f"  unloaded: {name}", flush=True)
        except Exception as e:
            print(f"  unload failed: {name} ({e})", flush=True)

BENCHMARK = ROOT / "tests" / "eval" / "canonical" / "semantic_assessment_benchmark_v2.json"
OUTPUT = ROOT / "tests" / "eval" / "canonical" / "reports" / "coliseo_v1_gpu.json"

MODELS = [
    {"name": "ibm/granite4.1:3b-q4_K_M", "label": "granite-3b-q4", "params": "3.4B"},
    {"name": "llama3.2:3b",              "label": "llama32-3b",     "params": "3B"},
    {"name": "qwen3-4b-rag:latest",      "label": "qwen3-4b-rag",   "params": "4B"},
]

CONFIGS = [
    {"label": "single",       "roles": ["neutral"]},
    {"label": "ensemble_2",   "roles": ["entailment", "skeptical"]},
    {"label": "ensemble_4",   "roles": ["entailment", "skeptical", "contradiction", "neutral"]},
]


def make_provider(model_name):
    return OllamaModelProvider(
        model=model_name, base_url="http://localhost:11434",
        num_gpu=99,  # GPU
        default_options={"num_predict": 10, "temperature": 0.0, "num_thread": 4},
    )


def make_workers(model_name, roles):
    workers = []
    for role in roles:
        w = SemanticWorker(
            worker_id=f"w-{role[0]}", role=role,
            model_provider=make_provider(model_name),
            prompt_fn=WORKER_PROMPTS[role],
        )
        workers.append(w)
    return workers


def run_config(model_name, roles, cases):
    workers = make_workers(model_name, roles)
    total_time = 0.0
    results = []

    if len(workers) == 1:
        w = workers[0]
        for case in cases:
            wr = w.assess(
                claim=case["claim"], evidence_text=case["evidence"],
                evidence_id=case["id"], run_id="coliseo-v1-gpu", timeout=90.0,
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
        ensemble = SemanticEnsemble(workers=workers, strategy=ConfidenceWeightedMajorityVote())
        for case in cases:
            assessment, worker_results, meta = ensemble.assess(
                claim=case["claim"], evidence_text=case["evidence"],
                evidence_id=case["id"], run_id="coliseo-v1-gpu", timeout=90.0,
            )
            lat = meta.get("total_latency_s", 0.0)
            total_time += lat
            produced = assessment.relation if assessment else "ERROR"
            results.append({
                "id": case["id"], "category": case["category"],
                "expected": case["expected"], "produced": produced,
                "correct": produced == case["expected"],
                "latency_s": round(lat, 2),
            })
    return results, total_time


def compute_metrics(results):
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

    return {
        "n": n, "accuracy": round(correct / n, 4) if n > 0 else 0.0,
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
                    for e in results if not e.get("correct")],
    }


def main():
    print("=" * 70, flush=True)
    print("COLOSEO v1 GPU: 3 modelos vs Benchmark v2 (GPU)", flush=True)
    print("Mismos modelos que coliseo v1 CPU, ahora en GPU", flush=True)
    print("=" * 70, flush=True)

    with BENCHMARK.open("r", encoding="utf-8") as f:
        cases = json.load(f)["cases"]
    n = len(cases)
    print(f"Cases: {n} | Models: {[m['label'] for m in MODELS]}", flush=True)
    print(f"Configs: {[c['label'] for c in CONFIGS]} | GPU (num_gpu=99)", flush=True)
    print(flush=True)

    for m in MODELS:
        p = make_provider(m["name"])
        if not p.is_available():
            print(f"SKIP: {m['name']} no disponible", flush=True)
            return 1
        print(f"  OK: {m['label']}", flush=True)
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
            results, total_time = run_config(mname, roles, cases)
            wall_time = time.time() - t0
            metrics = compute_metrics(results)

            all_results[mlabel]["configs"][clabel] = {
                "roles": roles, "metrics": metrics,
                "wall_time_s": round(wall_time, 1), "per_case": results,
            }

            print(f"    accuracy={metrics['accuracy']:.1%} ({metrics['correct']}/{n}) "
                  f"protocol={metrics['protocol_validity']:.0%} "
                  f"avg_lat={metrics['avg_latency_s']:.1f}s wall={wall_time:.0f}s", flush=True)

            if metrics["correct"] < n:
                for cat, s in sorted(metrics["by_category"].items()):
                    if s["correct"] < s["total"]:
                        print(f"      {cat:25s}: {s['correct']}/{s['total']}", flush=True)

        print(flush=True)

    # Tabla final
    print("=" * 70, flush=True)
    print("TABLA COMPARATIVA — COLOSEO v1 GPU", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)

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
                acc = all_results[mlabel]["configs"][clabel]["metrics"]["accuracy"]
                row += f" | {acc:>11.1%}"
            else:
                row += f" | {'N/A':>12s}"
        print(row, flush=True)

    print(flush=True)
    print("Latencia (s):", flush=True)
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
    print("Comparacion CPU vs GPU:", flush=True)
    print("  (CPU = coliseo v1, GPU = este coliseo)", flush=True)
    cpu_baselines = {
        "granite-3b-q4": {"single": 0.618, "ensemble_2": 0.673, "ensemble_4": 0.764},
        "llama32-3b": {"single": 0.164, "ensemble_2": 0.291, "ensemble_4": 0.200},
        "qwen3-4b-rag": {"single": 0.782, "ensemble_2": 0.836, "ensemble_4": 0.818},
    }
    cpu_lat = {
        "granite-3b-q4": {"single": 3.5, "ensemble_2": 4.7, "ensemble_4": 6.4},
        "llama32-3b": {"single": 3.8, "ensemble_2": 4.7, "ensemble_4": 6.1},
        "qwen3-4b-rag": {"single": 4.2, "ensemble_2": 5.6, "ensemble_4": 8.4},
    }
    for model in MODELS:
        mlabel = model["label"]
        if mlabel not in all_results or mlabel not in cpu_baselines:
            continue
        print(f"  {mlabel}:", flush=True)
        for cfg in CONFIGS:
            clabel = cfg["label"]
            if clabel in all_results[mlabel]["configs"]:
                gpu_acc = all_results[mlabel]["configs"][clabel]["metrics"]["accuracy"]
                gpu_lat = all_results[mlabel]["configs"][clabel]["metrics"]["avg_latency_s"]
                cpu_acc = cpu_baselines[mlabel][clabel]
                cpu_l = cpu_lat[mlabel][clabel]
                d_acc = gpu_acc - cpu_acc
                d_lat = gpu_lat - cpu_l
                print(f"    {clabel:12s}: acc {cpu_acc:.1%} -> {gpu_acc:.1%} ({d_acc:+.1%}) | "
                      f"lat {cpu_l:.1f}s -> {gpu_lat:.1f}s ({d_lat:+.1f}s)", flush=True)

    # Mejor
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
    print(flush=True)
    print(f"Mejor: {best_model} / {best_cfg} = {best_acc:.1%}", flush=True)

    # JSON
    report = {
        "pilot": "coliseo_v1_gpu",
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmark": str(BENCHMARK.name), "case_count": n,
        "execution": "GPU (num_gpu=99)",
        "models": [{"label": m["label"], "name": m["name"], "params": m["params"]} for m in MODELS],
        "configs": [{"label": c["label"], "roles": c["roles"]} for c in CONFIGS],
        "results": all_results,
        "best": {"model": best_model, "config": best_cfg, "accuracy": best_acc},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReport: {OUTPUT}", flush=True)

    # Cleanup: descargar modelos de Ollama para liberar RAM
    print(flush=True)
    print("Descargando modelos de Ollama...", flush=True)
    unload_models([m["name"] for m in MODELS])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
