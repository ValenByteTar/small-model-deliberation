"""
Coliseo v2: 4 modelos Ollama (GPU) vs SemanticAssessment Benchmark v2.

Segunda ronda de gladiadores. Corre en GPU para ganar velocidad.
Los modelos se evaluan individualmente y en ensemble (2 y 4 workers).

Modelos:
  - gemma3:4b                    (4B, Google, Q4)
  - dhiltgen/nemotron-3-nano:4b  (4B, NVIDIA, Q4_K_M)
  - ministral-3:3b               (3B, Mistral, Q4)
  - qwen3.5:4b                   (4B, Alibaba, Q4)

Todo corre en GPU (num_gpu=99) para maximizar throughput.
Si se necesita comparar latencia CPU, se hace despues con el modelo ganador.

Uso:
    python runners/run_coliseo_v2_gpu.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def unload_models(model_names, base_url="http://localhost:11434"):
    """Descarga los modelos de Ollama (keep_alive=0) para liberar RAM."""
    for name in model_names:
        try:
            requests.post(
                f"{base_url}/api/generate",
                json={"model": name, "keep_alive": 0},
                timeout=10,
            )
            print(f"  unloaded: {name}", flush=True)
        except Exception as e:
            print(f"  unload failed: {name} ({e})", flush=True)

from hybrid_rag.kernel.semantic_ensemble import (
    ConfidenceWeightedMajorityVote,
    SemanticEnsemble,
    SemanticWorker,
    WORKER_PROMPTS,
    WORKER_ROLES,
)
from hybrid_rag.kernel.state import SEMANTIC_RELATIONS
from hybrid_rag.providers.ollama_provider import OllamaModelProvider


class NoThinkOllamaModelProvider(OllamaModelProvider):
    def generate(
        self,
        prompt: str,
        *,
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> str:
        opts = {"num_gpu": self.num_gpu, **self.default_options}
        if options:
            opts.update(options)
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": opts,
                "keep_alive": "10m",
            },
            timeout=timeout or 120,
        )
        response.raise_for_status()
        return (response.json().get("response") or "").strip()


BENCHMARK = ROOT / "benchmarks" / "semantic_assessment_v2.json"
OUTPUT = ROOT / "results" / "raw" / "coliseo_v2_gpu.json"

MODELS = [
    {"name": "gemma3:4b-it-q4_K_M",                     "label": "gemma3-4b-q4",      "params": "4B"},
    {"name": "dhiltgen/nemotron-3-nano:4b",             "label": "nemotron-3-4b-q4",  "params": "4B"},
    {"name": "TechyShishy/ministral-3:3b-reasoning-2512-q4_K_M", "label": "ministral-3b-q4", "params": "3B"},
    {"name": "qwen3.5:4b-q4_K_M",                       "label": "qwen35-4b-q4",      "params": "4B"},
]

CONFIGS = [
    {"label": "single",       "roles": ["neutral"]},
    {"label": "ensemble_2",   "roles": ["entailment", "skeptical"]},
    {"label": "ensemble_4",   "roles": ["entailment", "skeptical", "contradiction", "neutral"]},
]


def make_provider(model_name: str, base_url: str = "http://localhost:11434") -> OllamaModelProvider:
    reasoning_models = ("nemotron", "ministral", "qwen3.5")
    provider_cls = (
        NoThinkOllamaModelProvider
        if any(name in model_name.lower() for name in reasoning_models)
        else OllamaModelProvider
    )
    return provider_cls(
        model=model_name,
        base_url=base_url,
        num_gpu=99,  # GPU forzado
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
    workers = make_workers(model_name, roles, base_url=base_url)
    total_time = 0.0
    results = []

    if len(workers) == 1:
        w = workers[0]
        for case in cases:
            wr = w.assess(
                claim=case["claim"], evidence_text=case["evidence"],
                evidence_id=case["id"], run_id="coliseo-v2-gpu", timeout=90.0,
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
                evidence_id=case["id"], run_id="coliseo-v2-gpu", timeout=90.0,
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
    parser = argparse.ArgumentParser(description="Coliseo v2 GPU")
    parser.add_argument("--port", type=int, default=11434,
                        help="Ollama instance port (default: 11434)")
    args = parser.parse_args()
    base_url = f"http://localhost:{args.port}"

    print("=" * 70, flush=True)
    print("COLOSEO v2: 4 modelos vs SemanticAssessment Benchmark v2", flush=True)
    print("GPU (num_gpu=99) | 55 casos | 10 categorias diagnosticas", flush=True)
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
    print(f"Execution: GPU (num_gpu=99)", flush=True)
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

            if metrics["correct"] < n:
                print(f"    Errores por categoria:", flush=True)
                for cat, s in sorted(metrics["by_category"].items()):
                    if s["correct"] < s["total"]:
                        print(f"      {cat:25s}: {s['correct']}/{s['total']}", flush=True)

        print(flush=True)

    # ==================== Tabla comparativa final ====================
    print("=" * 70, flush=True)
    print("TABLA COMPARATIVA FINAL — COLOSEO v2 (GPU)", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)

    # Accuracy
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

    # Latencia
    print("Latencia promedio (s):", flush=True)
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

    # Ensemble vs single
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

    # Por categoria del mejor
    print(flush=True)
    print(f"Categorias diagnosticas — {best_model} / {best_cfg}:", flush=True)
    if best_model and best_cfg:
        cats = all_results[best_model]["configs"][best_cfg]["metrics"]["by_category"]
        for cat, s in sorted(cats.items()):
            acc = s["accuracy"]
            bar = "=" * int(acc * 20)
            print(f"  {cat:25s}: {s['correct']:2d}/{s['total']:2d} = {acc:5.1%} {bar}", flush=True)

    # Comparacion contra coliseo v1
    print(flush=True)
    print("Comparacion contra Coliseo v1 (CPU):", flush=True)
    print("  granite-3b-q4 single:  61.8% (CPU)", flush=True)
    print("  granite-3b-q4 ens4:    76.4% (CPU)", flush=True)
    print("  qwen3-4b-rag single:   78.2% (CPU)", flush=True)
    print("  llama32-3b single:     16.4% (CPU)", flush=True)
    for model in MODELS:
        mlabel = model["label"]
        if mlabel in all_results:
            for cfg in CONFIGS:
                clabel = cfg["label"]
                if clabel in all_results[mlabel]["configs"]:
                    acc = all_results[mlabel]["configs"][clabel]["metrics"]["accuracy"]
                    print(f"  {mlabel} {clabel}: {acc:.1%} (GPU)", flush=True)

    # Reporte JSON
    report = {
        "pilot": "coliseo_v2_gpu",
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmark": str(BENCHMARK.name),
        "case_count": n,
        "execution": "GPU (num_gpu=99)",
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

    # Cleanup: descargar modelos de Ollama para liberar RAM
    print(flush=True)
    print("Descargando modelos de Ollama...", flush=True)
    unload_models([m["name"] for m in MODELS], base_url=base_url)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
