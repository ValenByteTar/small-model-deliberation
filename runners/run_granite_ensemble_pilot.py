"""
Granite 4.1 3B Q4 — Ensemble de 4 evaluadores semanticos (CPU-only).

Replica el experimento EXP-010 (BitNet ensemble) pero con
ibm/granite4.1:3b-q4_K_M via Ollama, forzado a CPU (num_gpu=0).

Los 4 workers apuntan al mismo Ollama (puerto 11434) con el mismo
modelo pero prompts deliberadamente diferentes. Ollama maneja la
concurrencia internamente.

Comparacion directa contra EXP-010 (BitNet ensemble):
  - Single worker (cada rol individual)
  - Ensemble de 2, 3, 4 workers
  - Correlacion de errores

Ver:
  - knowledge/postmortems/PM-003-bitnet-semantic-capacity-insufficient.md
  - knowledge/experiments/EXP-010-bitnet-ensemble-semantic-capacity.md

Uso:
    python scripts/run_granite_ensemble_pilot.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

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
from hybrid_rag.kernel.state import SemanticAssessment
from hybrid_rag.evaluation.semantic_adapter import SemanticAssessmentAdapter
from hybrid_rag.providers.ollama_provider import OllamaModelProvider

OUTPUT = ROOT / "tests" / "eval" / "canonical" / "reports" / "granite_ensemble_pilot.json"

MODEL = "ibm/granite4.1:3b-q4_K_M"

DATASET = [
    {"id": "s-001", "claim": "The NIST CSF has five core functions",
     "evidence": "The Framework Core consists of five concurrent and continuous Functions: Identify, Detect, Protect, Respond, and Recover.",
     "expected": "SUPPORTS"},
    {"id": "s-002", "claim": "ISO 27001 is an information security standard",
     "evidence": "ISO/IEC 27001 is an international standard for information security management systems (ISMS).",
     "expected": "SUPPORTS"},
    {"id": "s-003", "claim": "SOAR tools help automate security operations",
     "evidence": "Security Orchestration, Automation and Response (SOAR) platforms enable security teams to automate incident response workflows and integrate disparate security tools.",
     "expected": "SUPPORTS"},
    {"id": "c-001", "claim": "Python 4.0 was released in 2023",
     "evidence": "Python 3.12 was released on October 2, 2023. There is no Python 4.0 release.",
     "expected": "CONTRADICTS"},
    {"id": "c-002", "claim": "The NIST CSF has three core functions",
     "evidence": "The Framework Core consists of five concurrent and continuous Functions: Identify, Detect, Protect, Respond, and Recover.",
     "expected": "CONTRADICTS"},
    {"id": "c-003", "claim": "AES-256 is considered weak encryption",
     "evidence": "AES-256 is widely regarded as one of the strongest encryption algorithms available and is approved by NSA for top secret data.",
     "expected": "CONTRADICTS"},
    {"id": "u-001", "claim": "The sky is blue",
     "evidence": "The NIST Cybersecurity Framework provides guidance for organizations to manage cybersecurity risk.",
     "expected": "UNRELATED"},
    {"id": "u-002", "claim": "Water boils at 100 degrees Celsius",
     "evidence": "ISO 27001 requires organizations to conduct regular risk assessments and implement appropriate security controls.",
     "expected": "UNRELATED"},
    {"id": "u-003", "claim": "Paris is the capital of France",
     "evidence": "The CIA triad consists of Confidentiality, Integrity, and Availability, which are the core principles of information security.",
     "expected": "UNRELATED"},
    {"id": "p-001", "claim": "ISO 27001 requires a specific risk assessment methodology",
     "evidence": "The standard states that the organization shall define and apply a risk assessment process, but does not specify a particular methodology.",
     "expected": "PARTIAL"},
    {"id": "p-002", "claim": "The NIST CSF includes 10 functions for cybersecurity",
     "evidence": "The Framework Core includes five Functions. Some organizations extend the framework with additional custom functions.",
     "expected": "PARTIAL"},
    {"id": "p-003", "claim": "Zero trust means no network security controls are needed",
     "evidence": "Zero trust is a security model that assumes no implicit trust and requires verification for every access request. It does not eliminate security controls; it changes how they are applied.",
     "expected": "CONTRADICTS"},
]


def make_workers() -> List[SemanticWorker]:
    """Crea 4 SemanticWorkers, todos apuntando al mismo Ollama con Granite en CPU."""
    workers = []
    for i, role in enumerate(WORKER_ROLES):
        provider = OllamaModelProvider(
            model=MODEL,
            base_url="http://localhost:11434",
            num_gpu=0,  # CPU forzado
            default_options={
                "num_predict": 10,
                "temperature": 0.0,
                "num_thread": 2,
            },
        )
        w = SemanticWorker(
            worker_id=f"worker-{chr(65 + i)}",
            role=role,
            model_provider=provider,
            prompt_fn=WORKER_PROMPTS[role],
        )
        workers.append(w)
    return workers


def run_single_worker(worker: SemanticWorker, dataset: List[dict]) -> List[dict]:
    results = []
    for case in dataset:
        wr = worker.assess(
            claim=case["claim"],
            evidence_text=case["evidence"],
            evidence_id=case["id"],
            run_id="granite-ensemble-pilot",
            timeout=60.0,
        )
        results.append({
            "id": case["id"],
            "expected": case["expected"],
            "produced": wr.relation,
            "confidence": wr.confidence,
            "correct": wr.relation == case["expected"],
            "latency_s": round(wr.latency_s, 2),
            "error": wr.error,
        })
    return results


def run_all_singles_parallel(workers, dataset):
    results = {}
    with ThreadPoolExecutor(max_workers=len(workers)) as pool:
        futures = {pool.submit(run_single_worker, w, dataset): w.worker_id for w in workers}
        for fut in as_completed(futures):
            wid = futures[fut]
            results[wid] = fut.result()
            m = compute_metrics(results[wid])
            print(f"    {wid} done: accuracy={m['accuracy']:.1%} "
                  f"({m['correct']}/{m['n']}) avg_lat={m['avg_latency_s']:.1f}s",
                  flush=True)
    return results


def run_ensemble(ensemble, dataset):
    results = []
    for case in dataset:
        assessment, worker_results, meta = ensemble.assess(
            claim=case["claim"],
            evidence_text=case["evidence"],
            evidence_id=case["id"],
            run_id="granite-ensemble-pilot",
            timeout=60.0,
        )
        produced = assessment.relation if assessment else "ERROR"
        results.append({
            "id": case["id"],
            "expected": case["expected"],
            "produced": produced,
            "confidence": assessment.confidence if assessment else 0.0,
            "correct": produced == case["expected"],
            "agreement": meta.get("agreement", 0.0),
            "agreement_fraction": meta.get("agreement_fraction", "?"),
            "total_latency_s": meta.get("total_latency_s", 0.0),
            "worker_votes": meta.get("votes", []),
            "worker_latencies": meta.get("worker_latencies_s", {}),
        })
    return results


def compute_metrics(results):
    n = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    protocol_ok = sum(1 for r in results if r.get("produced") in
                      ("SUPPORTS", "CONTRADICTS", "UNRELATED", "PARTIAL"))
    latencies = [r.get("latency_s", r.get("total_latency_s", 0.0)) for r in results]
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    total_lat = sum(latencies)
    relation_dist = Counter(r.get("produced", "ERROR") for r in results)
    return {
        "n": n,
        "accuracy": round(correct / n, 4) if n > 0 else 0.0,
        "correct": correct,
        "protocol_validity": round(protocol_ok / n, 4) if n > 0 else 0.0,
        "avg_latency_s": round(avg_lat, 2),
        "total_latency_s": round(total_lat, 2),
        "relation_distribution": dict(relation_dist),
    }


def compute_error_correlation(single_results, dataset):
    worker_names = list(single_results.keys())
    n = len(dataset)
    error_sets = {}
    for wname, results in single_results.items():
        error_sets[wname] = {i for i, r in enumerate(results) if not r.get("correct")}

    correlations = {}
    for i, w1 in enumerate(worker_names):
        for w2 in worker_names[i + 1:]:
            e1, e2 = error_sets[w1], error_sets[w2]
            both = len(e1 & e2)
            union = len(e1 | e2)
            jaccard = both / union if union > 0 else 0.0
            correlations[f"{w1}__{w2}"] = {
                "both_wrong": both,
                "only_w1_wrong": len(e1 - e2),
                "only_w2_wrong": len(e2 - e1),
                "both_right": n - len(e1 | e2),
                "error_jaccard": round(jaccard, 3),
            }

    per_case = []
    for i, case in enumerate(dataset):
        n_wrong = sum(1 for w in worker_names if i in error_sets[w])
        per_case.append({
            "id": case["id"],
            "expected": case["expected"],
            "n_workers_wrong": n_wrong,
            "n_workers_total": len(worker_names),
        })

    return {
        "pairwise": correlations,
        "per_case": per_case,
        "error_sets": {w: sorted(list(s)) for w, s in error_sets.items()},
    }


def main() -> int:
    print("=" * 70, flush=True)
    print("Granite 4.1 3B Q4 — Ensemble de 4 evaluadores (CPU-only)", flush=True)
    print(f"Model: {MODEL} (num_gpu=0, 4 workers mismo Ollama)", flush=True)
    print("=" * 70, flush=True)
    print(f"Dataset: {len(DATASET)} claim-evidence pairs", flush=True)
    print(flush=True)

    # Verificar modelo disponible
    probe = OllamaModelProvider(model=MODEL, base_url="http://localhost:11434", num_gpu=0)
    if not probe.is_available():
        print(f"ERROR: Modelo {MODEL} no disponible en Ollama", flush=True)
        return 1
    print(f"  {MODEL} disponible", flush=True)
    print(flush=True)

    workers = make_workers()
    for w in workers:
        print(f"  {w.worker_id} ({w.role}) -> Ollama localhost:11434, CPU", flush=True)
    print(flush=True)

    # ==================== Fase 1: Single workers en paralelo ====================
    print("=" * 70, flush=True)
    print("FASE 1: Single worker (4 workers en paralelo)", flush=True)
    print("=" * 70, flush=True)

    single_results = run_all_singles_parallel(workers, DATASET)
    single_metrics = {}
    for wid, results in single_results.items():
        single_metrics[wid] = compute_metrics(results)
        m = single_metrics[wid]
        print(f"  {wid}: accuracy={m['accuracy']:.1%} ({m['correct']}/{m['n']}) "
              f"protocol={m['protocol_validity']:.0%} avg_lat={m['avg_latency_s']:.1f}s",
              flush=True)
        print(f"    distribution: {m['relation_distribution']}", flush=True)

    # ==================== Fase 2: Ensembles ====================
    print(flush=True)
    print("=" * 70, flush=True)
    print("FASE 2: Ensembles (2, 3, 4 workers)", flush=True)
    print("=" * 70, flush=True)

    ensemble_configs = {
        "ensemble_2_AB": [workers[0], workers[1]],
        "ensemble_2_CD": [workers[2], workers[3]],
        "ensemble_3_ABC": [workers[0], workers[1], workers[2]],
        "ensemble_4_ABCD": [workers[0], workers[1], workers[2], workers[3]],
    }

    ensemble_results = {}
    ensemble_metrics = {}

    for name, ws in ensemble_configs.items():
        print(f"\n  Ejecutando {name} ({len(ws)} workers)...", flush=True)
        ensemble = SemanticEnsemble(workers=ws, strategy=ConfidenceWeightedMajorityVote())
        results = run_ensemble(ensemble, DATASET)
        ensemble_results[name] = results
        m = compute_metrics(results)
        ensemble_metrics[name] = m
        agreements = [r.get("agreement", 0.0) for r in results]
        avg_agreement = sum(agreements) / len(agreements) if agreements else 0.0
        print(f"    accuracy={m['accuracy']:.1%} ({m['correct']}/{m['n']}) "
              f"protocol={m['protocol_validity']:.0%} avg_lat={m['avg_latency_s']:.1f}s "
              f"avg_agreement={avg_agreement:.2f}", flush=True)
        print(f"    distribution: {m['relation_distribution']}", flush=True)

    # ==================== Fase 3: Correlacion de errores ====================
    print(flush=True)
    print("=" * 70, flush=True)
    print("FASE 3: Correlacion de errores", flush=True)
    print("=" * 70, flush=True)

    error_corr = compute_error_correlation(single_results, DATASET)

    print("\nCorrelacion par a par (error Jaccard):", flush=True)
    for pair, stats in error_corr["pairwise"].items():
        print(f"  {pair}: both_wrong={stats['both_wrong']} "
              f"jaccard={stats['error_jaccard']:.2f}", flush=True)

    print("\nPor caso (cuantos workers fallaron):", flush=True)
    for ci in error_corr["per_case"]:
        print(f"  {ci['id']} (expected={ci['expected']}): "
              f"{ci['n_workers_wrong']}/{ci['n_workers_total']} wrong", flush=True)

    # ==================== Deliverable ====================
    print(flush=True)
    print("=" * 70, flush=True)
    print("DELIVERABLE — Resumen completo", flush=True)
    print("=" * 70, flush=True)

    print("\n1. ARQUITECTURA", flush=True)
    print("   4x OllamaModelProvider(Granite-3B-Q4, CPU) -> SemanticWorker(A/B/C/D)", flush=True)
    print("   -> ConfidenceWeightedMajorityVote -> SemanticAssessment", flush=True)

    print("\n2. CONFIGURACION", flush=True)
    for w in workers:
        print(f"   {w.worker_id} ({w.role}): Ollama localhost:11434, CPU, num_thread=2",
              flush=True)

    print("\n3. RESULTADOS POR WORKER", flush=True)
    for w in workers:
        m = single_metrics[w.worker_id]
        print(f"   {w.worker_id} ({w.role}): accuracy={m['accuracy']:.1%} "
              f"protocol={m['protocol_validity']:.0%} avg_lat={m['avg_latency_s']:.1f}s",
              flush=True)

    print("\n4. RESULTADOS DEL ENSEMBLE", flush=True)
    for name in ensemble_configs:
        m = ensemble_metrics[name]
        print(f"   {name}: accuracy={m['accuracy']:.1%} "
              f"protocol={m['protocol_validity']:.0%} avg_lat={m['avg_latency_s']:.1f}s",
              flush=True)

    print("\n5. COMPARACION", flush=True)
    print(f"   BitNet-2B single:          33.3%", flush=True)
    print(f"   BitNet-2B best single (D): 50.0%", flush=True)
    print(f"   BitNet-2B ensemble 4:      41.7%", flush=True)
    print(f"   Granite-3B single (pilot): 91.7%", flush=True)
    for w in workers:
        m = single_metrics[w.worker_id]
        print(f"   Granite-3B {w.worker_id}:      {m['accuracy']:.1%}", flush=True)
    for name in ensemble_configs:
        m = ensemble_metrics[name]
        print(f"   Granite-3B {name}:  {m['accuracy']:.1%}", flush=True)

    print("\n6. CORRELACION DE ERRORES", flush=True)
    for pair, stats in error_corr["pairwise"].items():
        print(f"   {pair}: both_wrong={stats['both_wrong']} "
              f"jaccard={stats['error_jaccard']:.2f}", flush=True)

    print("\n7. CONCLUSION", flush=True)
    best_single = max(single_metrics.values(), key=lambda m: m['accuracy'])
    best_ens_name = max(ensemble_metrics, key=lambda n: ensemble_metrics[n]['accuracy'])
    best_ens = ensemble_metrics[best_ens_name]
    print(f"   Best single: {best_single['accuracy']:.1%}", flush=True)
    print(f"   Best ensemble ({best_ens_name}): {best_ens['accuracy']:.1%}", flush=True)
    if best_ens['accuracy'] > best_single['accuracy']:
        print(f"   Ensemble aporta: +{best_ens['accuracy'] - best_single['accuracy']:.1%}",
              flush=True)
    else:
        print(f"   Ensemble no supera al best single "
              f"({best_ens['accuracy'] - best_single['accuracy']:.1%})", flush=True)

    # Reporte JSON
    report = {
        "pilot": "granite_ensemble_4_workers",
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": MODEL,
        "quantization": "Q4_K_M",
        "parameters": "3.4B",
        "execution": "CPU-only (num_gpu=0), 4 workers mismo Ollama",
        "dataset_size": len(DATASET),
        "baselines": {
            "bitnet_single": 0.333,
            "bitnet_best_single": 0.50,
            "bitnet_ensemble_4": 0.417,
            "granite_single_pilot": 0.917,
        },
        "single_worker_results": {
            w.worker_id: {
                "role": w.role,
                "metrics": single_metrics[w.worker_id],
                "per_case": single_results[w.worker_id],
            }
            for w in workers
        },
        "ensemble_results": {
            name: {"metrics": ensemble_metrics[name], "per_case": ensemble_results[name]}
            for name in ensemble_configs
        },
        "error_correlation": error_corr,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n   Report saved: {OUTPUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
