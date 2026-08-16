"""
DEPRECATED — Experimento documentado, no reejecutar sin leer PM-003.

Este script es parte del experimento LLMSupport ensemble con 4
instancias de BitNet-b1.58-2B-4T. El experimento fallo: el ensemble
(41.7%) no supera al mejor worker individual (50%), y la correlacion
de errores es alta (Jaccard 0.40-0.64).

Resultado: ADR-0031 deprecado, LLMSupport desacoplado del pipeline.

Ver:
  - knowledge/postmortems/PM-003-bitnet-semantic-capacity-insufficient.md
  - knowledge/experiments/EXP-010-bitnet-ensemble-semantic-capacity.md

---

Semantic Ensemble Pilot — 4 BitNet workers independientes.

Mide si un ensemble de 4 evaluadores semanticos especializados mejora
la accuracy respecto del baseline de 33% (single worker).

Comparacion:
  - Single worker (cada rol individual)
  - Ensemble de 2 workers (A+B, A+C, A+D, etc.)
  - Ensemble de 3 workers
  - Ensemble de 4 workers

Metricas:
  - accuracy
  - protocol validity
  - relation distribution
  - agreement
  - latency
  - RAM
  - CPU utilization
  - correlation of errors

Uso:
    python scripts/run_ensemble_pilot.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
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
from hybrid_rag.kernel.state import SemanticAssessment
from hybrid_rag.evaluation.semantic_adapter import SemanticAssessmentAdapter
from hybrid_rag.providers.bitnet_provider import BitNetModelProvider

OUTPUT = ROOT / "tests" / "eval" / "canonical" / "reports" / "ensemble_pilot.json"

# Mismo dataset que run_semantic_pilot.py
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

BITNET_ROOT = "C:/Users/Valen/Desktop/Proyectos/BitNet"
MODEL_PATH = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
SERVER_PATH = "build/bin/Release/llama-server.exe"
BASE_PORT = 8081


def start_workers(n: int) -> List[BitNetModelProvider]:
    """Inicia n instancias de BitNet en puertos consecutivos, cada una con 1 thread."""
    providers = []
    for i in range(n):
        port = BASE_PORT + i
        p = BitNetModelProvider(
            model_path=MODEL_PATH,
            server_path=SERVER_PATH,
            bitnet_root=BITNET_ROOT,
            port=port,
            threads=1,  # 1 thread por instancia
            ctx_size=2048,
        )
        if not p.ensure_running(timeout=30.0):
            print(f"ERROR: No se pudo iniciar worker {i} en puerto {port}")
            # Cleanup already started
            for pp in providers:
                pp.shutdown()
            return []
        providers.append(p)
        print(f"  Worker {i} activo en puerto {port}")
    return providers


def stop_all(providers: List[BitNetModelProvider]) -> None:
    for p in providers:
        p.shutdown()


def get_ram_usage_mb() -> float:
    """Retorna RAM usada por los procesos llama-server."""
    try:
        import psutil
        total = 0.0
        for proc in psutil.process_iter(["name", "memory_info"]):
            name = proc.info.get("name", "")
            if "llama-server" in name.lower():
                mi = proc.info.get("memory_info")
                if mi:
                    total += mi.rss / (1024 * 1024)
        return round(total, 1)
    except Exception:
        return -1.0


def get_cpu_percent() -> float:
    """Retorna CPU usage promedio de los procesos llama-server."""
    try:
        import psutil
        cpus = []
        for proc in psutil.process_iter(["name", "cpu_percent"]):
            name = proc.info.get("name", "")
            if "llama-server" in name.lower():
                cpus.append(proc.info.get("cpu_percent", 0.0))
        return round(sum(cpus) / len(cpus), 1) if cpus else 0.0
    except Exception:
        return -1.0


def run_single_worker(
    worker: SemanticWorker,
    dataset: List[dict],
) -> List[dict]:
    """Ejecuta un worker individual sobre el dataset."""
    results = []
    for case in dataset:
        wr = worker.assess(
            claim=case["claim"],
            evidence_text=case["evidence"],
            evidence_id=case["id"],
            run_id="ensemble-pilot",
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


def run_all_singles_parallel(
    workers: List[SemanticWorker],
    dataset: List[dict],
) -> Dict[str, List[dict]]:
    """Ejecuta todos los workers en paralelo (cada uno sobre el dataset completo)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    results = {}
    with ThreadPoolExecutor(max_workers=len(workers)) as pool:
        futures = {
            pool.submit(run_single_worker, w, dataset): w.worker_id
            for w in workers
        }
        for fut in as_completed(futures):
            wid = futures[fut]
            results[wid] = fut.result()
            m = compute_metrics(results[wid])
            print(f"    {wid} done: accuracy={m['accuracy']:.1%} "
                  f"({m['correct']}/{m['n']}) avg_lat={m['avg_latency_s']:.1f}s",
                  flush=True)
    return results



def run_ensemble(
    ensemble: SemanticEnsemble,
    dataset: List[dict],
) -> List[dict]:
    """Ejecuta el ensemble sobre el dataset."""
    results = []
    for case in dataset:
        assessment, worker_results, meta = ensemble.assess(
            claim=case["claim"],
            evidence_text=case["evidence"],
            evidence_id=case["id"],
            run_id="ensemble-pilot",
            timeout=60.0,
        )
        produced = assessment.relation if assessment else "ERROR"
        correct = produced == case["expected"]
        results.append({
            "id": case["id"],
            "expected": case["expected"],
            "produced": produced,
            "confidence": assessment.confidence if assessment else 0.0,
            "correct": correct,
            "agreement": meta.get("agreement", 0.0),
            "agreement_fraction": meta.get("agreement_fraction", "?"),
            "total_latency_s": meta.get("total_latency_s", 0.0),
            "worker_votes": meta.get("votes", []),
            "worker_latencies": meta.get("worker_latencies_s", {}),
        })
    return results


def compute_metrics(results: List[dict]) -> dict:
    n = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    protocol_ok = sum(1 for r in results if r.get("produced") in ("SUPPORTS", "CONTRADICTS", "UNRELATED", "PARTIAL"))
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


def compute_error_correlation(
    single_results: Dict[str, List[dict]],
    dataset: List[dict],
) -> dict:
    """
    Computa correlacion de errores entre workers.

    Para cada par de workers, cuenta en cuantos casos ambos fallaron
    (interseccion de errores) vs errores individuales.
    """
    worker_names = list(single_results.keys())
    n = len(dataset)

    # Para cada worker, conjunto de indices donde fallo
    error_sets = {}
    for wname, results in single_results.items():
        error_sets[wname] = {i for i, r in enumerate(results) if not r.get("correct")}

    # Correlacion par a par
    correlations = {}
    for i, w1 in enumerate(worker_names):
        for w2 in worker_names[i + 1:]:
            e1 = error_sets[w1]
            e2 = error_sets[w2]
            both = len(e1 & e2)
            only1 = len(e1 - e2)
            only2 = len(e2 - e1)
            neither = n - len(e1 | e2)
            # Jaccard similarity of error sets
            union = len(e1 | e2)
            jaccard = both / union if union > 0 else 0.0
            correlations[f"{w1}__{w2}"] = {
                "both_wrong": both,
                "only_w1_wrong": only1,
                "only_w2_wrong": only2,
                "both_right": neither,
                "error_jaccard": round(jaccard, 3),
            }

    # Por caso: cuantos workers fallaron
    per_case_errors = []
    for i, case in enumerate(dataset):
        n_wrong = sum(1 for w in worker_names if i in error_sets[w])
        per_case_errors.append({
            "id": case["id"],
            "expected": case["expected"],
            "n_workers_wrong": n_wrong,
            "n_workers_total": len(worker_names),
        })

    return {
        "pairwise": correlations,
        "per_case": per_case_errors,
        "error_sets": {w: sorted(list(s)) for w, s in error_sets.items()},
    }


def main() -> int:
    print("=" * 70)
    print("Semantic Ensemble Pilot: 4 BitNet workers independientes")
    print("=" * 70)
    print(f"Dataset: {len(DATASET)} claim-evidence pairs")
    print()

    # Iniciar 4 instancias de BitNet
    print("Iniciando 4 instancias de BitNet (1 thread cada una)...")
    providers = start_workers(4)
    if len(providers) < 4:
        print("ERROR: No se pudieron iniciar las 4 instancias")
        stop_all(providers)
        return 1

    print(f"RAM inicial: {get_ram_usage_mb():.0f} MB")
    print()

    # Crear 4 SemanticWorkers, cada uno con un prompt diferente
    workers = []
    for i, role in enumerate(WORKER_ROLES):
        w = SemanticWorker(
            worker_id=f"worker-{chr(65 + i)}",  # worker-A, worker-B, etc.
            role=role,
            model_provider=providers[i],
            prompt_fn=WORKER_PROMPTS[role],
        )
        workers.append(w)
        print(f"  {w.worker_id} ({role}) -> puerto {BASE_PORT + i}")

    print()

    # ==================== Fase 1: Single worker (cada rol individual, en paralelo) ====================
    print("=" * 70, flush=True)
    print("FASE 1: Single worker (baseline individual, 4 workers en paralelo)", flush=True)
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
    print()
    print("=" * 70)
    print("FASE 2: Ensembles (2, 3, 4 workers)")
    print("=" * 70)

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
        ensemble = SemanticEnsemble(
            workers=ws,
            strategy=ConfidenceWeightedMajorityVote(),
        )
        results = run_ensemble(ensemble, DATASET)
        ensemble_results[name] = results
        m = compute_metrics(results)
        ensemble_metrics[name] = m

        # Agreement promedio
        agreements = [r.get("agreement", 0.0) for r in results]
        avg_agreement = sum(agreements) / len(agreements) if agreements else 0.0

        print(f"    accuracy={m['accuracy']:.1%} ({m['correct']}/{m['n']}) "
              f"protocol={m['protocol_validity']:.0%} avg_lat={m['avg_latency_s']:.1f}s "
              f"avg_agreement={avg_agreement:.2f}", flush=True)
        print(f"    distribution: {m['relation_distribution']}", flush=True)

    # ==================== Fase 3: Correlacion de errores ====================
    print()
    print("=" * 70)
    print("FASE 3: Correlacion de errores")
    print("=" * 70)

    error_corr = compute_error_correlation(single_results, DATASET)

    print("\nCorrelacion par a par (error Jaccard):")
    for pair, stats in error_corr["pairwise"].items():
        print(f"  {pair}: both_wrong={stats['both_wrong']} "
              f"jaccard={stats['error_jaccard']:.2f}")

    print("\nPor caso (cuantos workers fallaron):")
    for case_info in error_corr["per_case"]:
        print(f"  {case_info['id']} (expected={case_info['expected']}): "
              f"{case_info['n_workers_wrong']}/{case_info['n_workers_total']} wrong")

    # ==================== Fase 4: Resource usage ====================
    print()
    print("=" * 70)
    print("FASE 4: Resource usage")
    print("=" * 70)

    ram = get_ram_usage_mb()
    cpu = get_cpu_percent()
    print(f"  RAM (llama-server total): {ram:.0f} MB")
    print(f"  CPU (llama-server avg): {cpu:.1f}%")

    # ==================== Shutdown ====================
    stop_all(providers)

    # ==================== Reporte ====================
    report = {
        "pilot": "semantic_ensemble_4_workers",
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset_size": len(DATASET),
        "baseline_single_worker_accuracy": 0.333,  # del pilot anterior
        "worker_config": {
            "model": "BitNet-b1.58-2B-4T",
            "threads_per_instance": 1,
            "instances": 4,
            "ports": [BASE_PORT + i for i in range(4)],
            "roles": WORKER_ROLES,
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
            name: {
                "metrics": ensemble_metrics[name],
                "per_case": ensemble_results[name],
            }
            for name in ensemble_configs
        },
        "error_correlation": error_corr,
        "resource_usage": {
            "ram_mb": ram,
            "cpu_percent": cpu,
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 70)
    print("DELIVERABLE — Resumen completo")
    print("=" * 70)

    # 1. Arquitectura
    print("\n1. ARQUITECTURA IMPLEMENTADA")
    print("   SemanticAssessment -> SemanticEnsemble (4 workers paralelo)")
    print("   -> ConfidenceWeightedMajorityVote -> SemanticAssessment")
    print("   -> SemanticAssessmentAdapter -> EvaluationSignal -> PolicyEngine")
    print("   Frontera intacta: LLMSupport produce opinion, Kernel decide")

    # 2. Configuracion
    print("\n2. CONFIGURACION DE CADA WORKER")
    for w in workers:
        print(f"   {w.worker_id} ({w.role}): puerto {BASE_PORT + ord(w.worker_id[-1]) - ord('A')}, 1 thread")

    # 3. Resultados por worker
    print("\n3. RESULTADOS POR WORKER")
    for w in workers:
        m = single_metrics[w.worker_id]
        print(f"   {w.worker_id} ({w.role}): accuracy={m['accuracy']:.1%} "
              f"protocol={m['protocol_validity']:.0%} avg_lat={m['avg_latency_s']:.1f}s")

    # 4. Resultados del ensemble
    print("\n4. RESULTADOS DEL ENSEMBLE")
    for name in ensemble_configs:
        m = ensemble_metrics[name]
        print(f"   {name}: accuracy={m['accuracy']:.1%} "
              f"protocol={m['protocol_validity']:.0%} avg_lat={m['avg_latency_s']:.1f}s")

    # 5. Comparacion contra baseline
    print("\n5. COMPARACION CONTRA BASELINE 33%")
    print(f"   Baseline (single, pilot anterior): 33.3%")
    for w in workers:
        m = single_metrics[w.worker_id]
        delta = m['accuracy'] - 0.333
        print(f"   {w.worker_id}: {m['accuracy']:.1%} (delta={delta:+.1%})")
    for name in ["ensemble_2_AB", "ensemble_2_CD", "ensemble_3_ABC", "ensemble_4_ABCD"]:
        m = ensemble_metrics[name]
        delta = m['accuracy'] - 0.333
        print(f"   {name}: {m['accuracy']:.1%} (delta={delta:+.1%})")

    # 6. Agreement y correlacion
    print("\n6. AGREEMENT Y CORRELACION DE ERRORES")
    for pair, stats in error_corr["pairwise"].items():
        print(f"   {pair}: both_wrong={stats['both_wrong']} jaccard={stats['error_jaccard']:.2f}")

    # 7. RAM/CPU/latencia
    print(f"\n7. RAM/CPU/LATENCIA")
    print(f"   RAM: {ram:.0f} MB (4 instancias)")
    print(f"   CPU: {cpu:.1f}% avg")
    for name in ["ensemble_4_ABCD"]:
        m = ensemble_metrics[name]
        print(f"   {name} avg latency: {m['avg_latency_s']:.1f}s/case")

    # 8. Conclusion
    print("\n8. CONCLUSION")
    best_single = max(single_metrics.values(), key=lambda m: m['accuracy'])
    best_ensemble_name = max(ensemble_metrics, key=lambda n: ensemble_metrics[n]['accuracy'])
    best_ensemble = ensemble_metrics[best_ensemble_name]
    print(f"   Best single: {best_single['accuracy']:.1%}")
    print(f"   Best ensemble ({best_ensemble_name}): {best_ensemble['accuracy']:.1%}")
    improvement = best_ensemble['accuracy'] - best_single['accuracy']
    if improvement > 0.05:
        print(f"   El ensemble aporta valor: +{improvement:.1%} sobre best single")
    elif best_ensemble['accuracy'] > 0.333 + 0.05:
        print(f"   El ensemble supera el baseline 33%: +{best_ensemble['accuracy'] - 0.333:.1%}")
    else:
        print(f"   El ensemble NO aporta mejora significativa sobre single")

    print(f"\n   Report saved: {OUTPUT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
