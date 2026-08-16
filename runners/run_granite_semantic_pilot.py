"""
Granite 4.1 3B Q4 — Semantic Capability Provider reevaluation pilot.

Reevalua LLMSupport como Semantic Capability Provider usando
ibm/granite4.1:3b-q4_K_M via Ollama, forzado a CPU (num_gpu=0).

Motivacion: PM-003 documento que BitNet-b1.58-2B-4T no tiene capacidad
semantica suficiente (33% accuracy). Granite 4.1 3B (3.4B params, Q4_K_M)
es un modelo denso mas capaz que BitNet-2B ternario. Este pilot determina
si supera el criterio de >60% accuracy establecido en PM-003.

Mismo dataset de 12 pares claim-evidence que EXP-010 para comparacion
directa contra BitNet.

Ver:
  - knowledge/postmortems/PM-003-bitnet-semantic-capacity-insufficient.md
  - knowledge/experiments/EXP-010-bitnet-ensemble-semantic-capacity.md

Uso:
    python scripts/run_granite_semantic_pilot.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hybrid_rag.kernel.llm_support import LLMSupport
from hybrid_rag.kernel.state import SemanticAssessment, SEMANTIC_RELATIONS
from hybrid_rag.evaluation.semantic_adapter import SemanticAssessmentAdapter
from hybrid_rag.providers.ollama_provider import OllamaModelProvider

OUTPUT = ROOT / "tests" / "eval" / "canonical" / "reports" / "granite_semantic_pilot.json"

MODEL = "ibm/granite4.1:3b-q4_K_M"

# Mismo dataset que EXP-010 para comparacion directa
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


def main() -> int:
    print("=" * 70, flush=True)
    print("Granite 4.1 3B Q4 — Semantic Capability Provider Reevaluation", flush=True)
    print(f"Model: {MODEL} (CPU-only, num_gpu=0)", flush=True)
    print("=" * 70, flush=True)
    print(f"Dataset: {len(DATASET)} claim-evidence pairs", flush=True)
    print(f"  SUPPORTS: {sum(1 for d in DATASET if d['expected'] == 'SUPPORTS')}", flush=True)
    print(f"  CONTRADICTS: {sum(1 for d in DATASET if d['expected'] == 'CONTRADICTS')}", flush=True)
    print(f"  UNRELATED: {sum(1 for d in DATASET if d['expected'] == 'UNRELATED')}", flush=True)
    print(f"  PARTIAL: {sum(1 for d in DATASET if d['expected'] == 'PARTIAL')}", flush=True)
    print(flush=True)

    # OllamaModelProvider con num_gpu=0 -> CPU forzado
    # No compite con el modelo principal del pipeline (GPU)
    print("Inicializando OllamaModelProvider (CPU-only)...", flush=True)
    provider = OllamaModelProvider(
        model=MODEL,
        base_url="http://localhost:11434",
        num_gpu=0,  # CPU forzado
        default_options={
            "num_predict": 10,
            "temperature": 0.0,
            "num_thread": 4,
        },
    )

    if not provider.is_available():
        print(f"ERROR: Modelo {MODEL} no disponible en Ollama", flush=True)
        return 1

    print(f"  {MODEL} disponible", flush=True)
    print(flush=True)

    # LLMSupport en modo semantic (reutiliza el mismo componente)
    llm = LLMSupport(
        model_provider=provider,
        mode="semantic",
        max_hypotheses=100,
        max_concurrent=1,
    )

    adapter = SemanticAssessmentAdapter()

    results = []
    correct = 0
    protocol_ok = 0
    total_time = 0.0

    for i, case in enumerate(DATASET, 1):
        cid = case["id"]
        claim = case["claim"]
        evidence = case["evidence"]
        expected = case["expected"]

        print(f"[{i}/{len(DATASET)}] {cid}: claim='{claim[:50]}...' expected={expected}",
              flush=True)

        t0 = time.time()
        assessment = llm.semantic_assess(
            claim=claim,
            evidence_text=evidence,
            evidence_id=cid,
            run_id="granite-semantic-pilot",
            timeout=60.0,
        )
        dt = time.time() - t0
        total_time += dt

        if assessment is None:
            print(f"  FAIL: no assessment produced ({dt:.1f}s)", flush=True)
            results.append({
                "id": cid, "claim": claim, "evidence": evidence[:200],
                "expected": expected, "produced": None,
                "correct": False, "protocol_ok": False,
                "timing_s": round(dt, 2),
            })
            continue

        is_correct = assessment.relation == expected
        if is_correct:
            correct += 1

        is_protocol = (
            assessment.relation in SEMANTIC_RELATIONS
            and 0.0 <= assessment.confidence <= 1.0
        )
        if is_protocol:
            protocol_ok += 1

        signal = adapter.adapt(assessment)

        print(
            f"  relation={assessment.relation} conf={assessment.confidence:.2f} "
            f"correct={is_correct} protocol={is_protocol} "
            f"signal: passed={signal.passed} score={signal.score:.2f} "
            f"({dt:.1f}s)",
            flush=True,
        )
        if assessment.reasoning:
            print(f"  reasoning: {assessment.reasoning[:100]}", flush=True)

        results.append({
            "id": cid, "claim": claim, "evidence": evidence[:200],
            "expected": expected, "produced": assessment.to_dict(),
            "correct": is_correct, "protocol_ok": is_protocol,
            "signal": signal.to_dict(), "timing_s": round(dt, 2),
        })
        print(flush=True)

    # Metricas
    n = len(DATASET)
    accuracy = correct / n if n > 0 else 0.0
    protocol_rate = protocol_ok / n if n > 0 else 0.0
    avg_time = total_time / n if n > 0 else 0.0

    by_relation = {}
    for case, r in zip(DATASET, results):
        exp = case["expected"]
        if exp not in by_relation:
            by_relation[exp] = {"total": 0, "correct": 0}
        by_relation[exp]["total"] += 1
        if r.get("correct"):
            by_relation[exp]["correct"] += 1

    print("=" * 70, flush=True)
    print("RESULTADOS", flush=True)
    print("=" * 70, flush=True)
    print(f"Modelo: {MODEL} (CPU-only)", flush=True)
    print(f"1. Calidad semantica (accuracy): {correct}/{n} = {accuracy:.1%}", flush=True)
    print(f"2. Adherencia al protocolo:     {protocol_ok}/{n} = {protocol_rate:.1%}", flush=True)
    print(f"3. Decision final:              adaptador produce EvaluationSignal", flush=True)
    print(f"4. Impacto end-to-end:          avg={avg_time:.1f}s total={total_time:.1f}s",
          flush=True)
    print(flush=True)
    print("Por relation:", flush=True)
    for rel, stats in sorted(by_relation.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        print(f"  {rel:12s}: {stats['correct']}/{stats['total']} = {acc:.1%}", flush=True)

    print(flush=True)
    print("Comparacion contra baselines:", flush=True)
    print(f"  BitNet-2B single (EXP-010):   33.3%", flush=True)
    print(f"  BitNet-2B best single (D):    50.0%", flush=True)
    print(f"  BitNet-2B ensemble 4:         41.7%", flush=True)
    print(f"  Granite-3B Q4 (este pilot):   {accuracy:.1%}", flush=True)
    print(f"  Criterio PM-003 (>60%):       {'PASS' if accuracy > 0.60 else 'FAIL'}",
          flush=True)

    report = {
        "pilot": "granite_semantic_reevaluation",
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": MODEL,
        "quantization": "Q4_K_M",
        "parameters": "3.4B",
        "execution": "CPU-only (num_gpu=0)",
        "dataset_size": n,
        "baselines": {
            "bitnet_single": 0.333,
            "bitnet_best_single": 0.50,
            "bitnet_ensemble_4": 0.417,
            "pm003_threshold": 0.60,
        },
        "dimensions": {
            "semantic_accuracy": accuracy,
            "protocol_adherence": protocol_rate,
            "end_to_end_avg_time_s": round(avg_time, 2),
            "end_to_end_total_time_s": round(total_time, 2),
        },
        "by_relation": {
            rel: {
                "total": s["total"],
                "correct": s["correct"],
                "accuracy": s["correct"] / s["total"] if s["total"] > 0 else 0.0,
            }
            for rel, s in by_relation.items()
        },
        "results": results,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(flush=True)
    print(f"Report saved: {OUTPUT}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
