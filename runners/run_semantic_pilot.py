"""
DEPRECATED — Experimento documentado, no reejecutar sin leer PM-003.

Este script es parte del experimento LLMSupport con BitNet-b1.58-2B-4T
como Semantic Capability Provider. El experimento fallo: BitNet-2B no
tiene capacidad semantica suficiente (33% accuracy, 100% protocolo).

Resultado: ADR-0031 deprecado, LLMSupport desacoplado del pipeline.

Ver:
  - knowledge/postmortems/PM-003-bitnet-semantic-capacity-insufficient.md
  - knowledge/experiments/EXP-010-bitnet-ensemble-semantic-capacity.md

Para reevaluar con un modelo diferente (>=7B, RES-007), este script
sirve como benchmark: el dataset de 12 pares claim-evidence esta
congelado. Criterio de aprobacion: >60% accuracy.

---

Semantic Capability Provider Pilot — mide 4 dimensiones del LLMSupport
como Semantic Capability Provider (experimento, ADR-0031 Fase 2 probe).

Dimensiones:
1. Calidad semantica del LLM: BitNet produce la relation correcta?
2. Adherencia al protocolo: el output sigue el formato esperado?
3. Calidad de la decision final: SemanticAssessmentAdapter -> EvaluationSignal
4. Impacto end-to-end: timing, latencia

No mide GOOD/RETRY. Mide SUPPORTS/CONTRADICTS/UNRELATED/PARTIAL
sobre pares claim-evidence con ground truth conocido.

Uso:
    python scripts/run_semantic_pilot.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hybrid_rag.kernel.llm_support import LLMSupport
from hybrid_rag.kernel.state import SemanticAssessment
from hybrid_rag.evaluation.semantic_adapter import SemanticAssessmentAdapter
from hybrid_rag.providers.bitnet_provider import BitNetModelProvider

OUTPUT = ROOT / "tests" / "eval" / "canonical" / "reports" / "semantic_pilot.json"

# Dataset de pares claim-evidence con ground truth.
# Cada caso tiene un claim, un evidence_text, y la relation esperada.
# Los casos son deliberadamente claros para medir capacidad semantica basica.
DATASET = [
    # SUPPORTS — la evidencia respalda el claim directamente
    {
        "id": "s-001",
        "claim": "The NIST CSF has five core functions",
        "evidence": (
            "The Framework Core consists of five concurrent and continuous "
            "Functions: Identify, Detect, Protect, Respond, and Recover."
        ),
        "expected": "SUPPORTS",
    },
    {
        "id": "s-002",
        "claim": "ISO 27001 is an information security standard",
        "evidence": (
            "ISO/IEC 27001 is an international standard for information "
            "security management systems (ISMS)."
        ),
        "expected": "SUPPORTS",
    },
    {
        "id": "s-003",
        "claim": "SOAR tools help automate security operations",
        "evidence": (
            "Security Orchestration, Automation and Response (SOAR) platforms "
            "enable security teams to automate incident response workflows "
            "and integrate disparate security tools."
        ),
        "expected": "SUPPORTS",
    },
    # CONTRADICTS — la evidencia contradice el claim
    {
        "id": "c-001",
        "claim": "Python 4.0 was released in 2023",
        "evidence": (
            "Python 3.12 was released on October 2, 2023. "
            "There is no Python 4.0 release."
        ),
        "expected": "CONTRADICTS",
    },
    {
        "id": "c-002",
        "claim": "The NIST CSF has three core functions",
        "evidence": (
            "The Framework Core consists of five concurrent and continuous "
            "Functions: Identify, Detect, Protect, Respond, and Recover."
        ),
        "expected": "CONTRADICTS",
    },
    {
        "id": "c-003",
        "claim": "AES-256 is considered weak encryption",
        "evidence": (
            "AES-256 is widely regarded as one of the strongest encryption "
            "algorithms available and is approved by NSA for top secret data."
        ),
        "expected": "CONTRADICTS",
    },
    # UNRELATED — la evidencia no tiene relacion con el claim
    {
        "id": "u-001",
        "claim": "The sky is blue",
        "evidence": (
            "The NIST Cybersecurity Framework provides guidance for "
            "organizations to manage cybersecurity risk."
        ),
        "expected": "UNRELATED",
    },
    {
        "id": "u-002",
        "claim": "Water boils at 100 degrees Celsius",
        "evidence": (
            "ISO 27001 requires organizations to conduct regular risk "
            "assessments and implement appropriate security controls."
        ),
        "expected": "UNRELATED",
    },
    {
        "id": "u-003",
        "claim": "Paris is the capital of France",
        "evidence": (
            "The CIA triad consists of Confidentiality, Integrity, and "
            "Availability, which are the core principles of information security."
        ),
        "expected": "UNRELATED",
    },
    # PARTIAL — la evidencia respalda parcialmente el claim
    {
        "id": "p-001",
        "claim": "ISO 27001 requires a specific risk assessment methodology",
        "evidence": (
            "The standard states that the organization shall define and apply "
            "a risk assessment process, but does not specify a particular "
            "methodology."
        ),
        "expected": "PARTIAL",
    },
    {
        "id": "p-002",
        "claim": "The NIST CSF includes 10 functions for cybersecurity",
        "evidence": (
            "The Framework Core includes five Functions. Some organizations "
            "extend the framework with additional custom functions."
        ),
        "expected": "PARTIAL",
    },
    {
        "id": "p-003",
        "claim": "Zero trust means no network security controls are needed",
        "evidence": (
            "Zero trust is a security model that assumes no implicit trust "
            "and requires verification for every access request. It does not "
            "eliminate security controls; it changes how they are applied."
        ),
        "expected": "CONTRADICTS",
    },
]


def main() -> int:
    print("=" * 70)
    print("Semantic Capability Provider Pilot")
    print("LLMSupport as Semantic Capability Provider (ADR-0031 Fase 2 probe)")
    print("=" * 70)
    print(f"Dataset: {len(DATASET)} claim-evidence pairs")
    print(f"  SUPPORTS: {sum(1 for d in DATASET if d['expected'] == 'SUPPORTS')}")
    print(f"  CONTRADICTS: {sum(1 for d in DATASET if d['expected'] == 'CONTRADICTS')}")
    print(f"  UNRELATED: {sum(1 for d in DATASET if d['expected'] == 'UNRELATED')}")
    print(f"  PARTIAL: {sum(1 for d in DATASET if d['expected'] == 'PARTIAL')}")
    print()

    # Inicializar BitNet provider
    print("Iniciando BitNet provider...")
    provider = BitNetModelProvider(
        model_path="models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf",
        server_path="build/bin/Release/llama-server.exe",
        bitnet_root="C:/Users/Valen/Desktop/Proyectos/BitNet",
        port=8081,
        threads=8,
        ctx_size=2048,
    )
    if not provider.ensure_running(timeout=30.0):
        print("ERROR: No se pudo iniciar llama-server")
        return 1

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

        print(f"[{i}/{len(DATASET)}] {cid}: claim='{claim[:50]}...' expected={expected}")

        t0 = time.time()
        assessment = llm.semantic_assess(
            claim=claim,
            evidence_text=evidence,
            evidence_id=cid,
            run_id="semantic-pilot",
            timeout=60.0,
        )
        dt = time.time() - t0
        total_time += dt

        if assessment is None:
            print(f"  FAIL: no assessment produced ({dt:.1f}s)")
            results.append({
                "id": cid,
                "claim": claim,
                "evidence": evidence[:200],
                "expected": expected,
                "produced": None,
                "correct": False,
                "protocol_ok": False,
                "timing_s": round(dt, 2),
            })
            continue

        # Dimension 1: calidad semantica (relation correcta?)
        is_correct = assessment.relation == expected
        if is_correct:
            correct += 1

        # Dimension 2: adherencia al protocolo (relation valida + confidence en rango?)
        from hybrid_rag.kernel.state import SEMANTIC_RELATIONS
        is_protocol = (
            assessment.relation in SEMANTIC_RELATIONS
            and 0.0 <= assessment.confidence <= 1.0
        )
        if is_protocol:
            protocol_ok += 1

        # Dimension 3: decision final (adaptador -> EvaluationSignal)
        signal = adapter.adapt(assessment)

        print(
            f"  relation={assessment.relation} conf={assessment.confidence:.2f} "
            f"correct={is_correct} protocol={is_protocol} "
            f"signal: passed={signal.passed} score={signal.score:.2f} "
            f"({dt:.1f}s)"
        )
        if assessment.reasoning:
            print(f"  reasoning: {assessment.reasoning[:80]}")

        results.append({
            "id": cid,
            "claim": claim,
            "evidence": evidence[:200],
            "expected": expected,
            "produced": assessment.to_dict(),
            "correct": is_correct,
            "protocol_ok": is_protocol,
            "signal": signal.to_dict(),
            "timing_s": round(dt, 2),
        })
        print()

    # Shutdown
    provider.shutdown()

    # Calcular metricas
    n = len(DATASET)
    accuracy = correct / n if n > 0 else 0.0
    protocol_rate = protocol_ok / n if n > 0 else 0.0
    avg_time = total_time / n if n > 0 else 0.0

    # Por-relation breakdown
    by_relation = {}
    for case, r in zip(DATASET, results):
        exp = case["expected"]
        if exp not in by_relation:
            by_relation[exp] = {"total": 0, "correct": 0}
        by_relation[exp]["total"] += 1
        if r.get("correct"):
            by_relation[exp]["correct"] += 1

    print("=" * 70)
    print("RESULTADOS")
    print("=" * 70)
    print(f"1. Calidad semantica (accuracy): {correct}/{n} = {accuracy:.1%}")
    print(f"2. Adherencia al protocolo:     {protocol_ok}/{n} = {protocol_rate:.1%}")
    print(f"3. Decision final:              adaptador produce EvaluationSignal")
    print(f"4. Impacto end-to-end:          avg={avg_time:.1f}s total={total_time:.1f}s")
    print()
    print("Por relation:")
    for rel, stats in sorted(by_relation.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        print(f"  {rel:12s}: {stats['correct']}/{stats['total']} = {acc:.1%}")

    # Guardar reporte
    report = {
        "pilot": "semantic_capability_provider",
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset_size": n,
        "dimensions": {
            "semantic_accuracy": accuracy,
            "protocol_adherence": protocol_rate,
            "decision_quality": "adapted_to_evaluation_signal",
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

    print()
    print(f"Report saved: {OUTPUT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
