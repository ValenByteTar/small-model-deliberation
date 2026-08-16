"""
SemanticAssessmentAdapter — adaptador deterministico (ADR-0020 P16, ADR-0019).

DEPRECATED como componente activo del pipeline (PM-003 / EXP-010, 2026-08-16).
El adaptador funciona correctamente pero el componente que produce
SemanticAssessment (LLMSupport con BitNet) no tiene capacidad suficiente.
Este modulo se preserva como infraestructura reusable si un futuro modelo
supera el criterio de >60% accuracy en el dataset de EXP-010.

Ver:
  - knowledge/postmortems/PM-003-bitnet-semantic-capacity-insufficient.md
  - knowledge/experiments/EXP-010-bitnet-ensemble-semantic-capacity.md

---

Convierte SemanticAssessment (opinion semantica de LLMSupport) en
EvaluationSignal (senal que el PolicyEngine puede consumir).

Frontera arquitectonica (ADR-0020 P16):
- LLMSupport produce la opinion semantica (SUPPORTS / CONTRADICTS / ...)
- Este adaptador la traduce a EvaluationSignal de forma deterministica
- EvidenceQuality / PolicyEngine toman la decision final

No es un Evaluator (no implementa el contrato Evaluator.evaluate(state)).
Es un adaptador puro: SemanticAssessment -> EvaluationSignal.

Uso:
    adapter = SemanticAssessmentAdapter()
    signal = adapter.adapt(assessment)
    # signal se agrega a state.signals para que el PolicyEngine la consuma
"""

from __future__ import annotations

from hybrid_rag.kernel.state import EvaluationSignal, SemanticAssessment


class SemanticAssessmentAdapter:
    """
    Adaptador deterministico: SemanticAssessment -> EvaluationSignal.

    Mapeo de relations:
    - SUPPORTS   -> passed=True,  score=confidence,  reason="semantic:supports"
    - PARTIAL    -> passed=None,  score=confidence*0.5, reason="semantic:partial"
    - UNRELATED  -> passed=None,  score=0.0,  reason="semantic:unrelated"
    - CONTRADICTS -> passed=False, score=0.0,  reason="semantic:contradicts"

    El passed=None (PARTIAL, UNRELATED) significa "no gatea": la senal
    informa pero no fuerza pass/fail. El PolicyEngine decide que hacer.
    """

    name = "semantic_assess"

    def adapt(self, assessment: SemanticAssessment) -> EvaluationSignal:
        relation = assessment.relation.upper()
        conf = max(0.0, min(1.0, assessment.confidence))

        if relation == "SUPPORTS":
            passed = True
            score = conf
            reason = f"semantic:supports (conf={conf:.2f})"
        elif relation == "CONTRADICTS":
            passed = False
            score = 0.0
            reason = f"semantic:contradicts (conf={conf:.2f})"
        elif relation == "PARTIAL":
            passed = None
            score = conf * 0.5
            reason = f"semantic:partial (conf={conf:.2f})"
        else:  # UNRELATED or unknown
            passed = None
            score = 0.0
            reason = f"semantic:unrelated (conf={conf:.2f})"

        return EvaluationSignal(
            name=self.name,
            score=score,
            passed=passed,
            reason=reason,
            metadata={
                "relation": relation,
                "confidence": conf,
                "reasoning": assessment.reasoning[:200],
                "evidence_id": assessment.evidence_id,
                "claim": assessment.claim[:200],
                "model": assessment.model,
            },
            source="online",
        )

    def adapt_batch(
        self, assessments: list[SemanticAssessment]
    ) -> list[EvaluationSignal]:
        """Adapta una lista de SemanticAssessment a EvaluationSignals."""
        return [self.adapt(a) for a in assessments if a is not None]
