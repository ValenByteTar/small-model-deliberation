"""
SemanticEnsemble — ensemble de 4 evaluadores semanticos independientes.

DEPRECATED como componente activo del pipeline (PM-003 / EXP-010, 2026-08-16).
El ensemble de 4 BitNet-b1.58-2B-4T no supero al mejor worker individual
(41.7% vs 50%) y mostro alta correlacion de errores (Jaccard 0.40-0.64).
Este modulo se preserva como experimento documentado y como infraestructura
reusable si un futuro modelo supera el criterio de >60% accuracy.

Ver:
  - knowledge/postmortems/PM-003-bitnet-semantic-capacity-insufficient.md
  - knowledge/experiments/EXP-010-bitnet-ensemble-semantic-capacity.md

---

Cada worker es una instancia de BitNet-b1.58-2B-4T con 1 thread interno,
ejecutando un prompt deliberadamente diferente para reducir correlacion
de errores.

Frontera arquitectonica (ADR-0020 P16):
- Los workers producen SemanticAssessment (opinion semantica)
- EnsembleAggregator combina deterministicamente (no usa LLM)
- El resultado final es SemanticAssessment, entregado a
  SemanticAssessmentAdapter -> EvaluationSignal -> PolicyEngine

No modifica Controller, PolicyEngine, ni Registry.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from hybrid_rag.kernel.state import SEMANTIC_RELATIONS, SemanticAssessment

logger = logging.getLogger(__name__)


# ==================== Prompt Strategies ====================


def _build_entailment_prompt(claim: str, evidence: str) -> str:
    """Worker A — Entailment Analyst: enfocado en logical entailment."""
    ev = evidence[:600].strip()
    cl = claim[:300].strip()
    return (
        "Task: Determine if the EVIDENCE logically entails or supports the CLAIM.\n"
        "Focus exclusively on logical entailment and semantic support.\n"
        "Relations: SUPPORTS, CONTRADICTS, UNRELATED, PARTIAL\n"
        "\n"
        "CLAIM: The NIST CSF has five core functions\n"
        "EVIDENCE: The Framework Core consists of five Functions: Identify, Detect, Protect, Respond, and Recover.\n"
        "RELATION: SUPPORTS\n"
        "\n"
        "CLAIM: Python 4.0 was released in 2023\n"
        "EVIDENCE: Python 3.12 was released on October 2, 2023. There is no Python 4.0.\n"
        "RELATION: CONTRADICTS\n"
        "\n"
        "CLAIM: The sky is blue\n"
        "EVIDENCE: The NIST Cybersecurity Framework provides guidance for managing cybersecurity risk.\n"
        "RELATION: UNRELATED\n"
        "\n"
        "CLAIM: ISO 27001 requires a specific risk assessment methodology\n"
        "EVIDENCE: The standard requires a risk assessment process but does not specify a particular methodology.\n"
        "RELATION: PARTIAL\n"
        "\n"
        f"CLAIM: {cl}\n"
        f"EVIDENCE: {ev}\n"
        f"RELATION:"
    )


def _build_skeptical_prompt(claim: str, evidence: str) -> str:
    """Worker B — Skeptical Evidence Analyst: postura escéptica."""
    ev = evidence[:600].strip()
    cl = claim[:300].strip()
    return (
        "Task: Skeptically evaluate whether the EVIDENCE truly proves the CLAIM.\n"
        "Ask: Does the evidence actually demonstrate the claim? Is there a logical gap?\n"
        "Does the evidence merely share vocabulary with the claim? Is the claim stronger than the evidence?\n"
        "Be conservative: only say SUPPORTS if the evidence clearly proves the claim.\n"
        "Relations: SUPPORTS, CONTRADICTS, UNRELATED, PARTIAL\n"
        "\n"
        "CLAIM: The NIST CSF has five core functions\n"
        "EVIDENCE: The Framework Core consists of five Functions: Identify, Detect, Protect, Respond, and Recover.\n"
        "RELATION: SUPPORTS\n"
        "\n"
        "CLAIM: ISO 27001 eliminates all cybersecurity risks\n"
        "EVIDENCE: ISO 27001 helps organizations manage information security risks through a systematic approach.\n"
        "RELATION: CONTRADICTS\n"
        "\n"
        "CLAIM: The sky is blue\n"
        "EVIDENCE: ISO 27001 requires regular risk assessments.\n"
        "RELATION: UNRELATED\n"
        "\n"
        "CLAIM: AES-256 is the only approved encryption standard\n"
        "EVIDENCE: AES-256 is approved by NSA for top secret data. Other algorithms like ChaCha20 are also approved.\n"
        "RELATION: PARTIAL\n"
        "\n"
        f"CLAIM: {cl}\n"
        f"EVIDENCE: {ev}\n"
        f"RELATION:"
    )


def _build_contradiction_prompt(claim: str, evidence: str) -> str:
    """Worker C — Contradiction/Alternative Analyst: busca contradicciones activamente."""
    ev = evidence[:600].strip()
    cl = claim[:300].strip()
    return (
        "Task: Actively search for contradictions between the CLAIM and EVIDENCE.\n"
        "Look for: incompatible facts, conflicting numbers, evidence that undermines the claim.\n"
        "Also check: is the evidence relevant but does not support the claim?\n"
        "Distinguish carefully between CONTRADICTS, PARTIAL, and UNRELATED.\n"
        "Relations: SUPPORTS, CONTRADICTS, UNRELATED, PARTIAL\n"
        "\n"
        "CLAIM: The NIST CSF has three core functions\n"
        "EVIDENCE: The Framework Core consists of five Functions: Identify, Detect, Protect, Respond, and Recover.\n"
        "RELATION: CONTRADICTS\n"
        "\n"
        "CLAIM: Python 4.0 was released in 2023\n"
        "EVIDENCE: Python 3.12 was released on October 2, 2023. There is no Python 4.0 release.\n"
        "RELATION: CONTRADICTS\n"
        "\n"
        "CLAIM: The sky is blue\n"
        "EVIDENCE: The CIA triad consists of Confidentiality, Integrity, and Availability.\n"
        "RELATION: UNRELATED\n"
        "\n"
        "CLAIM: ISO 27001 requires a specific risk assessment methodology\n"
        "EVIDENCE: The standard requires a risk assessment process but does not mandate a specific methodology.\n"
        "RELATION: PARTIAL\n"
        "\n"
        f"CLAIM: {cl}\n"
        f"EVIDENCE: {ev}\n"
        f"RELATION:"
    )


def _build_neutral_prompt(claim: str, evidence: str) -> str:
    """Worker D — Independent Neutral Analyst: evaluacion equilibrada."""
    ev = evidence[:600].strip()
    cl = claim[:300].strip()
    return (
        "Task: Classify the relationship between a CLAIM and EVIDENCE text objectively.\n"
        "Evaluate independently and balanced. Consider all four relations equally.\n"
        "Relations: SUPPORTS, CONTRADICTS, UNRELATED, PARTIAL\n"
        "\n"
        "CLAIM: The NIST CSF has five core functions\n"
        "EVIDENCE: The Framework Core consists of five concurrent and continuous Functions.\n"
        "RELATION: SUPPORTS\n"
        "\n"
        "CLAIM: Python 4.0 was released in 2023\n"
        "EVIDENCE: Python 3.12 was released on October 2, 2023. No Python 4.0 exists.\n"
        "RELATION: CONTRADICTS\n"
        "\n"
        "CLAIM: Water boils at 100 degrees Celsius\n"
        "EVIDENCE: ISO 27001 requires organizations to conduct regular risk assessments.\n"
        "RELATION: UNRELATED\n"
        "\n"
        "CLAIM: The NIST CSF includes 10 functions\n"
        "EVIDENCE: The Framework Core includes five Functions. Some organizations extend it.\n"
        "RELATION: PARTIAL\n"
        "\n"
        f"CLAIM: {cl}\n"
        f"EVIDENCE: {ev}\n"
        f"RELATION:"
    )


# Prompt registry
WORKER_PROMPTS = {
    "entailment": _build_entailment_prompt,
    "skeptical": _build_skeptical_prompt,
    "contradiction": _build_contradiction_prompt,
    "neutral": _build_neutral_prompt,
}

WORKER_ROLES = list(WORKER_PROMPTS.keys())


# ==================== SemanticWorker ====================


@dataclass
class WorkerResult:
    """Resultado de un worker individual."""
    worker_id: str
    role: str
    assessment: Optional[SemanticAssessment]
    latency_s: float
    error: Optional[str] = None

    @property
    def relation(self) -> str:
        return self.assessment.relation if self.assessment else "ERROR"

    @property
    def confidence(self) -> float:
        return self.assessment.confidence if self.assessment else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "role": self.role,
            "relation": self.relation,
            "confidence": self.confidence,
            "latency_s": round(self.latency_s, 2),
            "error": self.error,
            "assessment": self.assessment.to_dict() if self.assessment else None,
        }


class SemanticWorker:
    """
    Un evaluador semántico independiente.

    Wraps a ModelProvider with a specific prompt strategy.
    Each worker is independent — no shared state, no communication between workers.
    """

    def __init__(
        self,
        worker_id: str,
        role: str,
        model_provider: Any,  # ModelProvider
        prompt_fn: Any,  # Callable[[str, str], str]
    ) -> None:
        self.worker_id = worker_id
        self.role = role
        self._provider = model_provider
        self._prompt_fn = prompt_fn

    def assess(
        self,
        claim: str,
        evidence_text: str,
        *,
        evidence_id: str = "",
        run_id: str = "",
        timeout: float = 60.0,
    ) -> WorkerResult:
        t0 = time.time()
        try:
            prompt = self._prompt_fn(claim, evidence_text)
            raw = self._provider.generate(
                prompt,
                options={"num_predict": 10, "temperature": 0.0},
                timeout=timeout,
            )
            assessment = _parse_semantic_response(
                raw, claim, evidence_text, evidence_id, run_id,
                model=getattr(self._provider, "model", "unknown"),
                worker_id=self.worker_id,
            )
            dt = time.time() - t0
            return WorkerResult(
                worker_id=self.worker_id,
                role=self.role,
                assessment=assessment,
                latency_s=dt,
            )
        except Exception as exc:
            dt = time.time() - t0
            return WorkerResult(
                worker_id=self.worker_id,
                role=self.role,
                assessment=None,
                latency_s=dt,
                error=str(exc),
            )


# ==================== EnsembleAggregator ====================


class AggregationStrategy(Protocol):
    """Interfaz aislada para estrategias de agregacion."""

    def aggregate(self, results: List[WorkerResult]) -> tuple[str, float, Dict[str, Any]]:
        """
        Args:
            results: resultados de los workers

        Returns:
            (final_relation, final_confidence, metadata)
        """
        ...


class ConfidenceWeightedMajorityVote:
    """
    Agregador deterministico: majority vote ponderado por confidence.

    Estrategia:
    1. Para cada relation, sumar las confidences de los workers que la votaron
    2. La relation con mayor suma de confidence gana
    3. final_confidence = suma de confidence de la relation ganadora / suma total
    4. agreement = numero de workers que votaron la relation ganadora / total
    """

    def aggregate(self, results: List[WorkerResult]) -> tuple[str, float, Dict[str, Any]]:
        valid = [r for r in results if r.assessment is not None and r.relation in SEMANTIC_RELATIONS]

        if not valid:
            return "UNRELATED", 0.0, {"agreement": 0.0, "votes": [], "reason": "no_valid_results"}

        # Sumar confidences por relation
        relation_weights: Dict[str, float] = {}
        relation_votes: Dict[str, int] = {}
        for r in valid:
            rel = r.relation
            relation_weights[rel] = relation_weights.get(rel, 0.0) + r.confidence
            relation_votes[rel] = relation_votes.get(rel, 0) + 1

        # Relation con mayor peso
        final_relation = max(relation_weights, key=relation_weights.get)
        total_weight = sum(relation_weights.values())
        final_confidence = relation_weights[final_relation] / total_weight if total_weight > 0 else 0.0

        # Agreement: fraccion de workers que votaron la relation ganadora
        agreement = relation_votes[final_relation] / len(valid)

        votes = [
            {
                "worker_id": r.worker_id,
                "role": r.role,
                "relation": r.relation,
                "confidence": r.confidence,
            }
            for r in valid
        ]

        metadata = {
            "agreement": agreement,
            "agreement_fraction": f"{relation_votes[final_relation]}/{len(valid)}",
            "votes": votes,
            "relation_weights": {k: round(v, 3) for k, v in relation_weights.items()},
            "strategy": "confidence_weighted_majority_vote",
        }

        return final_relation, final_confidence, metadata


# ==================== SemanticEnsemble ====================


class SemanticEnsemble:
    """
    Ensemble de N evaluadores semánticos independientes.

    Ejecuta los workers en paralelo (ThreadPoolExecutor) y agrega
    los resultados deterministicamente.
    """

    def __init__(
        self,
        workers: List[SemanticWorker],
        strategy: Optional[AggregationStrategy] = None,
    ) -> None:
        self._workers = workers
        self._strategy = strategy or ConfidenceWeightedMajorityVote()

    @property
    def workers(self) -> List[SemanticWorker]:
        return list(self._workers)

    @property
    def size(self) -> int:
        return len(self._workers)

    def assess(
        self,
        claim: str,
        evidence_text: str,
        *,
        evidence_id: str = "",
        run_id: str = "",
        timeout: float = 60.0,
    ) -> tuple[SemanticAssessment, List[WorkerResult], Dict[str, Any]]:
        """
        Ejecuta todos los workers en paralelo y agrega.

        Returns:
            (final_assessment, worker_results, aggregation_metadata)
        """
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=len(self._workers)) as pool:
            futures = {
                pool.submit(
                    w.assess,
                    claim=claim,
                    evidence_text=evidence_text,
                    evidence_id=evidence_id,
                    run_id=run_id,
                    timeout=timeout,
                ): w
                for w in self._workers
            }
            results: List[WorkerResult] = []
            for fut in as_completed(futures):
                results.append(fut.result())

        # Ordenar por worker_id para consistencia
        results.sort(key=lambda r: r.worker_id)

        # Agregar
        final_relation, final_confidence, agg_meta = self._strategy.aggregate(results)

        total_latency = time.time() - t0
        agg_meta["total_latency_s"] = round(total_latency, 2)
        agg_meta["worker_latencies_s"] = {r.worker_id: round(r.latency_s, 2) for r in results}

        final_assessment = SemanticAssessment(
            relation=final_relation,
            confidence=final_confidence,
            reasoning=f"ensemble({len(results)} workers, agreement={agg_meta.get('agreement_fraction', '?')})",
            claim=claim[:300],
            evidence_id=evidence_id,
            evidence_preview=evidence_text[:200],
            run_id=run_id,
            model=f"ensemble-{len(self._workers)}x-bitnet",
        )

        return final_assessment, results, agg_meta


# ==================== Parser ====================


def _parse_semantic_response(
    raw: str,
    claim: str,
    evidence_text: str,
    evidence_id: str,
    run_id: str,
    *,
    model: str = "unknown",
    worker_id: str = "",
) -> Optional[SemanticAssessment]:
    """Parsea la respuesta del modelo a SemanticAssessment."""
    if not raw:
        return None

    text = raw.strip()
    if not text:
        return None

    words = text.split()
    if not words:
        return None
    first_word = words[0].strip(".,;:!?").upper()

    relation = ""
    if "SUPPORT" in first_word:
        relation = "SUPPORTS"
    elif "CONTRADICT" in first_word:
        relation = "CONTRADICTS"
    elif "UNRELATED" in first_word or "IRRELEVANT" in first_word:
        relation = "UNRELATED"
    elif "PARTIAL" in first_word:
        relation = "PARTIAL"
    else:
        text_upper = text.upper()
        for rel in ("SUPPORTS", "CONTRADICTS", "UNRELATED", "PARTIAL"):
            if rel in text_upper:
                relation = rel
                break
        if not relation:
            relation = "UNRELATED"

    confidence = 0.5 if first_word.startswith(relation[:6]) else 0.3
    reasoning = " ".join(words[1:])[:200] if len(words) > 1 else ""
    if worker_id:
        reasoning = f"[{worker_id}] {reasoning}"

    return SemanticAssessment(
        relation=relation,
        confidence=confidence,
        reasoning=reasoning,
        claim=claim[:300],
        evidence_id=evidence_id,
        evidence_preview=evidence_text[:200],
        run_id=run_id,
        model=model,
    )
