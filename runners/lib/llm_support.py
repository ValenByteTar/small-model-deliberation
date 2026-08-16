"""
LLMSupport — observador paralelo de hipotesis (RES-004, ADR-0031).

DEPRECATED como componente activo del pipeline (PM-003 / EXP-010, 2026-08-16).
BitNet-b1.58-2B-4T no tiene capacidad semantica suficiente. El wiring en
bootstrap.py esta comentado. Este modulo se preserva como experimento
documentado y como infraestructura reusable si un futuro modelo (>=7B,
RES-007) supera el criterio de >60% accuracy en el dataset de EXP-010.

Ver:
  - knowledge/postmortems/PM-003-bitnet-semantic-capacity-insufficient.md
  - knowledge/experiments/EXP-010-bitnet-ensemble-semantic-capacity.md

---

Componente transversal que corre paralelo al pipeline, observa eventos
del TraceSink y produce hipotesis usando un modelo dedicado pequeno en
CPU (BitNet). No bloquea, no decide, no reemplaza capabilities.

Fase 1 (passive): solo loggea hipotesis. No influye el pipeline.
Fase 2 (advisory): requiere ADR separado. No implementada aqui.

Principios invariantes (RES-004 §2):
- Nunca bloquea — corre en thread separado
- Nunca decide — no invoca capabilities, no modifica ExecutionState
- Nunca reemplaza — no sustituye ASSESS, VERIFY ni al Policy Engine
- Simplemente observa y produce hipotesis
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional

from hybrid_rag.kernel.contracts import ModelProvider
from hybrid_rag.kernel.observability import TraceSink
from hybrid_rag.kernel.semantic_ensemble import (
    ConfidenceWeightedMajorityVote,
    SemanticEnsemble,
    SemanticWorker,
    WorkerResult,
    WORKER_PROMPTS,
    WORKER_ROLES,
)
from hybrid_rag.kernel.state import (
    ExecutionState,
    Hypothesis,
    SemanticAssessment,
    TraceEvent,
)

logger = logging.getLogger(__name__)

# Eventos que disparan generacion de hipotesis
_TRIGGER_EVENTS = frozenset({
    "controller.start",
    "capability.end",       # despues de cada capability (retrieve, generate, etc.)
    "controller.end",
})

# Eventos que se observan pero no disparan hipotesis (contexto pasivo)
_OBSERVE_EVENTS = frozenset({
    "policy.decision",
    "capability.start",
    "capability.error",
    "controller.budget",
    "registry.miss",
})


class LLMSupport:
    """
    Observador paralelo + Semantic Capability Provider (RES-004, ADR-0031).

    Dos roles:

    1. Observador pasivo (Fase 1, ADR-0031): se suscribe al TraceSink,
       genera Hypothesis en background, no influye el pipeline.

    2. Semantic Capability Provider (experimento Fase 2 probe): aporta
       interpretacion semantica via semantic_assess(). Produce
       SemanticAssessment (relation, confidence) — no decide, no gatea.
       Un componente deterministico (SemanticAssessmentAdapter) convierte
       SemanticAssessment en EvaluationSignal para el PolicyEngine.

    Frontera (ADR-0020 P16): LLMSupport produce opinion semantica;
    EvidenceQuality / PolicyEngine toman la decision.

    Modos:
    - "off": no hace nada (default seguro)
    - "passive": observa y loggea hipotesis (Fase 1, ADR-0031)
    - "semantic": semantic capability provider (experimento)
    - "advisory": NO implementado (requiere ADR separado)
    """

    def __init__(
        self,
        model_provider: ModelProvider,
        trace_sink: Optional[TraceSink] = None,
        mode: str = "passive",
        max_hypotheses: int = 20,
        max_concurrent: int = 2,
    ) -> None:
        self._provider = model_provider
        self._mode = mode
        self._max_hypotheses = max_hypotheses
        self._max_concurrent = max_concurrent

        # Hipotesis producidas (para medicion offline de precision/recall)
        self.hypotheses: deque[Hypothesis] = deque(maxlen=max_hypotheses)

        # Cola de eventos + pool de workers
        self._event_queue: deque[tuple[TraceEvent, Optional[ExecutionState]]] = deque()
        self._lock = threading.Lock()
        self._workers: List[threading.Thread] = []
        self._running = False

        # Stats
        self._stats = {
            "events_observed": 0,
            "events_triggered": 0,
            "hypotheses_generated": 0,
            "hypotheses_failed": 0,
        }

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def stats(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._stats)

    def start(self) -> None:
        """Inicia los worker threads (no bloquea)."""
        if self._mode == "off" or self._running:
            return
        self._running = True
        for i in range(self._max_concurrent):
            t = threading.Thread(target=self._loop, daemon=True, name=f"llm_support_{i}")
            t.start()
            self._workers.append(t)
        logger.info("LLMSupport iniciado (mode=%s, workers=%d)", self._mode, self._max_concurrent)

    def stop(self) -> None:
        """Detiene los worker threads."""
        self._running = False
        for t in self._workers:
            t.join(timeout=3.0)
        self._workers = []

    def emit(self, event: TraceEvent, state: Optional[ExecutionState] = None) -> None:
        """
        Recibe evento del FanOutTraceSink.

        No bloquea: encola el evento y retorna inmediatamente.
        El worker thread procesa la cola en background.
        """
        if self._mode == "off":
            return
        with self._lock:
            self._stats["events_observed"] += 1
            if event.kind in _TRIGGER_EVENTS:
                self._stats["events_triggered"] += 1
                self._event_queue.append((event, state))

    def get_hypotheses(self) -> List[Hypothesis]:
        """Retorna hipotesis producidas (para medicion offline)."""
        with self._lock:
            return list(self.hypotheses)

    def clear(self) -> None:
        """Limpia hipotesis y stats (entre runs)."""
        with self._lock:
            self.hypotheses.clear()
            self._stats = {
                "events_observed": 0,
                "events_triggered": 0,
                "hypotheses_generated": 0,
                "hypotheses_failed": 0,
            }

    def _loop(self) -> None:
        """Worker thread: procesa eventos de la cola."""
        while self._running:
            try:
                with self._lock:
                    if not self._event_queue:
                        item = None
                    else:
                        item = self._event_queue.popleft()
                if item is None:
                    time.sleep(0.05)
                    continue
                event, state = item
                self._process_event(event, state)
            except Exception as exc:
                logger.debug("LLMSupport worker error: %s", exc)

    def _process_event(self, event: TraceEvent, state: Optional[ExecutionState]) -> None:
        """Genera una hipotesis para un evento trigger."""
        try:
            prompt = self._build_prompt(event, state)
            if not prompt:
                return
            raw = self._provider.generate(prompt, timeout=60.0)
            hypothesis = self._parse_hypothesis(raw, event, state)
            if hypothesis:
                with self._lock:
                    self.hypotheses.append(hypothesis)
                    self._stats["hypotheses_generated"] += 1
                logger.info(
                    "LLMSupport hypothesis [stage=%s]: %s (conf=%.2f)",
                    hypothesis.stage,
                    hypothesis.suggestion,
                    hypothesis.confidence,
                )
            else:
                with self._lock:
                    self._stats["hypotheses_failed"] += 1
        except Exception as exc:
            with self._lock:
                self._stats["hypotheses_failed"] += 1
            logger.debug("LLMSupport generate error: %s", exc)

    def _build_prompt(self, event: TraceEvent, state: Optional[ExecutionState]) -> str:
        """
        Construye un prompt few-shot completion para el modelo observador.

        BitNet-b1.58-2B-4T es un modelo base que no sigue instrucciones
        zero-shot. Usa few-shot completion con razonamiento explicito en
        los ejemplos para que el modelo complete el patron correctamente.
        """
        if state is None:
            return ""

        # Extraer senales del estado actual
        n_results = len(state.results or [])
        top_score = 0.0
        if state.results:
            for key in ("final_score", "rerank_score", "hybrid_score", "score"):
                val = state.results[0].get(key)
                if val is not None:
                    top_score = float(val)
                    break

        n_signals = len(state.signals)
        has_answer = bool(state.answer)
        iteration = state.iteration

        # Few-shot con razonamiento explicito (umbral 0.40)
        prefix = (
            "Classify RAG retrieval quality as GOOD or RETRY based on score threshold 0.40.\n"
            "\n"
            "10 docs, score 0.95 -> GOOD (score above 0.40)\n"
            "8 docs, score 0.88 -> GOOD (score above 0.40)\n"
            "12 docs, score 0.72 -> GOOD (score above 0.40)\n"
            "6 docs, score 0.50 -> GOOD (score above 0.40)\n"
            "3 docs, score 0.45 -> GOOD (score above 0.40)\n"
            "2 docs, score 0.20 -> RETRY (score below 0.40)\n"
            "1 doc, score 0.15 -> RETRY (score below 0.40)\n"
            "5 docs, score 0.10 -> RETRY (score below 0.40)\n"
            "4 docs, score 0.25 -> RETRY (score below 0.40)\n"
            "\n"
        )

        # Linea a completar por el modelo
        current = f"{n_results} docs, score {top_score:.2f} ->"
        return prefix + current

    def _parse_hypothesis(
        self, raw: str, event: TraceEvent, state: Optional[ExecutionState]
    ) -> Optional[Hypothesis]:
        """
        Parsea la respuesta few-shot completion a una Hypothesis.

        Formato esperado: "GOOD (score above 0.40)" o "RETRY (score below 0.40)"
        """
        if not raw:
            return None

        text = raw.strip()
        if not text:
            return None

        # Extraer primera palabra (GOOD o RETRY)
        first_word = text.split()[0] if text.split() else ""
        first_word = first_word.strip(".,;:!?").upper()

        if "GOOD" in first_word:
            suggestion = "GOOD_EVIDENCE"
            confidence = 0.7
        elif "RETRY" in first_word:
            suggestion = "RETRY_RETRIEVAL"
            confidence = 0.7
        else:
            suggestion = "OTHER"
            confidence = 0.0

        # Extraer razonamiento entre parentesis si existe
        reasoning = text[:200]
        if "(" in text:
            start = text.find("(")
            end = text.find(")", start)
            if end > start:
                reasoning = text[start + 1 : end]

        return Hypothesis(
            suggestion=suggestion,
            confidence=confidence,
            reasoning=reasoning,
            stage=event.kind,
            run_id=state.run_id if state else "",
            model=getattr(self._provider, "model", "unknown"),
        )

    # ==================== Semantic Capability Provider ====================

    def semantic_assess(
        self,
        claim: str,
        evidence_text: str,
        *,
        evidence_id: str = "",
        run_id: str = "",
        timeout: float = 60.0,
    ) -> Optional[SemanticAssessment]:
        """
        Evalua semantically la relacion claim-evidence.

        LLMSupport actua como Semantic Capability Provider: produce una
        representacion semantica estructurada (SemanticAssessment) que
        un componente deterministico puede convertir en EvaluationSignal.

        No decide, no gatea, no ejecuta. Solo aporta interpretacion semantica.

        Args:
            claim: texto del claim a evaluar
            evidence_text: texto de la evidencia recuperada
            evidence_id: identificador del evidence item
            run_id: run_id de la ejecucion actual

        Returns:
            SemanticAssessment o None si falla
        """
        if self._mode == "off":
            return None
        if not claim or not evidence_text:
            return None

        prompt = self._build_semantic_prompt(claim, evidence_text)
        if not prompt:
            return None

        try:
            raw = self._provider.generate(prompt, timeout=timeout)
            assessment = self._parse_semantic_assessment(
                raw, claim, evidence_text, evidence_id, run_id
            )
            if assessment:
                with self._lock:
                    self._stats["hypotheses_generated"] += 1
                logger.info(
                    "LLMSupport semantic_assess [relation=%s conf=%.2f]: %s",
                    assessment.relation,
                    assessment.confidence,
                    assessment.reasoning[:80],
                )
            else:
                with self._lock:
                    self._stats["hypotheses_failed"] += 1
            return assessment
        except Exception as exc:
            with self._lock:
                self._stats["hypotheses_failed"] += 1
            logger.debug("LLMSupport semantic_assess error: %s", exc)
            return None

    def _build_semantic_prompt(self, claim: str, evidence_text: str) -> str:
        """
        Construye un prompt few-shot para evaluacion semantica claim-evidence.

        La tarea es: dado un claim y un texto de evidencia, clasificar la
        relacion como SUPPORTS, CONTRADICTS, UNRELATED o PARTIAL.

        Esta tarea es mas natural para un modelo de lenguaje que la
        clasificacion abstracta GOOD/RETRY basada en scores, porque:
        - Es una tarea de comprension de texto (no meta-cognitiva)
        - Esta grounded en texto concreto (no en numeros abstractos)
        - El output es una relacion semantica (no una decision del pipeline)
        """
        # Truncar evidence para mantener prompt manejable
        ev_preview = evidence_text[:600].strip()
        claim_preview = claim[:300].strip()

        prefix = (
            "Task: Classify the relationship between a CLAIM and EVIDENCE text.\n"
            "Relations: SUPPORTS, CONTRADICTS, UNRELATED, PARTIAL\n"
            "\n"
            "CLAIM: The NIST CSF has five core functions\n"
            "EVIDENCE: The Framework Core is a set of cybersecurity activities, outcomes, and informative references that are common across critical infrastructure sectors. The Core consists of five concurrent and continuous Functions: Identify, Detect, Protect, Respond, and Recover.\n"
            "RELATION: SUPPORTS\n"
            "\n"
            "CLAIM: Python 4.0 was released in 2023\n"
            "EVIDENCE: Python 3.12 was released on October 2, 2023. It includes improvements to error messages and performance optimizations.\n"
            "RELATION: CONTRADICTS\n"
            "\n"
            "CLAIM: The sky is blue\n"
            "EVIDENCE: The NIST Cybersecurity Framework provides guidance for organizations to manage cybersecurity risk.\n"
            "RELATION: UNRELATED\n"
            "\n"
            "CLAIM: ISO 27001 requires a risk assessment\n"
            "EVIDENCE: The standard states that the organization shall define and apply a risk assessment process, but does not specify a particular methodology.\n"
            "RELATION: PARTIAL\n"
            "\n"
        )

        current = (
            f"CLAIM: {claim_preview}\n"
            f"EVIDENCE: {ev_preview}\n"
            f"RELATION:"
        )
        return prefix + current

    def _parse_semantic_assessment(
        self,
        raw: str,
        claim: str,
        evidence_text: str,
        evidence_id: str,
        run_id: str,
    ) -> Optional[SemanticAssessment]:
        """
        Parsea la respuesta del modelo a SemanticAssessment.

        Formato esperado: "SUPPORTS" o "CONTRADICTS" o "UNRELATED" o "PARTIAL"
        posiblemente seguido de texto explicativo.
        """
        if not raw:
            return None

        text = raw.strip()
        if not text:
            return None

        # Extraer primera palabra (la relacion)
        words = text.split()
        if not words:
            return None
        first_word = words[0].strip(".,;:!?").upper()

        # Mapear a relations validas
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
            # Intentar buscar cualquier relation en el texto
            text_upper = text.upper()
            for rel in ("SUPPORTS", "CONTRADICTS", "UNRELATED", "PARTIAL"):
                if rel in text_upper:
                    relation = rel
                    break
            if not relation:
                relation = "UNRELATED"  # fallback seguro

        # Confidence: basada en que tan limpio fue el parseo
        confidence = 0.5 if first_word.startswith(relation[:6]) else 0.3

        # Reasoning: el resto del texto despues de la primera palabra
        reasoning = " ".join(words[1:])[:200] if len(words) > 1 else ""

        return SemanticAssessment(
            relation=relation,
            confidence=confidence,
            reasoning=reasoning,
            claim=claim[:300],
            evidence_id=evidence_id,
            evidence_preview=evidence_text[:200],
            run_id=run_id,
            model=getattr(self._provider, "model", "unknown"),
        )

    # ==================== Semantic Ensemble ====================

    def semantic_assess_ensemble(
        self,
        claim: str,
        evidence_text: str,
        *,
        evidence_id: str = "",
        run_id: str = "",
        timeout: float = 60.0,
        ensemble: Optional[SemanticEnsemble] = None,
    ) -> tuple[Optional[SemanticAssessment], List[WorkerResult], Dict[str, Any]]:
        """
        Evalua semantically usando un ensemble de N workers independientes.

        Cada worker es una instancia de BitNet con un prompt diferente.
        Los workers ejecutan en paralelo. El aggregator combina
        deterministicamente (no usa LLM para agregar).

        Frontera (ADR-0020 P16): el ensemble produce SemanticAssessment;
        SemanticAssessmentAdapter -> EvaluationSignal -> PolicyEngine decide.

        Args:
            ensemble: SemanticEnsemble pre-construido. Si es None, no se
                      puede ejecutar (retorna None).

        Returns:
            (final_assessment, worker_results, aggregation_metadata)
            o (None, [], {}) si no hay ensemble.
        """
        if self._mode == "off" or ensemble is None:
            return None, [], {}

        if not claim or not evidence_text:
            return None, [], {}

        try:
            assessment, results, meta = ensemble.assess(
                claim=claim,
                evidence_text=evidence_text,
                evidence_id=evidence_id,
                run_id=run_id,
                timeout=timeout,
            )
            with self._lock:
                self._stats["hypotheses_generated"] += 1
            logger.info(
                "LLMSupport ensemble [%d workers] relation=%s conf=%.2f agreement=%s",
                ensemble.size,
                assessment.relation,
                assessment.confidence,
                meta.get("agreement_fraction", "?"),
            )
            return assessment, results, meta
        except Exception as exc:
            with self._lock:
                self._stats["hypotheses_failed"] += 1
            logger.debug("LLMSupport ensemble error: %s", exc)
            return None, [], {"error": str(exc)}

