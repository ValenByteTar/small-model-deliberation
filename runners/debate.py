"""
Micro-Coliseum: Deliberative Semantic Assessment

Experimento aislado para evaluar si la DELIBERACION entre LLMs mejora
la calidad de SemanticAssessment respecto al ensemble paralelo independiente.

NO modifica Controller, PolicyEngine, Registry, SemanticAssessmentAdapter,
el benchmark, ni el ensemble existente.

Arquitectura:

    SemanticAssessmentRequest
                |
        +-------+-------+-------+
        |       |       |       |
        A       B       C       D     Phase 1: Independent Assessment
        |       |       |       |
        +-------+-------+-------+
                |
        Independent Ensemble         Phase 2: Initial Ensemble (frozen)
                |
        Disagreement Detector        Phase 3: Debate trigger
                |
      Challenge / Debate Round       Phase 4: Workers see others' opinions
                |
          Final Judge                Phase 5: Judge decides
                |
        SemanticAssessment
                |
           [benchmark only]

Modos:
    --mode independent           E0: 4 workers + vote (sin debate)
    --mode debate-on-disagreement E1: debate solo si hay disagreement
    --mode debate-all             E2: debate siempre

Uso:
    python -m experiments.microcoliseum_deliberation.run_microcoliseum \
        --model ibm/granite4.1:3b-q4_K_M \
        --mode debate-on-disagreement

    python -m experiments.microcoliseum_deliberation.run_microcoliseum \
        --model qwen3-4b-rag:latest \
        --mode debate-all \
        --gpu
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import requests

from hybrid_rag.kernel.state import SEMANTIC_RELATIONS
from hybrid_rag.providers.ollama_provider import OllamaModelProvider

BENCHMARK = ROOT / "benchmarks" / "semantic_assessment_v2.json"
OUTPUT_DIR = ROOT / "results" / "raw"

VALID_RELATIONS = sorted(SEMANTIC_RELATIONS)  # ["CONTRADICTS", "PARTIAL", "SUPPORTS", "UNRELATED"]

RELATION_SCHEMA = {
    "type": "object",
    "properties": {
        "relation": {"type": "string", "enum": VALID_RELATIONS},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["relation", "confidence"],
    "additionalProperties": False,
}
ASSESSMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "worker": {"type": "string"},
        "relation": {"type": "string", "enum": VALID_RELATIONS},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["worker", "relation", "confidence"],
    "additionalProperties": False,
}
CHALLENGE_SCHEMA = {
    "type": "object",
    "properties": {
        "worker": {"type": "string"},
        "current_relation": {"type": "string", "enum": VALID_RELATIONS},
        "change_decision": {"type": "string", "enum": ["KEEP", "CHANGE"]},
        "proposed_relation": {"type": "string", "enum": VALID_RELATIONS},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["worker", "current_relation", "change_decision", "proposed_relation", "confidence"],
    "additionalProperties": False,
}


class StructuredOllamaProvider(OllamaModelProvider):
    """Ollama provider with strict JSON output and disabled reasoning."""

    def generate_structured(self, prompt: str, schema: dict, *, timeout: float) -> str:
        options = {"num_gpu": self.num_gpu, **self.default_options}
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "format": schema,
                "options": options,
                "keep_alive": "10m",
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return (response.json().get("response") or "").strip()


class LlamaServerProvider:
    """Provider para llama-server (bitnet.cpp).

    llama-server usa OpenAI-compatible API en /completion.
    No soporta format=json_schema ni think=false; los prompts ya
    incluyen instrucciones JSON explicitas y el parser _extract_json
    maneja la recuperacion.
    """

    name = "llama-server"

    def __init__(
        self,
        model: str = "",
        base_url: str = "http://127.0.0.1:8081",
        num_gpu: int = 0,
        default_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.num_gpu = num_gpu
        self.default_options = default_options or {}

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def generate_structured(self, prompt: str, schema: dict, *, timeout: float) -> str:
        opts = self.default_options
        response = requests.post(
            f"{self.base_url}/completion",
            json={
                "prompt": prompt,
                "stream": False,
                "temperature": opts.get("temperature", 0.0),
                "max_tokens": opts.get("num_predict", 128),
                "repeat_penalty": 1.1,
                "json_schema": schema,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return (response.json().get("content") or "").strip()


# ==================== Worker Prompts ====================

WORKER_SPECS = [
    {
        "id": "A",
        "role": "entailment analyst",
        "objective": (
            "Determine if the EVIDENCE actually supports the CLAIM.\n"
            "Pay special attention to:\n"
            "- entailment\n"
            "- direct support\n"
            "- partial support\n"
            "- difference between correlation and support\n"
            "- claims containing multiple components"
        ),
    },
    {
        "id": "B",
        "role": "skeptical analyst",
        "objective": (
            "Find what part of the CLAIM is NOT proven by the EVIDENCE.\n"
            "Pay special attention to:\n"
            "- PARTIAL\n"
            "- over-specificity\n"
            "- modality\n"
            "- quantitative claims\n"
            "- universal claims\n"
            "- absolute claims (\"requires\", \"always\", \"all\", \"never\", etc.)"
        ),
    },
    {
        "id": "C",
        "role": "contradiction analyst",
        "objective": (
            "Actively search for EVIDENCE incompatible with the CLAIM.\n"
            "Pay special attention to:\n"
            "- explicit contradiction\n"
            "- implicit contradiction\n"
            "- negation\n"
            "- modality\n"
            "- mutually exclusive states"
        ),
    },
    {
        "id": "D",
        "role": "context/entity analyst",
        "objective": (
            "Verify that CLAIM and EVIDENCE refer to exactly the same\n"
            "subject and context.\n"
            "Pay special attention to:\n"
            "- wrong subject\n"
            "- wrong product\n"
            "- wrong framework\n"
            "- wrong environment\n"
            "- wrong sector\n"
            "- wrong lifecycle phase\n"
            "- wrong version/platform"
        ),
    },
]


def _build_independent_prompt(worker_spec: dict, claim: str, evidence: str) -> str:
    """Phase 1: prompt para evaluacion independiente."""
    ev = evidence[:800].strip()
    cl = claim[:400].strip()
    return (
        f"You are Worker {worker_spec['id']}, a {worker_spec['role']}.\n"
        f"\n"
        f"Your objective:\n"
        f"{worker_spec['objective']}\n"
        f"\n"
        f"Relations: SUPPORTS, PARTIAL, CONTRADICTS, UNRELATED\n"
        f"\n"
        f"Respond with EXACTLY this JSON format and nothing else:\n"
        f'{{"worker": "{worker_spec["id"]}", "relation": "SUPPORTS|PARTIAL|CONTRADICTS|UNRELATED", "confidence": 0.0, "reason": "short explanation"}}\n'
        f"\n"
        f"CLAIM: {cl}\n"
        f"EVIDENCE: {ev}\n"
        f"\n"
        f"JSON:"
    )


def _build_challenge_prompt(
    worker_spec: dict,
    claim: str,
    evidence: str,
    own_assessment: dict,
    other_assessments: List[dict],
) -> str:
    """Phase 3: prompt para challenge round."""
    ev = evidence[:800].strip()
    cl = claim[:400].strip()

    own_json = json.dumps(own_assessment, ensure_ascii=False)
    others_json = json.dumps(other_assessments, ensure_ascii=False)

    return (
        f"You are Worker {worker_spec['id']}, a {worker_spec['role']}.\n"
        f"\n"
        f"CLAIM: {cl}\n"
        f"EVIDENCE: {ev}\n"
        f"\n"
        f"Your initial assessment:\n"
        f"{own_json}\n"
        f"\n"
        f"Other workers' assessments:\n"
        f"{others_json}\n"
        f"\n"
        f"You are now in a challenge round. You can see what other workers concluded.\n"
        f"There is NO ground truth available. Evaluate arguments, not popularity.\n"
        f"\n"
        f"Answer these questions:\n"
        f"1. What is the strongest argument AGAINST your current classification?\n"
        f"2. Which other worker's classification poses the greatest risk of error?\n"
        f"3. What concrete evidence justifies changing or maintaining your classification?\n"
        f"4. Would you change your classification?\n"
        f"\n"
        f"Respond with EXACTLY this JSON format and nothing else:\n"
        f'{{"worker": "{worker_spec["id"]}", "current_relation": "{own_assessment.get("relation", "")}", '
        f'"strongest_counterargument": "...", "challenged_worker": "A|B|C|D", '
        f'"assessment": "...", "change_decision": "KEEP|CHANGE", '
        f'"proposed_relation": "SUPPORTS|PARTIAL|CONTRADICTS|UNRELATED", "confidence": 0.0}}\n'
        f"\n"
        f"JSON:"
    )


def _build_judge_prompt(
    claim: str,
    evidence: str,
    initial_assessments: List[dict],
    challenge_responses: List[dict],
) -> str:
    """Phase 4: prompt para el Judge final."""
    ev = evidence[:800].strip()
    cl = claim[:400].strip()

    initial_json = json.dumps(initial_assessments, ensure_ascii=False, indent=2)
    challenges_json = json.dumps(challenge_responses, ensure_ascii=False, indent=2)

    return (
        "You are the final semantic assessment judge.\n"
        "\n"
        "Determine the relationship between CLAIM and EVIDENCE.\n"
        "\n"
        "You have access to independent expert assessments and their subsequent\n"
        "challenges. Do not choose a result merely because it has majority support.\n"
        "Evaluate the arguments and evidence yourself.\n"
        "\n"
        "Pay special attention to:\n"
        "- missing claim components\n"
        "- contradiction\n"
        "- negation\n"
        "- subject identity\n"
        "- context\n"
        "- specificity\n"
        "- modality\n"
        "- absolute claims\n"
        "\n"
        "Return exactly one relation: SUPPORTS, PARTIAL, CONTRADICTS, or UNRELATED\n"
        "Also provide confidence and a concise decision rationale.\n"
        "\n"
        f"CLAIM: {cl}\n"
        f"EVIDENCE: {ev}\n"
        f"\n"
        f"Initial worker assessments:\n"
        f"{initial_json}\n"
        f"\n"
        f"Challenge responses:\n"
        f"{challenges_json}\n"
        f"\n"
        'Respond with EXACTLY this JSON format and nothing else:\n'
        '{"relation": "SUPPORTS|PARTIAL|CONTRADICTS|UNRELATED", "confidence": 0.0, "reason": "..."}\n'
        "\n"
        "JSON:"
    )


# ==================== JSON Parsing ====================


def _extract_json(raw: str) -> Optional[dict]:
    """Intenta extraer JSON de la respuesta del modelo."""
    if not raw:
        return None
    text = raw.strip()

    # Intentar parse directo
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Buscar primer { y ultimo }
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        chunk = text[start : end + 1]
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            pass

    # Intentar linea por linea
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass

    return None


def _normalize_relation(rel: str) -> str:
    """Normaliza una relacion al vocabulario canonico."""
    if not rel:
        return "UNRELATED"
    r = rel.strip().upper().rstrip(".")
    for valid in VALID_RELATIONS:
        if valid in r:
            return valid
    if "SUPPORT" in r:
        return "SUPPORTS"
    if "CONTRADICT" in r:
        return "CONTRADICTS"
    if "UNRELAT" in r or "IRRELEV" in r:
        return "UNRELATED"
    if "PARTIAL" in r:
        return "PARTIAL"
    return "UNRELATED"


def _parse_assessment(raw: str, worker_id: str) -> dict:
    """Parsea la respuesta de Phase 1."""
    parsed = _extract_json(raw)
    if parsed:
        return {
            "worker": parsed.get("worker", worker_id),
            "relation": _normalize_relation(parsed.get("relation", "")),
            "confidence": float(parsed.get("confidence", 0.5)),
            "reason": str(parsed.get("reason", ""))[:300],
        }
    # Fallback: extraer primera palabra
    rel = _normalize_relation(raw.split()[0] if raw.split() else "")
    return {
        "worker": worker_id,
        "relation": rel,
        "confidence": 0.3,
        "reason": f"[fallback parse] {raw[:200]}",
    }


def _parse_challenge(raw: str, worker_id: str, current_relation: str) -> dict:
    """Parsea la respuesta de Phase 3."""
    parsed = _extract_json(raw)
    if parsed:
        decision = parsed.get("change_decision", "KEEP").strip().upper()
        if decision not in ("KEEP", "CHANGE"):
            decision = "KEEP"
        return {
            "worker": parsed.get("worker", worker_id),
            "current_relation": _normalize_relation(parsed.get("current_relation", current_relation)),
            "strongest_counterargument": str(parsed.get("strongest_counterargument", ""))[:300],
            "challenged_worker": str(parsed.get("challenged_worker", ""))[:2],
            "assessment": str(parsed.get("assessment", ""))[:300],
            "change_decision": decision,
            "proposed_relation": _normalize_relation(parsed.get("proposed_relation", current_relation)),
            "confidence": float(parsed.get("confidence", 0.5)),
        }
    return {
        "worker": worker_id,
        "current_relation": current_relation,
        "strongest_counterargument": "",
        "challenged_worker": "",
        "assessment": f"[fallback parse] {raw[:200]}",
        "change_decision": "KEEP",
        "proposed_relation": current_relation,
        "confidence": 0.3,
    }


def _parse_judge(raw: str) -> dict:
    """Parsea la respuesta del Judge."""
    parsed = _extract_json(raw)
    if parsed:
        return {
            "relation": _normalize_relation(parsed.get("relation", "")),
            "confidence": float(parsed.get("confidence", 0.5)),
            "reason": str(parsed.get("reason", ""))[:300],
        }
    rel = _normalize_relation(raw.split()[0] if raw.split() else "")
    return {
        "relation": rel,
        "confidence": 0.3,
        "reason": f"[fallback parse] {raw[:200]}",
    }


# ==================== Ensemble Aggregation ====================


def _aggregate_initial(assessments: List[dict]) -> Tuple[str, float, dict]:
    """
    Confidence-weighted majority vote (replica del strategy existente).
    Returns (relation, confidence, metadata).
    """
    valid = [a for a in assessments if a.get("relation") in SEMANTIC_RELATIONS]
    if not valid:
        return "UNRELATED", 0.0, {"agreement": 0.0, "votes": []}

    relation_weights: Dict[str, float] = {}
    relation_votes: Dict[str, int] = {}
    for a in valid:
        rel = a["relation"]
        relation_weights[rel] = relation_weights.get(rel, 0.0) + a.get("confidence", 0.5)
        relation_votes[rel] = relation_votes.get(rel, 0) + 1

    final_relation = max(relation_weights, key=relation_weights.get)
    total_weight = sum(relation_weights.values())
    final_confidence = relation_weights[final_relation] / total_weight if total_weight > 0 else 0.0
    agreement = relation_votes[final_relation] / len(valid)

    return final_relation, final_confidence, {
        "agreement": round(agreement, 4),
        "agreement_fraction": f"{relation_votes[final_relation]}/{len(valid)}",
        "votes": [{"worker": a["worker"], "relation": a["relation"], "confidence": a.get("confidence", 0.5)} for a in valid],
        "relation_weights": {k: round(v, 3) for k, v in relation_weights.items()},
    }


def _has_disagreement(assessments: List[dict]) -> bool:
    """True si los workers no estan unanimes."""
    rels = {a["relation"] for a in assessments if a.get("relation") in SEMANTIC_RELATIONS}
    return len(rels) > 1


# ==================== Experiment Runner ====================


@dataclass
class CaseResult:
    case_id: str
    category: str
    ground_truth: str
    # Phase 1
    worker_assessments: List[dict] = field(default_factory=list)
    # Phase 2
    initial_relation: str = ""
    initial_confidence: float = 0.0
    initial_agreement: float = 0.0
    initial_meta: dict = field(default_factory=dict)
    # Phase 3
    debate_triggered: bool = False
    challenge_responses: List[dict] = field(default_factory=list)
    workers_changed: List[str] = field(default_factory=list)
    # Phase 4
    final_relation: str = ""
    final_confidence: float = 0.0
    judge_reason: str = ""
    # Metrics
    initial_correct: bool = False
    final_correct: bool = False
    # Timing
    phase1_latency_s: float = 0.0
    phase3_latency_s: float = 0.0
    phase4_latency_s: float = 0.0
    total_latency_s: float = 0.0


def run_case(
    case: dict,
    model_name: str,
    num_gpu: int,
    mode: str,
    num_predict: int,
    timeout: float,
    base_url: str = "http://localhost:11434",
    backend: str = "ollama",
    num_ctx: int = 0,
) -> CaseResult:
    """Ejecuta un caso a traves de las 4 fases."""
    claim = case["claim"]
    evidence = case["evidence"]
    result = CaseResult(
        case_id=case["id"],
        category=case["category"],
        ground_truth=case["expected"],
    )

    default_opts: Dict[str, Any] = {"num_predict": num_predict, "temperature": 0.0, "num_thread": 4}
    if num_ctx > 0:
        default_opts["num_ctx"] = num_ctx

    if backend == "llama-server":
        provider = LlamaServerProvider(
            model=model_name,
            base_url=base_url,
            num_gpu=num_gpu,
            default_options={"num_predict": num_predict, "temperature": 0.0},
        )
    else:
        provider = StructuredOllamaProvider(
            model=model_name,
            base_url=base_url,
            num_gpu=num_gpu,
            default_options=default_opts,
        )

    # ==================== Phase 1: Independent Assessment ====================
    t0 = time.time()
    assessments = []
    for spec in WORKER_SPECS:
        prompt = _build_independent_prompt(spec, claim, evidence)
        raw = provider.generate_structured(prompt, ASSESSMENT_SCHEMA, timeout=timeout)
        parsed = _parse_assessment(raw, spec["id"])
        assessments.append(parsed)
    result.phase1_latency_s = round(time.time() - t0, 2)
    result.worker_assessments = assessments

    # ==================== Phase 2: Initial Ensemble (frozen) ====================
    initial_rel, initial_conf, initial_meta = _aggregate_initial(assessments)
    result.initial_relation = initial_rel
    result.initial_confidence = round(initial_conf, 4)
    result.initial_agreement = initial_meta.get("agreement", 0.0)
    result.initial_meta = initial_meta
    result.initial_correct = (initial_rel == result.ground_truth)

    # ==================== Phase 3: Disagreement Detection + Challenge ====================
    disagreement = _has_disagreement(assessments)

    if mode == "independent":
        result.debate_triggered = False
        result.final_relation = initial_rel
        result.final_confidence = initial_conf
        result.final_correct = result.initial_correct
    elif mode == "debate-on-disagreement":
        result.debate_triggered = disagreement
        if not disagreement:
            # No debate needed: final = initial
            result.final_relation = initial_rel
            result.final_confidence = initial_conf
            result.final_correct = result.initial_correct
    elif mode == "debate-all":
        result.debate_triggered = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if result.debate_triggered:
        # Challenge round
        t0 = time.time()
        challenge_responses = []
        for i, spec in enumerate(WORKER_SPECS):
            own = assessments[i]
            others = [a for j, a in enumerate(assessments) if j != i]
            prompt = _build_challenge_prompt(spec, claim, evidence, own, others)
            raw = provider.generate_structured(prompt, CHALLENGE_SCHEMA, timeout=timeout)
            challenge = _parse_challenge(raw, spec["id"], own["relation"])
            challenge_responses.append(challenge)
        result.phase3_latency_s = round(time.time() - t0, 2)
        result.challenge_responses = challenge_responses
        result.workers_changed = [c["worker"] for c in challenge_responses if c["change_decision"] == "CHANGE"]

        # ==================== Phase 4: Final Judge ====================
        t0 = time.time()
        judge_prompt = _build_judge_prompt(claim, evidence, assessments, challenge_responses)
        raw = provider.generate_structured(judge_prompt, RELATION_SCHEMA, timeout=timeout)
        judge_result = _parse_judge(raw)
        result.phase4_latency_s = round(time.time() - t0, 2)
        result.final_relation = judge_result["relation"]
        result.final_confidence = round(judge_result["confidence"], 4)
        result.judge_reason = judge_result["reason"]
        result.final_correct = (result.final_relation == result.ground_truth)

    result.total_latency_s = round(result.phase1_latency_s + result.phase3_latency_s + result.phase4_latency_s, 2)
    return result


# ==================== Metrics ====================


def compute_metrics(results: List[CaseResult], mode: str) -> dict:
    n = len(results)
    initial_correct = sum(1 for r in results if r.initial_correct)
    final_correct = sum(1 for r in results if r.final_correct)

    initial_acc = initial_correct / n if n > 0 else 0.0
    final_acc = final_correct / n if n > 0 else 0.0
    delta = final_acc - initial_acc

    # Correction: initial WRONG -> final CORRECT
    initial_wrong = [r for r in results if not r.initial_correct]
    corrections = sum(1 for r in initial_wrong if r.final_correct)
    correction_rate = corrections / len(initial_wrong) if initial_wrong else 0.0

    # Damage: initial CORRECT -> final WRONG
    initial_right = [r for r in results if r.initial_correct]
    damage = sum(1 for r in initial_right if not r.final_correct)
    damage_rate = damage / len(initial_right) if initial_right else 0.0

    # Stability: initial == final
    stable = sum(1 for r in results if r.initial_relation == r.final_relation)
    stability_rate = stable / n if n > 0 else 0.0

    # Debate trigger
    debates = sum(1 for r in results if r.debate_triggered)
    debate_trigger_rate = debates / n if n > 0 else 0.0

    # Revision rate
    total_workers_in_debate = sum(len(WORKER_SPECS) for r in results if r.debate_triggered)
    total_changed = sum(len(r.workers_changed) for r in results)
    revision_rate = total_changed / total_workers_in_debate if total_workers_in_debate > 0 else 0.0

    # By category
    by_cat = defaultdict(lambda: {"total": 0, "initial_correct": 0, "final_correct": 0})
    for r in results:
        by_cat[r.category]["total"] += 1
        if r.initial_correct:
            by_cat[r.category]["initial_correct"] += 1
        if r.final_correct:
            by_cat[r.category]["final_correct"] += 1

    # By disagreement level
    by_agreement = defaultdict(lambda: {"total": 0, "initial_correct": 0, "final_correct": 0})
    for r in results:
        # Bucket: unanimous (1.0) vs split (<1.0)
        bucket = "unanimous" if r.initial_agreement >= 1.0 else "split"
        by_agreement[bucket]["total"] += 1
        if r.initial_correct:
            by_agreement[bucket]["initial_correct"] += 1
        if r.final_correct:
            by_agreement[bucket]["final_correct"] += 1

    # Transition matrix: initial -> final
    relations = VALID_RELATIONS
    transition = {ri: {rf: 0 for rf in relations} for ri in relations}
    for r in results:
        if r.initial_relation in transition and r.final_relation in transition[r.initial_relation]:
            transition[r.initial_relation][r.final_relation] += 1

    return {
        "n": n,
        "mode": mode,
        "initial_accuracy": round(initial_acc, 4),
        "final_accuracy": round(final_acc, 4),
        "accuracy_delta": round(delta, 4),
        "initial_correct": initial_correct,
        "final_correct": final_correct,
        "corrections": corrections,
        "correction_rate": round(correction_rate, 4),
        "damage": damage,
        "damage_rate": round(damage_rate, 4),
        "stability_rate": round(stability_rate, 4),
        "debate_trigger_rate": round(debate_trigger_rate, 4),
        "debates_triggered": debates,
        "revision_rate": round(revision_rate, 4),
        "total_workers_changed": total_changed,
        "by_category": {
            cat: {
                "total": s["total"],
                "initial_accuracy": round(s["initial_correct"] / s["total"], 4) if s["total"] > 0 else 0.0,
                "final_accuracy": round(s["final_correct"] / s["total"], 4) if s["total"] > 0 else 0.0,
            }
            for cat, s in sorted(by_cat.items())
        },
        "by_agreement": {
            bucket: {
                "total": s["total"],
                "initial_accuracy": round(s["initial_correct"] / s["total"], 4) if s["total"] > 0 else 0.0,
                "final_accuracy": round(s["final_correct"] / s["total"], 4) if s["total"] > 0 else 0.0,
            }
            for bucket, s in sorted(by_agreement.items())
        },
        "transition_matrix": transition,
    }


# ==================== Report Generation ====================


def generate_markdown_report(
    model_name: str,
    mode: str,
    metrics: dict,
    results: List[CaseResult],
    config: dict,
) -> str:
    lines = []
    lines.append("# Micro-Coliseum - Deliberative Semantic Assessment\n")
    lines.append("## Configuration\n")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Model | `{model_name}` |")
    lines.append(f"| Mode | `{mode}` |")
    lines.append(f"| Workers | 4 (A=entailment, B=skeptical, C=contradiction, D=context) |")
    lines.append(f"| Cases | {metrics['n']} |")
    lines.append(f"| GPU | {config.get('gpu', False)} |")
    lines.append(f"| num_predict | {config.get('num_predict', 10)} |")
    lines.append(f"| temperature | 0.0 |")
    lines.append(f"| Benchmark | semantic_assessment_benchmark_v2.json |")
    lines.append(f"| Timestamp | {config.get('timestamp', '')} |")
    lines.append("")

    lines.append("## Accuracy\n")
    lines.append("| Metric | Initial | Deliberative | Delta |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| Accuracy | {metrics['initial_accuracy']:.1%} | {metrics['final_accuracy']:.1%} | {metrics['accuracy_delta']:+.1%} |")
    lines.append(f"| Correct | {metrics['initial_correct']}/{metrics['n']} | {metrics['final_correct']}/{metrics['n']} | {metrics['final_correct'] - metrics['initial_correct']:+d} |")
    lines.append("")

    lines.append("## Corrections vs Damage\n")
    lines.append("```\n")
    lines.append("                    FINAL")
    lines.append("                     ^")
    lines.append("        corrections  |  damage")
    lines.append("                     |")
    lines.append("INITIAL -------------+-------------")
    lines.append("                     |")
    lines.append("                 unchanged")
    lines.append("```\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---:|")
    lines.append(f"| Corrections (wrong->right) | {metrics['corrections']} |")
    lines.append(f"| Correction rate | {metrics['correction_rate']:.1%} |")
    lines.append(f"| Damage (right->wrong) | {metrics['damage']} |")
    lines.append(f"| Damage rate | {metrics['damage_rate']:.1%} |")
    lines.append(f"| Net effect | {metrics['corrections'] - metrics['damage']:+d} |")
    lines.append(f"| Stability rate | {metrics['stability_rate']:.1%} |")
    lines.append("")

    lines.append("## Debate Statistics\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---:|")
    lines.append(f"| Debates triggered | {metrics['debates_triggered']}/{metrics['n']} |")
    lines.append(f"| Debate trigger rate | {metrics['debate_trigger_rate']:.1%} |")
    lines.append(f"| Workers changed opinion | {metrics['total_workers_changed']} |")
    lines.append(f"| Revision rate | {metrics['revision_rate']:.1%} |")
    lines.append("")

    lines.append("## Accuracy by Category\n")
    lines.append("| Category | N | Initial | Deliberative | Delta |")
    lines.append("|---|---:|---:|---:|---:|")
    for cat, s in sorted(metrics["by_category"].items()):
        d = s["final_accuracy"] - s["initial_accuracy"]
        lines.append(f"| {cat} | {s['total']} | {s['initial_accuracy']:.1%} | {s['final_accuracy']:.1%} | {d:+.1%} |")
    lines.append("")

    lines.append("## Accuracy by Agreement Level\n")
    lines.append("| Agreement | N | Initial | Deliberative | Delta |")
    lines.append("|---|---:|---:|---:|---:|")
    for bucket, s in sorted(metrics["by_agreement"].items()):
        d = s["final_accuracy"] - s["initial_accuracy"]
        lines.append(f"| {bucket} | {s['total']} | {s['initial_accuracy']:.1%} | {s['final_accuracy']:.1%} | {d:+.1%} |")
    lines.append("")

    lines.append("## Initial -> Final Transition Matrix\n")
    lines.append("```\n")
    # Header
    header = "         "
    for rf in VALID_RELATIONS:
        header += f" {rf[:4]:>5s}"
    lines.append(header)
    lines.append("         " + "-" * (len(VALID_RELATIONS) * 6))
    for ri in VALID_RELATIONS:
        row = f"{ri[:7]:>8s}"
        for rf in VALID_RELATIONS:
            count = metrics["transition_matrix"][ri][rf]
            row += f" {count:>5d}"
        lines.append(row)
    lines.append("```\n")

    lines.append("## Case-level Results\n")
    lines.append("| ID | Category | Expected | Initial | Final | Init OK | Final OK | Debate | Changed |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        changed_str = ",".join(r.workers_changed) if r.workers_changed else "-"
        debate_str = "Y" if r.debate_triggered else "N"
        lines.append(
            f"| {r.case_id} | {r.category} | {r.ground_truth} | {r.initial_relation} | {r.final_relation} | "
            f"{'Y' if r.initial_correct else 'N'} | {'Y' if r.final_correct else 'N'} | {debate_str} | {changed_str} |"
        )
    lines.append("")

    lines.append("## Conclusion\n")
    net = metrics["corrections"] - metrics["damage"]
    if mode == "independent":
        lines.append("Mode: `independent` (baseline, no debate). This run establishes the initial ensemble accuracy without deliberation.\n")
    elif net > 0 and metrics["accuracy_delta"] > 0:
        lines.append(
            f"Mode: `{mode}`. The deliberation produced a **net positive** effect: "
            f"{metrics['corrections']} corrections vs {metrics['damage']} damage (net {net:+d}). "
            f"Accuracy improved from {metrics['initial_accuracy']:.1%} to {metrics['final_accuracy']:.1%} "
            f"(+{metrics['accuracy_delta']:.1%}). "
            f"This provides evidence that deliberative interaction between workers can correct errors "
            f"that independent voting cannot capture.\n"
        )
    elif net < 0:
        lines.append(
            f"Mode: `{mode}`. The deliberation produced a **net negative** effect: "
            f"{metrics['corrections']} corrections vs {metrics['damage']} damage (net {net:+d}). "
            f"Accuracy changed from {metrics['initial_accuracy']:.1%} to {metrics['final_accuracy']:.1%} "
            f"({metrics['accuracy_delta']:+.1%}). "
            f"The debate introduced more errors than it corrected. H0 is supported.\n"
        )
    elif net == 0 and metrics["corrections"] > 0:
        lines.append(
            f"Mode: `{mode}`. The deliberation produced a **neutral** effect: "
            f"{metrics['corrections']} corrections vs {metrics['damage']} damage (net {net:+d}). "
            f"Accuracy changed from {metrics['initial_accuracy']:.1%} to {metrics['final_accuracy']:.1%} "
            f"({metrics['accuracy_delta']:+.1%}). "
            f"The debate corrected as many errors as it introduced. "
            f"May be useful as a tie-breaking mechanism on difficult cases only.\n"
        )
    else:
        lines.append(
            f"Mode: `{mode}`. The deliberation had **no effect**: "
            f"0 corrections and 0 damage. Accuracy unchanged at {metrics['final_accuracy']:.1%}. "
            f"The judge confirmed the initial ensemble in all cases.\n"
        )

    return "\n".join(lines)


# ==================== Cleanup ====================


def unload_model(model_name: str, base_url: str = "http://localhost:11434") -> None:
    """Descarga el modelo de Ollama para liberar RAM."""
    try:
        requests.post(
            f"{base_url}/api/generate",
            json={"model": model_name, "keep_alive": 0},
            timeout=10,
        )
        print(f"  unloaded: {model_name}", flush=True)
    except Exception as e:
        print(f"  unload failed: {model_name} ({e})", flush=True)


# ==================== Main ====================


def main() -> int:
    parser = argparse.ArgumentParser(description="Micro-Coliseum: Deliberative Semantic Assessment")
    parser.add_argument("--model", required=True, help="Ollama model name")
    parser.add_argument("--mode", required=True, choices=["independent", "debate-on-disagreement", "debate-all"])
    parser.add_argument("--gpu", action="store_true", help="Use GPU (num_gpu=99)")
    parser.add_argument("--num-predict", type=int, default=60, help="Max tokens per response")
    parser.add_argument("--timeout", type=float, default=120.0, help="Timeout per LLM call (s)")
    parser.add_argument("--label", default="", help="Label for output files")
    parser.add_argument("--port", type=int, default=11434,
                        help="Ollama instance port (default: 11434). "
                             "Use a dedicated instance to avoid model thrashing.")
    parser.add_argument("--backend", default="ollama", choices=["ollama", "llama-server"],
                        help="Backend: ollama (default) or llama-server (BitNet).")
    parser.add_argument("--base-url", default="",
                        help="Base URL override (e.g. http://127.0.0.1:8081 for llama-server). "
                             "If empty, derives from --port for ollama.")
    parser.add_argument("--num-ctx", type=int, default=0,
                        help="Context size override (0 = model default). "
                             "Required for models with large default ctx (e.g. Qwen3 base = 40960).")
    args = parser.parse_args()

    num_gpu = 99 if args.gpu else 0
    if args.base_url:
        base_url = args.base_url
    elif args.backend == "llama-server":
        base_url = f"http://127.0.0.1:{args.port}"
    else:
        base_url = f"http://localhost:{args.port}"
    label = args.label or args.model.replace("/", "_").replace(":", "_")
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    print("=" * 70, flush=True)
    print("MICRO-COLISEUM: Deliberative Semantic Assessment", flush=True)
    print("=" * 70, flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Mode: {args.mode}", flush=True)
    print(f"Backend: {args.backend}", flush=True)
    print(f"GPU: {args.gpu} (num_gpu={num_gpu})", flush=True)
    print(f"num_predict: {args.num_predict}", flush=True)
    if args.num_ctx > 0:
        print(f"num_ctx: {args.num_ctx} (override)", flush=True)
    print(f"Endpoint: {base_url}", flush=True)
    print(flush=True)

    # Load benchmark
    with BENCHMARK.open("r", encoding="utf-8") as f:
        bench = json.load(f)
    cases = bench["cases"]
    n = len(cases)
    print(f"Cases: {n}", flush=True)
    print(flush=True)

    # Verify model availability
    if args.backend == "llama-server":
        check_provider = LlamaServerProvider(model=args.model, base_url=base_url, num_gpu=num_gpu)
    else:
        check_provider = OllamaModelProvider(model=args.model, base_url=base_url, num_gpu=num_gpu, default_options={"num_predict": args.num_predict, "temperature": 0.0})
    if not check_provider.is_available():
        print(f"ERROR: {args.backend} not available at {base_url}", flush=True)
        return 1
    print(f"Model OK: {args.model} ({base_url})", flush=True)
    print(flush=True)

    # Run cases
    results: List[CaseResult] = []
    t_total = time.time()

    for i, case in enumerate(cases):
        print(f"  [{i+1}/{n}] {case['id']} ({case['category']})... ", end="", flush=True)
        t0 = time.time()
        try:
            cr = run_case(
                case=case,
                model_name=args.model,
                num_gpu=num_gpu,
                mode=args.mode,
                num_predict=args.num_predict,
                timeout=args.timeout,
                base_url=base_url,
                backend=args.backend,
                num_ctx=args.num_ctx,
            )
        except Exception as e:
            print(f"ERROR: {e}", flush=True)
            cr = CaseResult(
                case_id=case["id"], category=case["category"], ground_truth=case["expected"],
                final_relation="ERROR", initial_relation="ERROR",
            )
        dt = time.time() - t0
        results.append(cr)
        status = ""
        if cr.debate_triggered:
            changed = len(cr.workers_changed)
            status = f"debate({changed} changed) "
        status += f"init={cr.initial_relation}({'OK' if cr.initial_correct else 'X'}) "
        status += f"final={cr.final_relation}({'OK' if cr.final_correct else 'X'})"
        print(f"{status} [{dt:.1f}s]", flush=True)

    wall_time = time.time() - t_total
    print(f"\nWall time: {wall_time:.0f}s ({wall_time/60:.1f} min)", flush=True)

    # Compute metrics
    metrics = compute_metrics(results, args.mode)

    # Print summary
    print(flush=True)
    print("=" * 70, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"  Initial accuracy:    {metrics['initial_accuracy']:.1%} ({metrics['initial_correct']}/{n})", flush=True)
    print(f"  Deliberative accuracy: {metrics['final_accuracy']:.1%} ({metrics['final_correct']}/{n})", flush=True)
    print(f"  Delta:               {metrics['accuracy_delta']:+.1%}", flush=True)
    print(f"  Corrections:         {metrics['corrections']} (rate: {metrics['correction_rate']:.1%})", flush=True)
    print(f"  Damage:              {metrics['damage']} (rate: {metrics['damage_rate']:.1%})", flush=True)
    print(f"  Net:                 {metrics['corrections'] - metrics['damage']:+d}", flush=True)
    print(f"  Stability:           {metrics['stability_rate']:.1%}", flush=True)
    print(f"  Debates triggered:   {metrics['debates_triggered']}/{n} ({metrics['debate_trigger_rate']:.1%})", flush=True)
    print(f"  Revision rate:       {metrics['revision_rate']:.1%}", flush=True)
    print(flush=True)

    # Transition matrix
    print("Transition matrix (initial -> final):", flush=True)
    header = "         "
    for rf in VALID_RELATIONS:
        header += f" {rf[:4]:>5s}"
    print(header, flush=True)
    print("         " + "-" * (len(VALID_RELATIONS) * 6), flush=True)
    for ri in VALID_RELATIONS:
        row = f"{ri[:7]:>8s}"
        for rf in VALID_RELATIONS:
            count = metrics["transition_matrix"][ri][rf]
            row += f" {count:>5d}"
        print(row, flush=True)
    print(flush=True)

    # Config for report
    config = {
        "gpu": args.gpu,
        "num_predict": args.num_predict,
        "temperature": 0.0,
        "num_thread": 4,
        "backend": args.backend,
        "num_ctx": args.num_ctx,
        "timestamp": timestamp,
        "wall_time_s": round(wall_time, 1),
    }

    # Generate reports
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / f"microcoliseum_{label}_{args.mode}.json"
    md_path = OUTPUT_DIR / f"microcoliseum_{label}_{args.mode}.md"

    report = {
        "experiment": "microcoliseum_deliberation",
        "timestamp": timestamp,
        "model": args.model,
        "mode": args.mode,
        "backend": args.backend,
        "gpu": args.gpu,
        "num_gpu": num_gpu,
        "num_predict": args.num_predict,
        "temperature": 0.0,
        "num_thread": 4,
        "num_ctx": args.num_ctx,
        "benchmark": "semantic_assessment_benchmark_v2.json",
        "case_count": n,
        "wall_time_s": round(wall_time, 1),
        "metrics": metrics,
        "cases": [
            {
                "case_id": r.case_id,
                "category": r.category,
                "ground_truth": r.ground_truth,
                "worker_assessments": r.worker_assessments,
                "initial_relation": r.initial_relation,
                "initial_confidence": r.initial_confidence,
                "initial_agreement": r.initial_agreement,
                "initial_correct": r.initial_correct,
                "debate_triggered": r.debate_triggered,
                "challenge_responses": r.challenge_responses,
                "workers_changed": r.workers_changed,
                "final_relation": r.final_relation,
                "final_confidence": r.final_confidence,
                "final_correct": r.final_correct,
                "judge_reason": r.judge_reason,
                "phase1_latency_s": r.phase1_latency_s,
                "phase3_latency_s": r.phase3_latency_s,
                "phase4_latency_s": r.phase4_latency_s,
                "total_latency_s": r.total_latency_s,
            }
            for r in results
        ],
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md_content = generate_markdown_report(args.model, args.mode, metrics, results, config)
    with md_path.open("w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"JSON report: {json_path}", flush=True)
    print(f"MD report:   {md_path}", flush=True)

    # Cleanup
    print(flush=True)
    print("Unloading model from Ollama...", flush=True)
    unload_model(args.model, base_url=base_url)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
