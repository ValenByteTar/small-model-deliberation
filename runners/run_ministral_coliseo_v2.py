"""Coliseo v2 aislado para Ministral con salida estructurada estricta."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK = ROOT / "benchmarks" / "semantic_assessment_v2.json"
OUTPUT = ROOT / "results" / "raw" / "coliseo_v2_ministral_structured.json"
MODEL = "TechyShishy/ministral-3:3b-reasoning-2512-q4_K_M"
BASE_URL = "http://localhost:11434"
RELATIONS = ("SUPPORTS", "CONTRADICTS", "UNRELATED", "PARTIAL")
CONFIGS = {
    "single": ("neutral",),
    "ensemble_2": ("entailment", "skeptical"),
    "ensemble_4": ("entailment", "skeptical", "contradiction", "neutral"),
}
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "relation": {"type": "string", "enum": list(RELATIONS)},
    },
    "required": ["relation"],
    "additionalProperties": False,
}
ROLE_INSTRUCTIONS = {
    "neutral": "Evaluate objectively and give equal weight to all four relations.",
    "entailment": "Focus on whether the evidence logically supports the claim.",
    "skeptical": "Be conservative: require clear evidence before choosing SUPPORTS.",
    "contradiction": "Actively check for facts that contradict or undermine the claim.",
}


def classify(claim: str, evidence: str, role: str) -> dict[str, Any]:
    prompt = (
        "Classify the relation between CLAIM and EVIDENCE.\n"
        f"Role: {ROLE_INSTRUCTIONS[role]}\n"
        "Allowed relations: SUPPORTS, CONTRADICTS, UNRELATED, PARTIAL.\n"
        "Return only the JSON object required by the schema. Do not explain.\n\n"
        f"CLAIM: {claim[:300]}\n"
        f"EVIDENCE: {evidence[:600]}"
    )
    started = time.time()
    response = requests.post(
        f"{BASE_URL}/api/generate",
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "format": RESPONSE_SCHEMA,
            "options": {
                "num_gpu": 99,
                "num_predict": 32,
                "temperature": 0.0,
                "num_thread": 4,
            },
            "keep_alive": "10m",
        },
        timeout=90,
    )
    response.raise_for_status()
    payload = response.json()
    raw = (payload.get("response") or "").strip()
    result: dict[str, Any] = {
        "role": role,
        "raw_response": raw,
        "latency_s": round(time.time() - started, 2),
        "protocol_valid": False,
        "relation": "ERROR",
    }
    try:
        parsed = json.loads(raw)
        relation = parsed.get("relation")
        if relation in RELATIONS and set(parsed) == {"relation"}:
            result["relation"] = relation
            result["protocol_valid"] = True
        else:
            result["parse_error"] = "schema_mismatch"
    except (json.JSONDecodeError, AttributeError):
        result["parse_error"] = "invalid_json"
    return result


def assess_case(case: dict[str, Any], roles: tuple[str, ...]) -> dict[str, Any]:
    started = time.time()
    if len(roles) == 1:
        workers = [classify(case["claim"], case["evidence"], roles[0])]
    else:
        with ThreadPoolExecutor(max_workers=len(roles)) as pool:
            futures = [pool.submit(classify, case["claim"], case["evidence"], role) for role in roles]
            workers = [future.result() for future in futures]

    valid = [worker for worker in workers if worker["protocol_valid"]]
    votes: dict[str, int] = defaultdict(int)
    for worker in valid:
        votes[worker["relation"]] += 1
    relation = max(votes, key=votes.get) if votes else "ERROR"
    return {
        "id": case["id"],
        "category": case["category"],
        "expected": case["expected"],
        "produced": relation,
        "correct": relation == case["expected"],
        "protocol_valid": len(valid) == len(workers) and bool(valid),
        "latency_s": round(time.time() - started, 2),
        "votes": dict(votes),
        "workers": workers,
    }


def metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    correct = sum(result["correct"] for result in results)
    protocol = sum(result["protocol_valid"] for result in results)
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    for result in results:
        summary = by_category[result["category"]]
        summary["total"] += 1
        summary["correct"] += int(result["correct"])
    return {
        "n": total,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "correct": correct,
        "protocol_validity": round(protocol / total, 4) if total else 0.0,
        "by_category": {
            category: {
                **summary,
                "accuracy": round(summary["correct"] / summary["total"], 4),
            }
            for category, summary in sorted(by_category.items())
        },
    }


def main() -> int:
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    cases = benchmark["cases"]
    print(f"Ministral estructurado: {MODEL}", flush=True)
    print(f"Casos: {len(cases)} | Salida: {OUTPUT}", flush=True)
    tags = requests.get(f"{BASE_URL}/api/tags", timeout=5).json().get("models", [])
    if not any(item.get("name") == MODEL for item in tags):
        print(f"ERROR: modelo no disponible: {MODEL}", flush=True)
        return 1

    report: dict[str, Any] = {
        "pilot": "coliseo_v2_ministral_structured",
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": MODEL,
        "execution": "GPU (num_gpu=99), think=false, structured-json",
        "benchmark": BENCHMARK.name,
        "case_count": len(cases),
        "configs": {},
    }
    try:
        for label, roles in CONFIGS.items():
            print(f"Config: {label} roles={roles}", flush=True)
            results = [assess_case(case, roles) for case in cases]
            summary = metrics(results)
            report["configs"][label] = {"roles": roles, "metrics": summary, "per_case": results}
            print(
                f"  accuracy={summary['accuracy']:.1%} "
                f"protocol={summary['protocol_validity']:.1%}",
                flush=True,
            )
    finally:
        try:
            requests.post(
                f"{BASE_URL}/api/generate",
                json={"model": MODEL, "keep_alive": 0},
                timeout=10,
            )
        except requests.RequestException:
            pass

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Reporte: {OUTPUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
