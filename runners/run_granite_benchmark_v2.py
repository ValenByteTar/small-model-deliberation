"""
Granite 3B Q4 vs SemanticAssessment Benchmark v2 (55 casos, CPU-only).

Corre Granite 4.1 3B Q4 contra el benchmark endurecido con 55 casos
distribuidos en 10 categorias diagnosticas. Reporta accuracy por
categoria y por relacion, identificando donde el modelo falla y por que.

Ver:
  - tests/eval/canonical/semantic_assessment_benchmark_v2.json
  - knowledge/postmortems/PM-003-bitnet-semantic-capacity-insufficient.md

Uso:
    python scripts/run_granite_benchmark_v2.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hybrid_rag.kernel.llm_support import LLMSupport
from hybrid_rag.kernel.state import SEMANTIC_RELATIONS
from hybrid_rag.evaluation.semantic_adapter import SemanticAssessmentAdapter
from hybrid_rag.providers.ollama_provider import OllamaModelProvider

BENCHMARK = ROOT / "tests" / "eval" / "canonical" / "semantic_assessment_benchmark_v2.json"
OUTPUT = ROOT / "tests" / "eval" / "canonical" / "reports" / "granite_benchmark_v2.json"

MODEL = "ibm/granite4.1:3b-q4_K_M"


def main() -> int:
    print("=" * 70, flush=True)
    print("Granite 3B Q4 vs SemanticAssessment Benchmark v2", flush=True)
    print(f"Model: {MODEL} (CPU-only)", flush=True)
    print("=" * 70, flush=True)

    with BENCHMARK.open("r", encoding="utf-8") as f:
        bench = json.load(f)

    cases = bench["cases"]
    n = len(cases)
    print(f"Cases: {n}", flush=True)
    dist = Counter(c["expected"] for c in cases)
    print(f"Expected distribution: {dict(dist)}", flush=True)
    cats = Counter(c["category"] for c in cases)
    print(f"Categories: {dict(sorted(cats.items()))}", flush=True)
    print(flush=True)

    provider = OllamaModelProvider(
        model=MODEL,
        base_url="http://localhost:11434",
        num_gpu=0,
        default_options={"num_predict": 10, "temperature": 0.0, "num_thread": 4},
    )

    if not provider.is_available():
        print(f"ERROR: {MODEL} no disponible", flush=True)
        return 1

    llm = LLMSupport(model_provider=provider, mode="semantic", max_hypotheses=200, max_concurrent=1)
    adapter = SemanticAssessmentAdapter()

    results = []
    correct = 0
    protocol_ok = 0
    total_time = 0.0

    for i, case in enumerate(cases, 1):
        cid = case["id"]
        cat = case["category"]
        claim = case["claim"]
        evidence = case["evidence"]
        expected = case["expected"]
        reason = case["diagnostic_reason"]

        print(f"[{i:2d}/{n}] {cid} ({cat}): ", end="", flush=True)

        t0 = time.time()
        assessment = llm.semantic_assess(
            claim=claim, evidence_text=evidence, evidence_id=cid,
            run_id="granite-bench-v2", timeout=60.0,
        )
        dt = time.time() - t0
        total_time += dt

        if assessment is None:
            print(f"FAIL ({dt:.1f}s)", flush=True)
            results.append({
                "id": cid, "category": cat, "claim": claim, "evidence": evidence[:200],
                "expected": expected, "produced": None, "correct": False,
                "protocol_ok": False, "timing_s": round(dt, 2),
                "diagnostic_reason": reason,
            })
            continue

        is_correct = assessment.relation == expected
        is_protocol = assessment.relation in SEMANTIC_RELATIONS and 0.0 <= assessment.confidence <= 1.0
        if is_correct:
            correct += 1
        if is_protocol:
            protocol_ok += 1

        signal = adapter.adapt(assessment)

        mark = "OK" if is_correct else "XX"
        print(f"{mark} {assessment.relation} (exp={expected}) {dt:.1f}s", flush=True)
        if not is_correct:
            print(f"       diagnostic: {reason[:100]}", flush=True)

        results.append({
            "id": cid, "category": cat, "claim": claim, "evidence": evidence[:200],
            "expected": expected, "produced": assessment.to_dict(),
            "correct": is_correct, "protocol_ok": is_protocol,
            "signal": signal.to_dict(), "timing_s": round(dt, 2),
            "diagnostic_reason": reason,
        })

    # ==================== Analisis ====================
    accuracy = correct / n
    protocol_rate = protocol_ok / n
    avg_time = total_time / n

    # Por categoria
    by_cat = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        by_cat[r["category"]]["total"] += 1
        if r["correct"]:
            by_cat[r["category"]]["correct"] += 1

    # Por relation
    by_rel = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        by_rel[r["expected"]]["total"] += 1
        if r["correct"]:
            by_rel[r["expected"]]["correct"] += 1

    # Errores por categoria
    errors_by_cat = defaultdict(list)
    for r in results:
        if not r["correct"]:
            errors_by_cat[r["category"]].append({
                "id": r["id"],
                "expected": r["expected"],
                "produced": r["produced"]["relation"] if r["produced"] else "ERROR",
                "claim": r["claim"][:80],
            })

    # Confusion matrix
    confusion = defaultdict(lambda: defaultdict(int))
    for r in results:
        exp = r["expected"]
        prod = r["produced"]["relation"] if r["produced"] else "ERROR"
        confusion[exp][prod] += 1

    # ==================== Reporte ====================
    print(flush=True)
    print("=" * 70, flush=True)
    print("RESULTADOS", flush=True)
    print("=" * 70, flush=True)
    print(f"Model: {MODEL} (CPU-only)", flush=True)
    print(f"Cases: {n}", flush=True)
    print(f"1. Accuracy:     {correct}/{n} = {accuracy:.1%}", flush=True)
    print(f"2. Protocol:     {protocol_ok}/{n} = {protocol_rate:.1%}", flush=True)
    print(f"3. Avg latency:  {avg_time:.1f}s  Total: {total_time:.1f}s", flush=True)
    print(flush=True)

    print("Por categoria:", flush=True)
    for cat in sorted(by_cat.keys()):
        s = by_cat[cat]
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
        print(f"  {cat:25s}: {s['correct']:2d}/{s['total']:2d} = {acc:5.1%}", flush=True)

    print(flush=True)
    print("Por relation:", flush=True)
    for rel in sorted(by_rel.keys()):
        s = by_rel[rel]
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
        print(f"  {rel:12s}: {s['correct']:2d}/{s['total']:2d} = {acc:5.1%}", flush=True)

    print(flush=True)
    print("Confusion matrix (expected -> produced):", flush=True)
    for exp in sorted(confusion.keys()):
        prods = confusion[exp]
        parts = [f"{prod}:{cnt}" for prod, cnt in sorted(prods.items())]
        print(f"  {exp:12s} -> {', '.join(parts)}", flush=True)

    print(flush=True)
    print("Errores por categoria:", flush=True)
    for cat in sorted(errors_by_cat.keys()):
        errs = errors_by_cat[cat]
        print(f"  {cat} ({len(errs)} errors):", flush=True)
        for e in errs:
            print(f"    {e['id']}: expected={e['expected']} produced={e['produced']}", flush=True)
            print(f"      claim: {e['claim']}", flush=True)

    # Reporte JSON
    report = {
        "pilot": "granite_benchmark_v2",
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": MODEL,
        "execution": "CPU-only (num_gpu=0)",
        "benchmark": str(BENCHMARK.name),
        "case_count": n,
        "overall": {
            "accuracy": round(accuracy, 4),
            "correct": correct,
            "protocol_validity": round(protocol_rate, 4),
            "avg_latency_s": round(avg_time, 2),
            "total_latency_s": round(total_time, 2),
        },
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
        "confusion_matrix": {exp: dict(prods) for exp, prods in confusion.items()},
        "errors_by_category": {cat: errs for cat, errs in errors_by_cat.items()},
        "results": results,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(flush=True)
    print(f"Report: {OUTPUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
