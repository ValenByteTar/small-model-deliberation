"""
DEPRECATED — Experimento documentado, no reejecutar sin leer PM-003.

Este script es parte del primer experimento LLMSupport con BitNet-b1.58-2B-4T
como observador pasivo generando hipotesis GOOD_EVIDENCE / RETRY_RETRIEVAL.
El experimento fallo: el modelo no seguia el formato few-shot y produjo
45 RETRY_RETRIEVAL sobre 50 casos sin razonamiento coherente.

Resultado: ADR-0031 deprecado, LLMSupport desacoplado del pipeline.

Ver:
  - knowledge/postmortems/PM-003-bitnet-semantic-capacity-insufficient.md
  - knowledge/experiments/EXP-010-bitnet-ensemble-semantic-capacity.md

---

Run 5 canonical benchmark queries with LLMSupport as passive observer.

Executes q-001 through q-005 with:
- Kernel enabled (linear controller)
- LLMSupport active (passive mode, BitNet on CPU)
- Collects both pipeline results and LLMSupport hypotheses
- Compares hypotheses against actual pipeline outcomes

Usage:
    python scripts/run_llm_support_pilot.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hybrid_rag.rag_hybrid import HybridRAG

V2_CASES = ROOT / "tests" / "eval" / "canonical" / "cybersec_eval_questions_v2.json"
OUTPUT = ROOT / "tests" / "eval" / "canonical" / "reports" / "llm_support_pilot.json"


def load_cases(limit: int = 5) -> list:
    with V2_CASES.open(encoding="utf-8") as f:
        data = json.load(f)
    return data.get("questions", [])[:limit]


def main() -> int:
    cases = load_cases(5)

    print("=" * 70)
    print("LLMSupport Pilot: 5 canonical queries with passive observer")
    print("=" * 70)

    # Patch config to enable llm_support
    import yaml
    config_path = ROOT / "config.yaml"
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["llm_support"] = {
        "enabled": True,
        "mode": "passive",
        "model_path": "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf",
        "server_path": "build/bin/Release/llama-server.exe",
        "bitnet_root": "C:/Users/Valen/Desktop/Proyectos/BitNet",
        "port": 8081,
        "threads": 8,
        "ctx_size": 2048,
        "max_hypotheses": 50,
        "max_concurrent": 2,
    }

    # Write temp config
    temp_config = ROOT / "config_llm_support_pilot.yaml"
    with temp_config.open("w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    print(f"Config temporal: {temp_config}")
    print(f"LLMSupport: enabled=True, mode=passive, model=BitNet-b1.58-2B-4T")
    print()

    # Init RAG with patched config
    rag = HybridRAG(config_path=str(temp_config), variant="bge", heuristics="balanced")
    rag.kernel_enabled = True

    results = []

    for i, case in enumerate(cases, 1):
        qid = case.get("id")
        query = case.get("query", {}).get("raw", "")
        print(f"[{i}/5] {qid}: {query}")

        t0 = time.time()
        try:
            result = rag.execute(query, top_k=10, stream=False)
            dt = time.time() - t0

            answer = result.answer or ""
            state = result.execution_state

            # Collect LLMSupport hypotheses for this run
            llm_support = getattr(rag, "_llm_support", None)
            hypotheses = []
            if llm_support:
                hyps = llm_support.get_hypotheses()
                # Filter hypotheses for this run_id
                run_hyps = [h for h in hyps if h.run_id == state.run_id]
                hypotheses = [h.to_dict() for h in run_hyps]

            entry = {
                "case_id": qid,
                "query": query,
                "answer_preview": answer[:300],
                "answer_length": len(answer),
                "run_id": state.run_id,
                "iteration": state.iteration,
                "llm_calls": state.llm_calls,
                "timing_s": round(dt, 2),
                "evidence_count": len(state.results or []),
                "signals": [s.to_dict() for s in state.signals],
                "decline": state.decline,
                "hypotheses": hypotheses,
                "hypothesis_count": len(hypotheses),
            }
            results.append(entry)

            print(f"  time={dt:.1f}s  evidence={len(state.results or [])}  signals={len(state.signals)}  hypotheses={len(hypotheses)}")
            for h in hypotheses:
                print(f"    [{h['stage']}] {h['suggestion']} (conf={h['confidence']:.2f}): {h['reasoning'][:80]}")

        except Exception as exc:
            dt = time.time() - t0
            results.append({
                "case_id": qid,
                "query": query,
                "error": str(exc),
                "timing_s": round(dt, 2),
            })
            print(f"  ERROR: {exc}")
        print()

    # Collect LLMSupport stats
    llm_support = getattr(rag, "_llm_support", None)
    stats = llm_support.stats if llm_support else {}

    # Stop LLMSupport
    if llm_support:
        llm_support.stop()

    # Shutdown BitNet server
    if hasattr(rag, "model_provider") and rag.model_provider and hasattr(rag.model_provider, "shutdown"):
        rag.model_provider.shutdown()

    # Cleanup temp config
    try:
        temp_config.unlink()
    except Exception:
        pass

    # Build report
    report = {
        "pilot": "llm_support_passive",
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "queries": 5,
        "llm_support_stats": stats,
        "results": results,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print(f"Report saved: {OUTPUT}")
    print(f"Stats: {stats}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
