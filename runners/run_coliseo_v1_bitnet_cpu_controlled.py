"""
Coliseo v1 - BitNet b1.58-2B-4T (CPU, protocolo corregido).

Motivacion:
  BitNet fue condenado en PM-003/EXP-010 basado en:
    1. semantic_pilot: 33.3% accuracy (12 casos, max_tokens=256, NO truncado)
    2. ensemble_pilot: 41.7% ensemble / 50% best single (12 casos,
       num_predict=10, SI truncado)

  Pero BitNet nunca paso por el Coliseo v1 (55 casos, benchmark v2,
  10 categorias diagnosticas). Y el experimento que mas peso tuvo en
  la condena (EXP-010 ensemble) estuvo contaminado por num_predict=10.

  Este runner prueba BitNet en el benchmark v2 completo con protocolo
  corregido:
    - max_tokens=128 (presupuesto ampliado)
    - temperature=0.0
    - prompt con instruccion JSON explicita (llama-server no soporta
      format=json_schema nativamente)
    - parser estricto (no defaultea a UNRELATED)
    - raw conservado (respuesta cruda + tokens generados)

  Usa llama-server directamente (no Ollama) porque el formato i2_s de
  BitNet no es compatible con Ollama.

  No sobrescribe resultados crudos previos: escribe a
  results/raw/coliseo_v1_bitnet_cpu_controlled.json.

Uso:
    python runners/run_coliseo_v1_bitnet_cpu_controlled.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hybrid_rag.kernel.semantic_ensemble import WORKER_PROMPTS
from hybrid_rag.kernel.state import SEMANTIC_RELATIONS


# ----------------------------- Configuracion -----------------------------

BITNET_ROOT = os.environ.get("BITNET_ROOT", os.path.expanduser("~/BitNet"))
MODEL_PATH = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
SERVER_PATH = "build/bin/Release/llama-server.exe"

BENCHMARK = ROOT / "benchmarks" / "semantic_assessment_v2.json"
OUTPUT = ROOT / "results" / "raw" / "coliseo_v1_bitnet_cpu_controlled.json"

BASE_PORT = 8081
N_INSTANCES = 4  # 4 instancias paralelas, 1 thread c/u (patron ensemble_pilot original)
MAX_TOKENS = 128
TEMPERATURE = 0.0
THREADS_PER_INSTANCE = 1
CTX_SIZE = 2048
TIMEOUT_S = 120.0

CONFIGS = [
    {"label": "single",     "roles": ["neutral"]},
    {"label": "ensemble_2", "roles": ["entailment", "skeptical"]},
    {"label": "ensemble_4", "roles": ["entailment", "skeptical", "contradiction", "neutral"]},
]


# ----------------------------- llama-server management -----------------------------

_processes: List[Any] = []  # PIDs de las instancias activas


def _instance_url(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def _start_instance(port: int) -> bool:
    """Arranca una instancia de llama-server en el puerto dado."""
    import os
    import subprocess

    exe = os.path.join(BITNET_ROOT, SERVER_PATH)
    model_abs = os.path.join(BITNET_ROOT, MODEL_PATH)

    if not os.path.exists(exe):
        print(f"ERROR: llama-server no encontrado: {exe}", flush=True)
        return False
    if not os.path.exists(model_abs):
        print(f"ERROR: modelo no encontrado: {model_abs}", flush=True)
        return False

    url = _instance_url(port)

    # Si ya esta activo, reusar
    try:
        r = requests.get(f"{url}/health", timeout=2)
        if r.status_code == 200:
            print(f"  instancia ya activa en {url}", flush=True)
            return True
    except Exception:
        pass

    cmd = [
        exe,
        "-m", model_abs,
        "--host", "127.0.0.1",
        "--port", str(port),
        "-t", str(THREADS_PER_INSTANCE),
        "-c", str(CTX_SIZE),
        "-ngl", "0",
        "--temp", str(TEMPERATURE),
        "--override-kv", "tokenizer.ggml.pre=str:llama3",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    _processes.append(proc)

    t0 = time.time()
    while time.time() - t0 < 60:
        if proc.poll() is not None:
            print(f"ERROR: instancia puerto {port} murio (code={proc.returncode})", flush=True)
            return False
        try:
            r = requests.get(f"{url}/health", timeout=2)
            if r.status_code == 200:
                print(f"  instancia activa en {url} (PID={proc.pid})", flush=True)
                return True
        except Exception:
            pass
        time.sleep(1)

    print(f"ERROR: timeout esperando instancia puerto {port}", flush=True)
    return False


def ensure_servers_running(n: int) -> bool:
    """Arranca n instancias de llama-server en puertos consecutivos (BASE_PORT..+n-1)."""
    print(f"  Arrancando {n} instancia(s) de llama-server (1 thread c/u)...", flush=True)
    for i in range(n):
        port = BASE_PORT + i
        if not _start_instance(port):
            return False
    return True


def shutdown_all() -> None:
    """Termina todas las instancias activas."""
    for proc in _processes:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    _processes.clear()


# ----------------------------- Provider -----------------------------

def generate(prompt: str, base_url: str) -> Dict[str, Any]:
    """Llama a una instancia de llama-server con presupuesto ampliado.

    Devuelve un dict con: raw, tokens_predicted, latency_s, ok.
    No lanza; captura errores y los devuelve como ok=False.
    """
    t0 = time.time()
    try:
        resp = requests.post(
            f"{base_url}/completion",
            json={
                "prompt": prompt,
                "stream": False,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "repeat_penalty": 1.1,
            },
            timeout=TIMEOUT_S,
        )
        resp.raise_for_status()
        data = resp.json()
        latency = time.time() - t0
        return {
            "raw": (data.get("content") or "").strip(),
            "tokens_predicted": data.get("tokens_predicted", 0),
            "latency_s": latency,
            "ok": True,
            "error": None,
        }
    except Exception as exc:
        return {
            "raw": "",
            "tokens_predicted": 0,
            "latency_s": time.time() - t0,
            "ok": False,
            "error": str(exc),
        }


# ----------------------------- Parser estricto -----------------------------

def parse_strict(gen: Dict[str, Any]) -> Tuple[str, float, bool, str]:
    """Parser estricto. No defaultea a UNRELATED.

    Returns: (relation, confidence, valid_json, parse_note)
    """
    if not gen["ok"]:
        return "PROTOCOL_ERROR", 0.0, False, f"request_error: {gen['error']}"

    raw = gen["raw"]
    if not raw:
        return "PROTOCOL_ERROR", 0.0, False, "empty_response"

    # Intentar parse JSON primero
    text = raw.strip()
    try:
        obj = json.loads(text)
        relation = str(obj.get("relation", "")).strip().upper()
        confidence = obj.get("confidence", 0.5)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.5
        if relation in SEMANTIC_RELATIONS:
            return relation, confidence, True, "ok"
        return "PROTOCOL_ERROR", 0.0, False, f"invalid_relation_json: {relation!r}"
    except json.JSONDecodeError:
        pass

    # Buscar JSON embebido
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start:end+1])
            relation = str(obj.get("relation", "")).strip().upper()
            confidence = float(obj.get("confidence", 0.5))
            if relation in SEMANTIC_RELATIONS:
                return relation, confidence, False, "embedded_json_recovered"
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # Buscar relacion literal en el texto
    up = raw.upper()
    for rel in SEMANTIC_RELATIONS:
        if rel in up:
            return rel, 0.3, False, "literal_recovered"

    return "PROTOCOL_ERROR", 0.0, False, "no_relation_found"


# ----------------------------- Ejecucion -----------------------------

def run_single(cases: List[dict]) -> List[dict]:
    """Config single: un worker neutral por caso, instancia 0."""
    url = _instance_url(BASE_PORT)
    results = []
    for case in cases:
        prompt = WORKER_PROMPTS["neutral"](case["claim"], case["evidence"])
        gen = generate(prompt, url)
        relation, confidence, valid_json, note = parse_strict(gen)
        results.append({
            "id": case["id"],
            "category": case["category"],
            "expected": case["expected"],
            "produced": relation,
            "correct": relation == case["expected"],
            "valid_json": valid_json,
            "confidence": round(confidence, 3),
            "latency_s": round(gen["latency_s"], 2),
            "tokens_predicted": gen["tokens_predicted"],
            "raw": gen["raw"][:500],
            "parse_note": note,
        })
    return results


def run_ensemble(cases: List[dict], roles: List[str], port_offset: int = 0) -> List[dict]:
    """Config ensemble: un worker por rol en paralelo (1 instancia c/u), agregacion por mayoria ponderada.

    port_offset: indice de la primera instancia a usar (0=8081, 1=8082, etc.)
    """
    from concurrent.futures import ThreadPoolExecutor

    # Cada worker usa una instancia dedicada (puerto distinto)
    worker_urls = [_instance_url(BASE_PORT + port_offset + i) for i in range(len(roles))]

    def _run_worker(args: Tuple[str, str, dict, str]) -> dict:
        role, url, case, _ = args
        prompt = WORKER_PROMPTS[role](case["claim"], case["evidence"])
        gen = generate(prompt, url)
        relation, confidence, valid_json, note = parse_strict(gen)
        return {
            "role": role,
            "relation": relation,
            "confidence": round(confidence, 3),
            "valid_json": valid_json,
            "latency_s": round(gen["latency_s"], 2),
            "tokens_predicted": gen["tokens_predicted"],
            "raw": gen["raw"][:500],
            "parse_note": note,
        }

    results = []
    for case in cases:
        task_args = [(role, worker_urls[i], case, "") for i, role in enumerate(roles)]
        with ThreadPoolExecutor(max_workers=len(roles)) as pool:
            votes = list(pool.map(_run_worker, task_args))

        # Ordenar votos por orden de roles original (pool.map preserva orden)
        valid_votes = [v for v in votes if v["relation"] in SEMANTIC_RELATIONS]
        if not valid_votes:
            produced = "PROTOCOL_ERROR"
            agreement = 0.0
        else:
            weights: Dict[str, float] = {}
            counts: Dict[str, int] = {}
            for v in valid_votes:
                weights[v["relation"]] = weights.get(v["relation"], 0.0) + v["confidence"]
                counts[v["relation"]] = counts.get(v["relation"], 0) + 1
            produced = max(weights, key=weights.get)
            agreement = counts[produced] / len(valid_votes)

        total_lat = sum(v["latency_s"] for v in votes)
        results.append({
            "id": case["id"],
            "category": case["category"],
            "expected": case["expected"],
            "produced": produced,
            "correct": produced == case["expected"],
            "agreement": round(agreement, 3),
            "latency_s": round(total_lat, 2),
            "votes": votes,
        })
    return results


def compute_metrics(results: List[dict]) -> dict:
    n = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    protocol_ok = sum(1 for r in results if r.get("produced") in SEMANTIC_RELATIONS)
    latencies = [r.get("latency_s", 0.0) for r in results]
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0

    by_cat = defaultdict(lambda: {"total": 0, "correct": 0})
    by_rel = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        by_cat[r["category"]]["total"] += 1
        by_rel[r["expected"]]["total"] += 1
        if r["correct"]:
            by_cat[r["category"]]["correct"] += 1
            by_rel[r["expected"]]["correct"] += 1

    errors = [r for r in results if not r["correct"]]

    return {
        "n": n,
        "accuracy": round(correct / n, 4) if n > 0 else 0.0,
        "correct": correct,
        "protocol_validity": round(protocol_ok / n, 4) if n > 0 else 0.0,
        "avg_latency_s": round(avg_lat, 2),
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
        "errors": [{"id": e["id"], "category": e["category"],
                    "expected": e["expected"], "produced": e["produced"]}
                   for e in errors],
    }


# ----------------------------- Main -----------------------------

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Coliseo v1 - BitNet b1.58-2B-4T (CPU, protocolo corregido)"
    )
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("COLOSEO v1 - BITNET b1.58-2B-4T (CPU, protocolo corregido)", flush=True)
    print(f"Protocolo: max_tokens={MAX_TOKENS}, temp={TEMPERATURE}, parser estricto", flush=True)
    print(f"Instancias: {N_INSTANCES} paralelas (1 thread c/u, puertos {BASE_PORT}-{BASE_PORT+N_INSTANCES-1})", flush=True)
    print("=" * 70, flush=True)

    # Arrancar N_INSTancias de llama-server
    if not ensure_servers_running(N_INSTANCES):
        shutdown_all()
        return 1

    with BENCHMARK.open("r", encoding="utf-8") as f:
        cases = json.load(f)["cases"]
    n = len(cases)
    print(f"Cases: {n} | Model: BitNet-b1.58-2B-4T | CPU", flush=True)
    print(f"Configs: {[c['label'] for c in CONFIGS]}", flush=True)
    print(flush=True)

    all_configs: Dict[str, Any] = {}

    # ==================== Fase 1: single + ensemble_2 en paralelo ====================
    # single usa instancia 0 (puerto 8081), ensemble_2 usa instancias 1-2 (8082-8083)
    # Instancia 3 (8084) idle en esta fase
    print("=" * 70, flush=True)
    print("FASE 1: single + ensemble_2 en paralelo (3 instancias)", flush=True)
    print("=" * 70, flush=True)

    from concurrent.futures import ThreadPoolExecutor

    t_phase1 = time.time()

    def _run_single_wrapper() -> Tuple[str, List[dict], float]:
        t0 = time.time()
        r = run_single(cases)
        return "single", r, time.time() - t0

    def _run_ensemble2_wrapper() -> Tuple[str, List[dict], float]:
        t0 = time.time()
        r = run_ensemble(cases, ["entailment", "skeptical"], port_offset=1)
        return "ensemble_2", r, time.time() - t0

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_run_single_wrapper), pool.submit(_run_ensemble2_wrapper)]
        for fut in futures:
            clabel, results, wall = fut.result()
            roles = [c["roles"] for c in CONFIGS if c["label"] == clabel][0]
            metrics = compute_metrics(results)
            all_configs[clabel] = {
                "roles": roles,
                "metrics": metrics,
                "wall_time_s": round(wall, 1),
                "per_case": results,
            }
            print(f"  {clabel}: accuracy={metrics['accuracy']:.1%} ({metrics['correct']}/{n}) "
                  f"protocol={metrics['protocol_validity']:.0%} "
                  f"avg_lat={metrics['avg_latency_s']:.1f}s wall={wall:.0f}s", flush=True)
            if metrics["correct"] < n:
                print(f"    Errores por categoria:", flush=True)
                for cat, s in sorted(metrics["by_category"].items()):
                    if s["correct"] < s["total"]:
                        print(f"      {cat:25s}: {s['correct']}/{s['total']}", flush=True)

    print(f"  Fase 1 total: {time.time() - t_phase1:.0f}s", flush=True)
    print(flush=True)

    # ==================== Fase 2: ensemble_4 solo (4 instancias) ====================
    print("=" * 70, flush=True)
    print("FASE 2: ensemble_4 (4 instancias en paralelo)", flush=True)
    print("=" * 70, flush=True)

    t0 = time.time()
    results = run_ensemble(cases, ["entailment", "skeptical", "contradiction", "neutral"])
    wall = time.time() - t0
    metrics = compute_metrics(results)
    all_configs["ensemble_4"] = {
        "roles": ["entailment", "skeptical", "contradiction", "neutral"],
        "metrics": metrics,
        "wall_time_s": round(wall, 1),
        "per_case": results,
    }
    print(f"  ensemble_4: accuracy={metrics['accuracy']:.1%} ({metrics['correct']}/{n}) "
          f"protocol={metrics['protocol_validity']:.0%} "
          f"avg_lat={metrics['avg_latency_s']:.1f}s wall={wall:.0f}s", flush=True)
    if metrics["correct"] < n:
        print(f"    Errores por categoria:", flush=True)
        for cat, s in sorted(metrics["by_category"].items()):
            if s["correct"] < s["total"]:
                print(f"      {cat:25s}: {s['correct']}/{s['total']}", flush=True)
    print(flush=True)

    # Comparacion vs historico
    print("=" * 70, flush=True)
    print("COMPARACION vs HISTORICO (12 casos, num_predict=10)", flush=True)
    print("=" * 70, flush=True)
    print(f"  {'config':<12s} | {'historico':>10s} | {'controlado':>11s} | {'delta':>8s}", flush=True)
    print("  " + "-" * 52, flush=True)
    historico = {"single": 0.333, "ensemble_2": 0.333, "ensemble_4": 0.417}
    for cfg in CONFIGS:
        cl = cfg["label"]
        hist = historico.get(cl, 0.0)
        ctrl = all_configs[cl]["metrics"]["accuracy"]
        delta = ctrl - hist
        print(f"  {cl:<12s} | {hist:>9.1%} | {ctrl:>10.1%} | {delta:>+7.1%}", flush=True)

    print(flush=True)
    print("Nota: el historico fue sobre 12 casos (no 55).", flush=True)
    print("El single historico (33.3%) NO fue truncado (max_tokens=256).", flush=True)
    print("El ensemble historico (41.7%) SI fue truncado (num_predict=10).", flush=True)

    # Reporte JSON
    report = {
        "pilot": "coliseo_v1_bitnet_cpu_controlled",
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmark": str(BENCHMARK.name),
        "case_count": n,
        "model": {"label": "bitnet-2b-ternary", "name": "BitNet-b1.58-2B-4T"},
        "execution": f"CPU (llama-server, num_gpu=0, {N_INSTANCES} instancias paralelas)",
        "protocol": {
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "threads_per_instance": THREADS_PER_INSTANCE,
            "n_instances": N_INSTANCES,
            "ports": [BASE_PORT + i for i in range(N_INSTANCES)],
            "ctx_size": CTX_SIZE,
            "parser": "strict_no_default",
            "note": "llama-server no soporta think=false ni format=json_schema; "
                    "JSON via prompt instruction + strict parser; "
                    "ensemble workers corren en paralelo (1 instancia c/u)",
        },
        "configs": [{"label": c["label"], "roles": c["roles"]} for c in CONFIGS],
        "results": {cl: all_configs[cl] for cl in all_configs},
        "historical_baseline_12cases": historico,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(flush=True)
    print(f"Report: {OUTPUT}", flush=True)

    # Shutdown instancias
    print(flush=True)
    print("Apagando instancias de llama-server...", flush=True)
    shutdown_all()
    print("  done", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
