"""
Micro-Coliseum Trio: BitNet + Llama3.2 + Qwen3 4B base.

Lanza 9 corridas (3 modelos x 3 modos) del microcoliseum deliberativo:
  - BitNet b1.58-2B-4T  (llama-server, CPU)
  - Llama3.2 3B         (Ollama, GPU)
  - Qwen3 4B Q4_K_M     (Ollama, GPU, num_ctx=4096)

Modos:
  - independent
  - debate-on-disagreement
  - debate-all

BitNet corre primero (CPU, sin competir con GPU). Luego Llama3.2 y
Qwen3 corren via Ollama (GPU). El orquestador arranca/detiene
llama-server para BitNet automaticamente.

Uso:
    python runners/run_microcoliseum_trio.py
    python runners/run_microcoliseum_trio.py --ollama-port 11434
    python runners/run_microcoliseum_trio.py --skip-bitnet
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import requests

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
SCRIPT = str(ROOT / "runners" / "debate.py")

# ----------------------------- Modelos -----------------------------

# BitNet via llama-server (CPU)
BITNET_ROOT = os.environ.get("BITNET_ROOT", os.path.expanduser("~/BitNet"))
BITNET_MODEL_PATH = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
BITNET_SERVER = "build/bin/Release/llama-server.exe"
BITNET_PORT = 8081
BITNET_THREADS = 4  # 1 instancia, 4 threads (secuencial, no paralelo)
BITNET_CTX = 2048
BITNET_MAX_TOKENS = 128
BITNET_LABEL = "bitnet-2b"

# Llama3.2 via Ollama (GPU)
LLAMA32_MODEL = "llama3.2:3b"
LLAMA32_LABEL = "llama32-3b"
LLAMA32_MAX_TOKENS = 64

# Qwen3 4B base via Ollama (GPU, necesita num_ctx=4096)
QWEN3_MODEL = "hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M"
QWEN3_LABEL = "qwen3-4b-base"
QWEN3_MAX_TOKENS = 64
QWEN3_NUM_CTX = 4096  # El modelo base tiene ctx=40960; sin esto OOM en GPU de 6GB

MODES = ["independent", "debate-on-disagreement", "debate-all"]

# ----------------------------- llama-server management -----------------------------

_processes: List[subprocess.Popen] = []


def start_bitnet_server() -> bool:
    """Arranca una instancia de llama-server para BitNet."""
    exe = os.path.join(BITNET_ROOT, BITNET_SERVER)
    model_abs = os.path.join(BITNET_ROOT, BITNET_MODEL_PATH)

    if not os.path.exists(exe):
        print(f"ERROR: llama-server no encontrado: {exe}", flush=True)
        return False
    if not os.path.exists(model_abs):
        print(f"ERROR: modelo no encontrado: {model_abs}", flush=True)
        return False

    url = f"http://127.0.0.1:{BITNET_PORT}"

    # Si ya esta activo, reusar
    try:
        r = requests.get(f"{url}/health", timeout=2)
        if r.status_code == 200:
            print(f"  llama-server ya activo en {url}", flush=True)
            return True
    except Exception:
        pass

    cmd = [
        exe,
        "-m", model_abs,
        "--host", "127.0.0.1",
        "--port", str(BITNET_PORT),
        "-t", str(BITNET_THREADS),
        "-c", str(BITNET_CTX),
        "-ngl", "0",
        "--temp", "0.0",
        "--override-kv", "tokenizer.ggml.pre=str:llama3",
    ]

    print(f"  Arrancando llama-server (port={BITNET_PORT}, threads={BITNET_THREADS})...", flush=True)
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
            print(f"ERROR: llama-server murio (code={proc.returncode})", flush=True)
            return False
        try:
            r = requests.get(f"{url}/health", timeout=2)
            if r.status_code == 200:
                print(f"  llama-server activo en {url} (PID={proc.pid})", flush=True)
                return True
        except Exception:
            pass
        time.sleep(1)

    print(f"ERROR: timeout esperando llama-server", flush=True)
    return False


def stop_bitnet_server() -> None:
    """Detiene todas las instancias de llama-server."""
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
    print("  llama-server detenido.", flush=True)


# ----------------------------- Runner -----------------------------

def run_debate(
    model: str,
    mode: str,
    label: str,
    backend: str = "ollama",
    base_url: str = "",
    port: int = 11434,
    gpu: bool = True,
    num_predict: int = 64,
    num_ctx: int = 0,
) -> int:
    """Lanza una corrida de debate.py con los parametros dados."""
    cmd = [PY, "-u", SCRIPT,
           "--model", model,
           "--mode", mode,
           "--label", label,
           "--num-predict", str(num_predict),
           "--backend", backend]
    if gpu:
        cmd.append("--gpu")
    if base_url:
        cmd.extend(["--base-url", base_url])
    if num_ctx > 0:
        cmd.extend(["--num-ctx", str(num_ctx)])
    if backend == "llama-server":
        cmd.extend(["--port", str(port)])

    print(f"\n{'='*70}", flush=True)
    print(f"RUN: {label} / {mode} / {backend}", flush=True)
    print(f"{'='*70}", flush=True)

    return subprocess.call(cmd)


# ----------------------------- Main -----------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Micro-Coliseum Trio: BitNet + Llama3.2 + Qwen3 4B base"
    )
    parser.add_argument("--ollama-port", type=int, default=11434,
                        help="Ollama instance port (default: 11434)")
    parser.add_argument("--skip-bitnet", action="store_true",
                        help="Skip BitNet runs (only Llama3.2 + Qwen3)")
    parser.add_argument("--skip-llama32", action="store_true",
                        help="Skip Llama3.2 runs")
    parser.add_argument("--skip-qwen3", action="store_true",
                        help="Skip Qwen3 runs")
    parser.add_argument("--modes", nargs="+", default=MODES,
                        help=f"Modes to run (default: {MODES})")
    args = parser.parse_args()

    modes = args.modes
    t0 = time.time()
    runs: List[str] = []
    errors: List[str] = []

    # ==================== Fase 1: BitNet (CPU, llama-server) ====================
    if not args.skip_bitnet:
        print("\n" + "=" * 70, flush=True)
        print("FASE 1: BitNet b1.58-2B-4T (llama-server, CPU)", flush=True)
        print("=" * 70, flush=True)

        if not start_bitnet_server():
            print("SKIP: no se pudo iniciar llama-server para BitNet", flush=True)
            errors.append("bitnet: server start failed")
        else:
            for mode in modes:
                rc = run_debate(
                    model="bitnet",  # etiqueta; llama-server no usa model name
                    mode=mode,
                    label=BITNET_LABEL,
                    backend="llama-server",
                    base_url=f"http://127.0.0.1:{BITNET_PORT}",
                    port=BITNET_PORT,
                    gpu=False,
                    num_predict=BITNET_MAX_TOKENS,
                )
                run_id = f"bitnet/{mode}"
                runs.append(run_id)
                if rc != 0:
                    errors.append(f"{run_id}: exit code {rc}")

            stop_bitnet_server()

    # ==================== Fase 2: Llama3.2 (Ollama, GPU) ====================
    if not args.skip_llama32:
        print("\n" + "=" * 70, flush=True)
        print("FASE 2: Llama3.2 3B (Ollama, GPU)", flush=True)
        print("=" * 70, flush=True)

        for mode in modes:
            rc = run_debate(
                model=LLAMA32_MODEL,
                mode=mode,
                label=LLAMA32_LABEL,
                backend="ollama",
                port=args.ollama_port,
                gpu=True,
                num_predict=LLAMA32_MAX_TOKENS,
            )
            run_id = f"llama32/{mode}"
            runs.append(run_id)
            if rc != 0:
                errors.append(f"{run_id}: exit code {rc}")

    # ==================== Fase 3: Qwen3 4B base (Ollama, GPU) ====================
    if not args.skip_qwen3:
        print("\n" + "=" * 70, flush=True)
        print("FASE 3: Qwen3 4B base Q4_K_M (Ollama, GPU, num_ctx=4096)", flush=True)
        print("=" * 70, flush=True)

        for mode in modes:
            rc = run_debate(
                model=QWEN3_MODEL,
                mode=mode,
                label=QWEN3_LABEL,
                backend="ollama",
                port=args.ollama_port,
                gpu=True,
                num_predict=QWEN3_MAX_TOKENS,
                num_ctx=QWEN3_NUM_CTX,
            )
            run_id = f"qwen3-4b-base/{mode}"
            runs.append(run_id)
            if rc != 0:
                errors.append(f"{run_id}: exit code {rc}")

    # ==================== Resumen ====================
    wall = time.time() - t0
    print("\n" + "=" * 70, flush=True)
    print("MICRO-COLISEUM TRIO - RESUMEN", flush=True)
    print("=" * 70, flush=True)
    print(f"  Total runs: {len(runs)}", flush=True)
    print(f"  Errors: {len(errors)}", flush=True)
    if errors:
        for e in errors:
            print(f"    - {e}", flush=True)
    print(f"  Wall time: {wall/60:.1f} min", flush=True)
    print(f"  Output: results/raw/microcoliseum_*.json + .md", flush=True)
    print("=" * 70, flush=True)

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
