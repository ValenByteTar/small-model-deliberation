"""
Gestor de instancias Ollama multi-puerto.

Permite levantar multiples instancias de Ollama en puertos diferentes,
cada una con su propio modelo cargado y pinneado (OLLAMA_MAX_LOADED_MODELS=1),
eliminando el model-thrashing cuando multiples experimentos corren en paralelo.

Arquitectura:

    Instancia A (puerto 11434, GPU)  ← modelo X pinneado
    Instancia B (puerto 11435, CPU)  ← modelo Y pinneado
    Instancia C (puerto 11436, GPU)  ← modelo Z pinneado (si VRAM permite)

Cada instancia:
    - Su propio proceso ollama.exe serve
    - Su propio OLLAMA_HOST (puerto)
    - OLLAMA_MAX_LOADED_MODELS=1 (sin thrashing dentro de la instancia)
    - Modelo pre-cargado via keep_alive=24h
    - Comparte el store de modelos (~/.ollama/models) — lectura simultanea OK

Uso:

    # Lanzar instancia CPU en puerto 11435 con llama3.2:3b
    python runners/ollama_instances.py start --port 11435 --cpu --model llama3.2:3b

    # Lanzar instancia GPU en puerto 11436 con gemma3:4b-it-q4_K_M
    python runners/ollama_instances.py start --port 11436 --gpu --model gemma3:4b-it-q4_K_M

    # Ver estado de todas las instancias gestionadas
    python runners/ollama_instances.py status

    # Detener instancia en puerto 11435
    python runners/ollama_instances.py stop --port 11435

    # Detener todas las instancias gestionadas
    python runners/ollama_instances.py stop --all

Notas:
    - El puerto 11434 es la instancia por defecto de Ollama (no gestionada
      por este script a menos que se arranque explicitamente).
    - En Windows, ollama.exe serve es el proceso servidor. El tray app
      (ollama app.exe) gestiona la instancia por defecto pero no las
      adicionales; este script las gestiona directamente.
    - El estado se persiste en runners/.ollama-instances.json para
      sobrevivir entre invocaciones del script.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# ......................... Configuracion ..........................

ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = ROOT / "runners" / ".ollama-instances.json"
DEFAULT_BASE_PORT = 11434
HEALTH_TIMEOUT = 2.0
STARTUP_TIMEOUT = 60.0  # segundos max para que la instancia responda
PRELOAD_TIMEOUT = 120.0  # segundos max para cargar el modelo


# ......................... Estado ..........................

@dataclass
class InstanceRecord:
    """Registro de una instancia gestionada por este script."""
    port: int
    host: str  # "127.0.0.1"
    num_gpu: int  # 0=CPU, 99=GPU
    model: str  # modelo pinneado
    pid: int  # PID del proceso ollama.exe serve
    started_at: str  # timestamp ISO


def _load_state() -> List[InstanceRecord]:
    if not STATE_FILE.exists():
        return []
    try:
        with STATE_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return [InstanceRecord(**r) for r in data.get("instances", [])]
    except Exception:
        return []


def _save_state(instances: List[InstanceRecord]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with STATE_FILE.open("w", encoding="utf-8") as f:
        json.dump(
            {"instances": [asdict(r) for r in instances]},
            f, ensure_ascii=False, indent=2,
        )


# ......................... Health checks ..........................

def _base_url(port: int, host: str = "127.0.0.1") -> str:
    return f"http://{host}:{port}"


def _is_port_listening(port: int, host: str = "127.0.0.1") -> bool:
    """True si hay algo escuchando en el puerto."""
    try:
        r = requests.get(f"{_base_url(port, host)}/api/tags", timeout=HEALTH_TIMEOUT)
        return r.status_code == 200
    except Exception:
        return False


def _get_loaded_models(port: int, host: str = "127.0.0.1") -> List[str]:
    """Lista los modelos disponibles en la instancia."""
    try:
        r = requests.get(f"{_base_url(port, host)}/api/ps", timeout=HEALTH_TIMEOUT)
        if r.status_code == 200:
            return [m.get("name", "") for m in r.json().get("models", [])]
    except Exception:
        pass
    return []


def _is_process_alive(pid: int) -> bool:
    try:
        proc = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
            capture_output=True, text=True, timeout=5,
        )
        return str(pid) in proc.stdout
    except Exception:
        return False


# ......................... Lanzar / Detener ..........................

def _build_env(num_gpu: int) -> Dict[str, str]:
    """Construye el entorno para la nueva instancia de Ollama.

    Hereda el entorno actual pero sobreescribe:
      - OLLAMA_HOST  → lo setea el caller via parametro
      - OLLAMA_MAX_LOADED_MODELS = 1  (pinnea 1 modelo, sin thrashing)
      - OLLAMA_NUM_GPU = num_gpu (0=CPU, 99=GPU)
    """
    env = dict(os.environ)
    env["OLLAMA_MAX_LOADED_MODELS"] = "1"
    env["OLLAMA_NUM_GPU"] = str(num_gpu)
    # Mantener OLLAMA_FLASH_ATTENTION, OLLAMA_NUM_THREAD, etc. del entorno actual
    return env


def start_instance(
    port: int,
    num_gpu: int,
    model: str,
    host: str = "127.0.0.1",
) -> InstanceRecord:
    """Lanza una instancia de Ollama en el puerto indicado y pre-carga el modelo.

    Raises:
        RuntimeError: si el puerto ya esta en uso o el proceso no arranca.
    """
    # 1. Verificar que el puerto no este ocupado
    if _is_port_listening(port, host):
        # Podria ser una instancia ya gestionada
        existing = _load_state()
        for r in existing:
            if r.port == port:
                raise RuntimeError(
                    f"Puerto {port} ya tiene una instancia gestionada "
                    f"(model={r.model}, pid={r.pid}). Usa 'stop --port {port}' primero."
                )
        raise RuntimeError(
            f"Puerto {port} ya esta en uso por otro proceso. "
            f"Usa otro puerto o detén el proceso existente."
        )

    # 2. Localizar ollama.exe
    ollama_exe = shutil_which("ollama")
    if not ollama_exe:
        raise RuntimeError("ollama.exe no encontrado en PATH")

    # 3. Construir entorno
    env = _build_env(num_gpu)
    env["OLLAMA_HOST"] = f"{host}:{port}"

    # 4. Lanzar ollama serve
    print(f"  Lanzando ollama serve en {host}:{port} "
          f"(num_gpu={num_gpu}, model={model})...", flush=True)

    proc = subprocess.Popen(
        [ollama_exe, "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )

    # 5. Esperar a que la instancia responda
    print(f"  Esperando health-check...", flush=True)
    t0 = time.time()
    while time.time() - t0 < STARTUP_TIMEOUT:
        if _is_port_listening(port, host):
            break
        if proc.poll() is not None:
            raise RuntimeError(
                f"Proceso ollama serve murio prematuramente (exit code={proc.returncode})"
            )
        time.sleep(1)
    else:
        proc.kill()
        raise RuntimeError(
            f"Timeout: la instancia no respondio en {STARTUP_TIMEOUT}s"
        )

    print(f"  Instancia activa en {host}:{port} (PID={proc.pid})", flush=True)

    # 6. Pre-cargar el modelo (keep_alive largo para pinnearlo)
    print(f"  Pre-cargando modelo {model}...", flush=True)
    try:
        r = requests.post(
            f"{_base_url(port, host)}/api/generate",
            json={"model": model, "prompt": "", "stream": False, "keep_alive": "24h"},
            timeout=PRELOAD_TIMEOUT,
        )
        r.raise_for_status()
        print(f"  Modelo {model} cargado y pinneado (keep_alive=24h)", flush=True)
    except Exception as e:
        print(f"  WARNING: No se pudo pre-cargar el modelo: {e}", flush=True)
        print(f"  La instancia esta activa pero sin modelo pre-cargado.", flush=True)

    # 7. Registrar la instancia
    record = InstanceRecord(
        port=port,
        host=host,
        num_gpu=num_gpu,
        model=model,
        pid=proc.pid,
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    instances = _load_state()
    instances.append(record)
    _save_state(instances)

    return record


def stop_instance(port: int) -> bool:
    """Detiene una instancia gestionada. Retorna True si se detuvo."""
    instances = _load_state()
    target = None
    for r in instances:
        if r.port == port:
            target = r
            break

    if target is None:
        print(f"  No hay instancia gestionada en puerto {port}", flush=True)
        # Intentar detener aunque no este registrada
        if _is_port_listening(port):
            print(f"  (Hay algo escuchando en {port} pero no es gestionada por este script)", flush=True)
        return False

    # Descargar el modelo primero
    try:
        requests.post(
            f"{_base_url(target.port, target.host)}/api/generate",
            json={"model": target.model, "keep_alive": 0},
            timeout=10,
        )
        print(f"  Modelo {target.model} descargado", flush=True)
    except Exception:
        pass

    # Matar el proceso
    if _is_process_alive(target.pid):
        try:
            subprocess.run(
                ["taskkill", "/PID", str(target.pid), "/F"],
                capture_output=True, timeout=10,
            )
            print(f"  Proceso PID={target.pid} terminado", flush=True)
        except Exception as e:
            print(f"  ERROR al terminar PID={target.pid}: {e}", flush=True)
    else:
        print(f"  Proceso PID={target.pid} ya no estaba vivo", flush=True)

    # Eliminar del estado
    instances = [r for r in instances if r.port != port]
    _save_state(instances)

    return True


def stop_all() -> int:
    """Detiene todas las instancias gestionadas. Retorna cuantas se detuvieron."""
    instances = _load_state()
    count = 0
    for r in instances:
        print(f"  Deteniendo instancia puerto {r.port} (model={r.model})...", flush=True)
        if stop_instance(r.port):
            count += 1
    return count


# ......................... Status ..........................

def print_status() -> None:
    """Imprime el estado de todas las instancias (gestionadas + default)."""
    instances = _load_state()

    print("=" * 70, flush=True)
    print("ESTADO DE INSTANCIAS OLLAMA", flush=True)
    print("=" * 70, flush=True)

    # Instancia por defecto (11434)
    default_up = _is_port_listening(DEFAULT_BASE_PORT)
    if default_up:
        loaded = _get_loaded_models(DEFAULT_BASE_PORT)
        print(f"  [DEFAULT] :{DEFAULT_BASE_PORT}  ACTIVA  modelos_cargados={loaded or '—'}", flush=True)
    else:
        print(f"  [DEFAULT] :{DEFAULT_BASE_PORT}  INACTIVA", flush=True)

    # Instancias gestionadas
    if not instances:
        print(flush=True)
        print("  (Sin instancias gestionadas por este script)", flush=True)
    else:
        print(flush=True)
        print("  Instancias gestionadas:", flush=True)
        for r in instances:
            alive = _is_process_alive(r.pid)
            up = _is_port_listening(r.port, r.host)
            loaded = _get_loaded_models(r.port, r.host) if up else []
            status = "ACTIVA" if (alive and up) else "DEAD"
            gpu_tag = "GPU" if r.num_gpu > 0 else "CPU"
            print(
                f"    :{r.port}  {status:6s}  {gpu_tag:3s}  "
                f"model={r.model}  pid={r.pid}  "
                f"cargados={loaded or '—'}",
                flush=True,
            )

    print("=" * 70, flush=True)


# ......................... Utils ..........................

def shutil_which(name: str) -> Optional[str]:
    """Localiza un ejecutable en PATH."""
    try:
        import shutil
        return shutil.which(name)
    except Exception:
        return None


# ......................... CLI ..........................

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gestor de instancias Ollama multi-puerto",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # start
    p_start = sub.add_parser("start", help="Lanzar nueva instancia")
    p_start.add_argument("--port", type=int, required=True, help="Puerto (ej: 11435)")
    p_start.add_argument("--model", required=True, help="Modelo a pinnear (ej: llama3.2:3b)")
    p_start.add_argument("--cpu", action="store_true", help="CPU-only (num_gpu=0)")
    p_start.add_argument("--gpu", action="store_true", help="GPU (num_gpu=99)")
    p_start.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")

    # stop
    p_stop = sub.add_parser("stop", help="Detener instancia")
    p_stop.add_argument("--port", type=int, help="Puerto de la instancia a detener")
    p_stop.add_argument("--all", action="store_true", help="Detener todas las gestionadas")

    # status
    sub.add_parser("status", help="Mostrar estado de instancias")

    args = parser.parse_args()

    if args.command == "status":
        print_status()
        return 0

    if args.command == "start":
        if args.cpu and args.gpu:
            print("ERROR: --cpu y --gpu son mutuamente excluyentes", flush=True)
            return 1
        if not args.cpu and not args.gpu:
            print("ERROR: debe especificar --cpu o --gpu", flush=True)
            return 1
        num_gpu = 0 if args.cpu else 99
        try:
            rec = start_instance(
                port=args.port,
                num_gpu=num_gpu,
                model=args.model,
                host=args.host,
            )
            print(flush=True)
            print(f"OK: instancia {rec.host}:{rec.port} lista (model={rec.model})", flush=True)
            print(f"    Usar en runners: --port {rec.port}", flush=True)
            return 0
        except RuntimeError as e:
            print(f"ERROR: {e}", flush=True)
            return 1

    if args.command == "stop":
        if args.all:
            n = stop_all()
            print(flush=True)
            print(f"Detenidas {n} instancias", flush=True)
            return 0
        if not args.port:
            print("ERROR: usar --port N o --all", flush=True)
            return 1
        ok = stop_instance(args.port)
        return 0 if ok else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
