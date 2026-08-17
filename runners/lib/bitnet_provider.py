"""
BitNetModelProvider — ModelProvider sobre llama-server en CPU (ADR-0007, ADR-0031).

Implementa el contrato ModelProvider (ADR-0007) usando llama-server.exe
del entorno bitnet.cpp como backend HTTP. El modelo corre en CPU sin
competir por GPU con el pipeline principal (RES-004 §8.1).

Requisitos:
- bitnet.cpp compilado (build/bin/Release/llama-server.exe)
- modelo GGUF i2_s descargado (ej. BitNet-b1.58-2B-4T)

Uso:
    provider = BitNetModelProvider(
        model_path="models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf",
        server_path="build/bin/Release/llama-server.exe",
        bitnet_root="~/BitNet",  # or set BITNET_ROOT env var
    )
    provider.ensure_running()  # inicia el servidor si no está activo
    response = provider.generate("Prompt here")
"""

from __future__ import annotations

import atexit
import logging
import os
import subprocess
import time
from typing import Any, Dict, Iterator, Optional

import requests

logger = logging.getLogger(__name__)


class BitNetModelProvider:
    """ModelProvider sobre llama-server (bitnet.cpp) en CPU (ADR-0007, ADR-0031)."""

    name = "bitnet"

    def __init__(
        self,
        model_path: str,
        server_path: str = "build/bin/Release/llama-server.exe",
        bitnet_root: str = ".",
        host: str = "127.0.0.1",
        port: int = 8081,
        threads: int = 4,
        ctx_size: int = 2048,
        num_gpu: int = 0,
        default_options: Optional[Dict[str, Any]] = None,
        auto_start: bool = True,
    ) -> None:
        self.model = os.path.basename(model_path)
        self.model_path = model_path
        self.server_path = server_path
        self.bitnet_root = bitnet_root
        self.host = host
        self.port = port
        self.threads = threads
        self.ctx_size = ctx_size
        self.num_gpu = num_gpu
        self.default_options = default_options or {}
        self.auto_start = auto_start
        self._process: Optional[subprocess.Popen] = None
        self._base_url = f"http://{host}:{port}"

    def ensure_running(self, timeout: float = 30.0) -> bool:
        """Inicia llama-server si no está activo. Returns True si está listo."""
        if self._is_server_up():
            return True
        if not self.auto_start:
            return False
        return self._start_server(timeout=timeout)

    def _is_server_up(self) -> bool:
        try:
            r = requests.get(f"{self._base_url}/health", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def _start_server(self, timeout: float = 30.0) -> bool:
        exe = os.path.join(self.bitnet_root, self.server_path)
        model_abs = os.path.join(self.bitnet_root, self.model_path)
        if not os.path.exists(exe):
            logger.error("llama-server no encontrado: %s", exe)
            return False
        if not os.path.exists(model_abs):
            logger.error("modelo no encontrado: %s", model_abs)
            return False

        cmd = [
            exe,
            "-m", model_abs,
            "--host", self.host,
            "--port", str(self.port),
            "-t", str(self.threads),
            "-c", str(self.ctx_size),
            "-ngl", str(self.num_gpu),
            "--temp", "0.8",
            "--override-kv", "tokenizer.ggml.pre=str:llama3",
        ]
        logger.info("Iniciando llama-server: %s", " ".join(cmd[:4]) + " ...")
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            atexit.register(self.shutdown)
        except Exception as exc:
            logger.error("Error iniciando llama-server: %s", exc)
            return False

        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._process.poll() is not None:
                logger.error("llama-server termino prematuramente (code=%d)", self._process.returncode)
                return False
            if self._is_server_up():
                logger.info("llama-server activo en %s", self._base_url)
                return True
            time.sleep(0.5)

        logger.error("Timeout esperando llama-server (%.1fs)", timeout)
        self.shutdown()
        return False

    def shutdown(self) -> None:
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None

    def is_available(self) -> bool:
        return self._is_server_up()

    def generate(
        self,
        prompt: str,
        *,
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> str:
        if not self._is_server_up() and not self.ensure_running():
            logger.warning("llama-server no disponible, retornando vacio")
            return ""

        payload: Dict[str, Any] = {
            "prompt": prompt,
            "stream": False,
            "temperature": options.get("temperature", 0.3) if options else 0.3,
            "max_tokens": options.get("num_predict", 256) if options else 256,
            "repeat_penalty": options.get("repeat_penalty", 1.3) if options else 1.3,
            "repeat_last_n": options.get("repeat_last_n", 256) if options else 256,
        }
        try:
            r = requests.post(
                f"{self._base_url}/completion",
                json=payload,
                timeout=timeout or 60,
            )
            r.raise_for_status()
            return (r.json().get("content") or "").strip()
        except Exception as exc:
            logger.error("Error en generate: %s", exc)
            return ""

    def stream(
        self,
        prompt: str,
        *,
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Iterator[str]:
        if not self._is_server_up() and not self.ensure_running():
            return

        payload: Dict[str, Any] = {
            "prompt": prompt,
            "stream": True,
            "temperature": options.get("temperature", 0.8) if options else 0.8,
            "max_tokens": options.get("num_predict", 256) if options else 256,
        }
        try:
            with requests.post(
                f"{self._base_url}/completion",
                json=payload,
                stream=True,
                timeout=timeout or 120,
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        import json
                        data = json.loads(line)
                        chunk = data.get("content") or ""
                        if chunk:
                            yield chunk
                    except Exception:
                        continue
        except Exception as exc:
            logger.error("Error en stream: %s", exc)
            return
