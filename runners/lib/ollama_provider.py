"""
OllamaModelProvider — primera implementacion de ModelProvider (ADR-0007).

Vive fuera del Kernel. El Kernel solo conoce el contrato ModelProvider.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, Optional

import requests


class OllamaModelProvider:
    """ModelProvider sobre Ollama local (ADR-0007, ADR-0011)."""

    name = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        num_gpu: int = 99,
        default_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.num_gpu = num_gpu
        self.default_options = default_options or {}

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if r.status_code != 200:
                return False
            models = [m.get("name", "") for m in r.json().get("models", [])]
            return any(self.model in m for m in models)
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        *,
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> str:
        opts = {"num_gpu": self.num_gpu, **self.default_options}
        if options:
            opts.update(options)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": opts,
            "keep_alive": "10m",
        }
        r = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=timeout or 120,
        )
        r.raise_for_status()
        return (r.json().get("response") or "").strip()

    def stream(
        self,
        prompt: str,
        *,
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Iterator[str]:
        opts = {"num_gpu": self.num_gpu, **self.default_options}
        if options:
            opts.update(options)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": opts,
            "keep_alive": "10m",
        }
        with requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            stream=True,
            timeout=timeout or 300,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    import json
                    data = json.loads(line)
                except Exception:
                    continue
                chunk = data.get("response") or ""
                if chunk:
                    yield chunk
                if data.get("done"):
                    break
