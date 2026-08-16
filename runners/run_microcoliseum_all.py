"""
Lanza las 9 corridas del microcoliseum: 3 modelos x 3 modos.
"""
import subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
SCRIPT = str(ROOT / "experiments" / "microcoliseum_deliberation" / "run_microcoliseum.py")

MODELS = [
    ("ibm/granite4.1:3b-q4_K_M", "granite-3b-q4"),
    ("qwen3-4b-rag:latest",       "qwen3-4b-rag"),
    ("llama3.2:3b",               "llama32-3b"),
]

MODES = ["independent", "debate-on-disagreement", "debate-all"]

t0 = time.time()
total = len(MODELS) * len(MODES)
idx = 0

for model_name, label in MODELS:
    for mode in MODES:
        idx += 1
        print(f"\n{'='*70}", flush=True)
        print(f"RUN {idx}/{total}: {label} / {mode}", flush=True)
        print(f"{'='*70}", flush=True)

        cmd = [PY, "-u", SCRIPT, "--model", model_name, "--mode", mode, "--gpu", "--label", label]
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"WARNING: run {idx} exited with code {rc}", flush=True)

wall = time.time() - t0
print(f"\n{'='*70}", flush=True)
print(f"ALL DONE: {total} runs in {wall/60:.1f} min", flush=True)
print(f"{'='*70}", flush=True)
