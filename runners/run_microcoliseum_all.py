"""
Lanza las 12 corridas del microcoliseum: 4 modelos x 3 modos.
"""
import argparse, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
SCRIPT = str(ROOT / "runners" / "debate.py")

MODELS = [
    ("gemma3:4b-it-q4_K_M", "gemma3-4b-q4"),
    ("dhiltgen/nemotron-3-nano:4b", "nemotron-3-4b-q4"),
    ("TechyShishy/ministral-3:3b-reasoning-2512-q4_K_M", "ministral-3b-q4"),
    ("qwen3.5:4b-q4_K_M", "qwen35-4b-q4"),
]

MODES = ["independent", "debate-on-disagreement", "debate-all"]

parser = argparse.ArgumentParser(description="Lanza las 12 corridas del microcoliseum")
parser.add_argument("--port", type=int, default=11434,
                    help="Ollama instance port (default: 11434). "
                         "Use a dedicated instance to avoid model thrashing.")
parser.add_argument("--gpu", action="store_true", default=True,
                    help="Use GPU (default: True)")
args = parser.parse_args()

t0 = time.time()
total = len(MODELS) * len(MODES)
idx = 0

for model_name, label in MODELS:
    for mode in MODES:
        idx += 1
        print(f"\n{'='*70}", flush=True)
        print(f"RUN {idx}/{total}: {label} / {mode}  (port={args.port})", flush=True)
        print(f"{'='*70}", flush=True)

        cmd = [PY, "-u", SCRIPT,
               "--model", model_name,
               "--mode", mode,
               "--gpu" if args.gpu else "--cpu",
               "--label", label,
               "--port", str(args.port)]
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"WARNING: run {idx} exited with code {rc}", flush=True)

wall = time.time() - t0
print(f"\n{'='*70}", flush=True)
print(f"ALL DONE: {total} runs in {wall/60:.1f} min (port={args.port})", flush=True)
print(f"{'='*70}", flush=True)
