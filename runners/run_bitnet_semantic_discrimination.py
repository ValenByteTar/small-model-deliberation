"""
EXP-024: Semantic Discrimination x Decoding.

Controla la variable de confusion descubierta en EXP-023 (grammar)
para determinar si BitNet tiene señal semantica explotable en la
distribucion de probabilidades.

Pregunta: ¿La distribucion P(TRUE)/P(FALSE) cambia sistematicamente
segun la relacion semantica entre claim y evidence?

Diseño factorial:
  - Pares minimos: mismo evidence, claims que difieren en un elemento
  - 3 condiciones de grammar: strict / permissive / none
  - 2 vocabularios: TRUE/FALSE y YES/NO
  - Medir: P(TRUE), P(FALSE), P(CANNOT_TELL) por caso

Si P(TRUE|SUPPORTS) > P(TRUE|CONTRADICTS): hay señal semantica.
Si P(TRUE|SUPPORTS) ≈ P(TRUE|CONTRADICTS): no hay señal, solo sesgo.

No busca maximizar accuracy. Busca detectar señal.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

ROOT = Path(__file__).resolve().parent.parent

BITNET_ROOT = os.environ.get("BITNET_ROOT", os.path.expanduser("~/BitNet"))
MODEL_PATH = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
SERVER_EXE = "build/bin/Release/llama-server.exe"

BENCHMARK = ROOT / "benchmarks" / "semantic_discrimination_v1.json"
OUTPUT_DIR = ROOT / "results" / "raw"

PORT = 8130
SEED = 42
N_PROBS = 15  # top-15 logprobs para capturar todos los tokens relevantes
NUM_PREDICT = 4
TEMPERATURE = 0.0
NUM_THREAD = 4

# ----------------------------- Grammars -----------------------------

GRAMMARS = {
    "strict": 'root ::= "TRUE" | "FALSE" | "CANNOT_TELL"',
    "permissive": (
        'root ::= "TRUE" | "FALSE" | "CANNOT_TELL" '
        '| " TRUE" | " FALSE" | " CANNOT_TELL" '
        '| " True" | " False" | " Cannot_tell" '
        '| " true" | " false" | " cannot_tell"'
    ),
    "none": "",
}

# ----------------------------- Token Maps -----------------------------

# Mapa completo: todas las variantes posibles
TOKEN_MAP = {
    "TRUE": "TRUE", " TRUE": "TRUE", "True": "TRUE", " True": "TRUE",
    "true": "TRUE", " true": "TRUE", " TRUE": "TRUE",
    "FALSE": "FALSE", " FALSE": "FALSE", "False": "FALSE", " False": "FALSE",
    "false": "FALSE", " false": "FALSE", " FALSE": "FALSE",
    "CANNOT_TELL": "CANNOT_TELL", " CANNOT_TELL": "CANNOT_TELL",
    "Cannot_tell": "CANNOT_TELL", " Cannot_tell": "CANNOT_TELL",
    "cannot_tell": "CANNOT_TELL", " cannot_tell": "CANNOT_TELL",
}

# ----------------------------- Prompts -----------------------------

# Prompt NLI 3a (mismo de EXP-018/023)
FEW_SHOT = '''Task: Based on the EVIDENCE, determine if the CLAIM is TRUE, FALSE, or CANNOT_TELL.

CLAIM: The NIST CSF has five core functions
EVIDENCE: The Framework Core consists of five Functions: Identify, Detect, Protect, Respond, and Recover.
Based on the evidence, the claim is: TRUE

CLAIM: The NIST CSF requires all organizations to use multi-factor authentication
EVIDENCE: The NIST Cybersecurity Framework provides guidance for managing cybersecurity risk.
Based on the evidence, the claim is: FALSE

CLAIM: The NIST CSF mandates specific encryption algorithms for data protection
EVIDENCE: The NIST Cybersecurity Framework provides guidance for managing cybersecurity risk.
Based on the evidence, the claim is: CANNOT_TELL

'''

# ----------------------------- Server -----------------------------

_server_proc: Optional[subprocess.Popen] = None


def start_server() -> bool:
    global _server_proc
    exe = os.path.join(BITNET_ROOT, SERVER_EXE)
    model = os.path.join(BITNET_ROOT, MODEL_PATH)
    if not os.path.exists(exe) or not os.path.exists(model):
        print(f"ERROR: Binario o modelo no encontrado", flush=True)
        return False
    try:
        r = requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2)
        if r.status_code == 200:
            print(f"  llama-server ya activo en puerto {PORT}", flush=True)
            return True
    except Exception:
        pass
    cmd = [exe, "-m", model, "--host", "127.0.0.1", "--port", str(PORT),
           "-t", str(NUM_THREAD), "-c", "2048", "-ngl", "0",
           "--seed", str(SEED),
           "--override-kv", "tokenizer.ggml.pre=str:llama3"]
    print(f"  Iniciando llama-server en puerto {PORT} (seed={SEED})...", flush=True)
    _server_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    creationflags=0x08000000)
    t0 = time.time()
    while time.time() - t0 < 60:
        try:
            r = requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2)
            if r.status_code == 200:
                print(f"  Puerto {PORT} listo", flush=True)
                return True
        except Exception:
            pass
        time.sleep(1)
    print(f"  ERROR: Timeout en puerto {PORT}", flush=True)
    return False


def stop_server():
    global _server_proc
    if _server_proc:
        try:
            _server_proc.terminate()
            _server_proc.wait(timeout=5)
        except Exception:
            try:
                _server_proc.kill()
            except Exception:
                pass
    _server_proc = None
    print("  Servidor detenido.", flush=True)


# ----------------------------- LLM Call -----------------------------

def call_with_logprobs(prompt: str, grammar: str) -> dict:
    url = f"http://127.0.0.1:{PORT}"
    payload: Dict[str, Any] = {
        "prompt": prompt, "stream": False, "temperature": TEMPERATURE,
        "max_tokens": NUM_PREDICT, "repeat_penalty": 1.0,
        "seed": SEED, "n_probs": N_PROBS,
    }
    if grammar:
        payload["grammar"] = grammar
    try:
        resp = requests.post(f"{url}/completion", json=payload, timeout=90)
        resp.raise_for_status()
        data = resp.json()
        return {"content": data.get("content", "").strip(),
                "probs": data.get("completion_probabilities", [])}
    except Exception as exc:
        return {"content": f"ERROR: {exc}", "probs": [], "error": str(exc)}


def get_first_token_logprobs(probs: list) -> List[dict]:
    if not probs:
        return []
    return probs[0].get("top_logprobs", [])


# ----------------------------- Probability Extraction -----------------------------

def extract_probabilities(token_logprobs: list) -> Dict[str, float]:
    """Extrae P(TRUE), P(FALSE), P(CANNOT_TELL) desde los top-N logprobs.

    Usa logsumexp para agregar variantes del mismo token (TRUE,  TRUE, True, etc.)
    Convierte a probabilidades normalizadas sobre los tres labels.
    """
    by_label = defaultdict(list)
    for tl in token_logprobs:
        tok = tl.get("token", "")
        lp = tl.get("logprob", -999)
        # Mapear token a label
        label = (TOKEN_MAP.get(tok) or TOKEN_MAP.get(tok.strip())
                 or TOKEN_MAP.get(tok.lower()) or TOKEN_MAP.get(" " + tok.strip().lower()))
        if label:
            by_label[label].append(lp)

    # Logsumexp por label
    log_probs = {}
    for label, lps in by_label.items():
        if lps:
            max_lp = max(lps)
            log_probs[label] = max_lp + math.log(sum(math.exp(lp - max_lp) for lp in lps))

    # Convertir a probabilidades normalizadas sobre TRUE/FALSE/CANNOT_TELL
    labels = ["TRUE", "FALSE", "CANNOT_TELL"]
    # Si un label no aparece en top-N, asignar logprob muy bajo
    for label in labels:
        if label not in log_probs:
            log_probs[label] = -20.0  # ~e^-20 ≈ 2e-9

    # Softmax sobre los tres
    max_lp = max(log_probs[l] for l in labels)
    exp_vals = {l: math.exp(log_probs[l] - max_lp) for l in labels}
    total = sum(exp_vals.values())
    probs = {l: exp_vals[l] / total for l in labels}

    return probs, log_probs


# ----------------------------- Experiment -----------------------------

def run_case(claim: str, evidence: str, grammar_name: str, grammar: str) -> dict:
    """Ejecuta un caso bajo una condicion de grammar."""
    prompt = f"{FEW_SHOT}CLAIM: {claim.strip()}\nEVIDENCE: {evidence.strip()}\nBased on the evidence, the claim is:"
    result = call_with_logprobs(prompt, grammar)
    raw = result["content"]
    lps = get_first_token_logprobs(result["probs"])
    probs, log_probs = extract_probabilities(lps)

    # Token raw elegido (greedy)
    greedy_token = ""
    if lps:
        greedy_token = lps[0].get("token", "")

    return {
        "raw_output": raw,
        "greedy_token": greedy_token,
        "probabilities": {k: round(v, 6) for k, v in probs.items()},
        "log_probs": {k: round(v, 4) for k, v in log_probs.items()},
        "first_token_logprobs": [{"token": tl.get("token", ""), "logprob": round(tl.get("logprob", 0), 4)} for tl in lps],
    }


def main():
    print("=" * 75, flush=True)
    print("EXP-024: SEMANTIC DISCRIMINATION x DECODING", flush=True)
    print("=" * 75, flush=True)
    print(f"\nModel: BitNet-b1.58-2B-4T", flush=True)
    print(f"Seed: {SEED}", flush=True)
    print(f"Temperature: {TEMPERATURE}", flush=True)
    print(f"n_probs: {N_PROBS}", flush=True)
    print(f"Grammar conditions: {list(GRAMMARS.keys())}", flush=True)

    # Cargar benchmark
    with BENCHMARK.open("r", encoding="utf-8") as f:
        bench = json.load(f)

    # Construir lista de casos: cada par genera 3 claims (A, B, C)
    all_cases = []
    for group in bench["groups"]:
        for pair in group["pairs"]:
            for variant in ["a", "b", "c"]:
                claim_key = f"claim_{variant}"
                expected_key = f"claim_{variant}_expected"
                if claim_key in pair and expected_key in pair:
                    all_cases.append({
                        "group_id": group["group_id"],
                        "group_desc": group["description"],
                        "pair_id": pair["pair_id"],
                        "evidence": pair["evidence"],
                        "claim": pair[claim_key],
                        "expected": pair[expected_key],
                        "variant": variant.upper(),
                        "rationale": pair.get("rationale", ""),
                    })

    print(f"Cases: {len(all_cases)} (20 pairs x 3 variants)", flush=True)
    print(f"Grammar conditions: {len(GRAMMARS)}", flush=True)
    print(f"Total LLM calls: {len(all_cases) * len(GRAMMARS)}", flush=True)

    if not start_server():
        return 1

    try:
        # Ejecutar: para cada grammar, correr todos los casos
        all_results = {}
        t_total = time.time()

        for grammar_name, grammar in GRAMMARS.items():
            print(f"\n{'='*70}", flush=True)
            print(f"  Grammar: {grammar_name}", flush=True)
            print(f"{'='*70}", flush=True)

            all_results[grammar_name] = []
            for i, case in enumerate(all_cases):
                t0 = time.time()
                result = run_case(case["claim"], case["evidence"], grammar_name, grammar)
                dt = time.time() - t0

                probs = result["probabilities"]
                p_true = probs.get("TRUE", 0)
                p_false = probs.get("FALSE", 0)
                p_cannot = probs.get("CANNOT_TELL", 0)

                result["case"] = case
                result["grammar"] = grammar_name
                result["latency_s"] = round(dt, 4)
                all_results[grammar_name].append(result)

                print(f"  [{i+1:2d}/{len(all_cases)}] {case['pair_id']}-{case['variant']} ({case['expected']:<12s}) "
                      f"P(T)={p_true:.3f} P(F)={p_false:.3f} P(C)={p_cannot:.3f} "
                      f"greedy={result['greedy_token'][:10]:<10s} [{dt:.2f}s]", flush=True)

        wall = time.time() - t_total

        # ==================== Analisis ====================
        print(f"\n{'='*75}", flush=True)
        print("ANALISIS: SENAL SEMANTICA EN LA DISTRIBUCION", flush=True)
        print(f"{'='*75}", flush=True)

        # Para cada grammar, agrupar por expected relation y calcular P(TRUE)/P(FALSE) promedio
        for grammar_name in GRAMMARS:
            results = all_results[grammar_name]
            by_expected = defaultdict(lambda: {"p_true": [], "p_false": [], "p_cannot": [], "cases": []})

            for r in results:
                exp = r["case"]["expected"]
                probs = r["probabilities"]
                by_expected[exp]["p_true"].append(probs.get("TRUE", 0))
                by_expected[exp]["p_false"].append(probs.get("FALSE", 0))
                by_expected[exp]["p_cannot"].append(probs.get("CANNOT_TELL", 0))
                by_expected[exp]["cases"].append(r)

            print(f"\n  Grammar: {grammar_name}", flush=True)
            print(f"  {'Expected':<14s} {'N':>4s} {'P(TRUE)':>10s} {'P(FALSE)':>10s} {'P(CANNOT)':>10s} {'P(T)-P(F)':>10s}", flush=True)
            print(f"  {'-'*14} {'-'*4} {'-'*10} {'-'*10} {'-'*10} {'-'*10}", flush=True)

            for exp in ["SUPPORTS", "CONTRADICTS", "PARTIAL", "UNRELATED"]:
                if exp in by_expected:
                    d = by_expected[exp]
                    n = len(d["p_true"])
                    avg_t = sum(d["p_true"]) / n
                    avg_f = sum(d["p_false"]) / n
                    avg_c = sum(d["p_cannot"]) / n
                    delta = avg_t - avg_f
                    print(f"  {exp:<14s} {n:>4d} {avg_t:>10.4f} {avg_f:>10.4f} {avg_c:>10.4f} {delta:>+10.4f}", flush=True)

        # ==================== Analisis de pares minimos ====================
        print(f"\n{'='*75}", flush=True)
        print("ANALISIS: PARES MINIMOS (sensibilidad a modificacion semantica)", flush=True)
        print(f"{'='*75}", flush=True)

        # Para cada par, comparar P(TRUE) entre variantes
        for grammar_name in GRAMMARS:
            results = all_results[grammar_name]
            print(f"\n  Grammar: {grammar_name}", flush=True)
            print(f"  {'Pair':<10s} {'A(exp)':<14s} {'P(T|A)':>8s} {'B(exp)':<14s} {'P(T|B)':>8s} {'C(exp)':<14s} {'P(T|C)':>8s} {'A>B':>6s} {'A>C':>6s}", flush=True)
            print(f"  {'-'*10} {'-'*14} {'-'*8} {'-'*14} {'-'*8} {'-'*14} {'-'*8} {'-'*6} {'-'*6}", flush=True)

            # Agrupar por pair_id
            by_pair = defaultdict(dict)
            for r in results:
                pid = r["case"]["pair_id"]
                variant = r["case"]["variant"]
                by_pair[pid][variant] = r

            signal_count = 0
            total_pairs = 0

            for pid in sorted(by_pair.keys()):
                variants = by_pair[pid]
                a = variants.get("A", {})
                b = variants.get("B", {})
                c = variants.get("C", {})

                pta = a.get("probabilities", {}).get("TRUE", 0) if a else 0
                ptb = b.get("probabilities", {}).get("TRUE", 0) if b else 0
                ptc = c.get("probabilities", {}).get("TRUE", 0) if c else 0

                exp_a = a["case"]["expected"] if a else "?"
                exp_b = b["case"]["expected"] if b else "?"
                exp_c = c["case"]["expected"] if c else "?"

                a_gt_b = "Y" if pta > ptb + 0.05 else ("N" if pta < ptb - 0.05 else "~")
                a_gt_c = "Y" if pta > ptc + 0.05 else ("N" if pta < ptc - 0.05 else "~")

                # Contar señal: si A es SUPPORTS y B/C no lo son, A>B y A>C esperados
                if exp_a == "SUPPORTS" and exp_b != "SUPPORTS":
                    total_pairs += 1
                    if pta > ptb + 0.05:
                        signal_count += 1
                if exp_a == "SUPPORTS" and exp_c != "SUPPORTS":
                    total_pairs += 1
                    if pta > ptc + 0.05:
                        signal_count += 1

                print(f"  {pid:<10s} {exp_a:<14s} {pta:>8.4f} {exp_b:<14s} {ptb:>8.4f} {exp_c:<14s} {ptc:>8.4f} {a_gt_b:>6s} {a_gt_c:>6s}", flush=True)

            if total_pairs > 0:
                print(f"\n  Signal detection: {signal_count}/{total_pairs} pares donde A=SUPPORTS y B/C no -> P(T|A) > P(T|B/C) + 0.05", flush=True)
                print(f"  Signal rate: {signal_count/total_pairs:.1%}", flush=True)

        # ==================== Resumen final ====================
        print(f"\n{'='*75}", flush=True)
        print("VEREDICTO: ¿HAY SENAL SEMANTICA?", flush=True)
        print(f"{'='*75}", flush=True)

        for grammar_name in GRAMMARS:
            results = all_results[grammar_name]
            # Calcular P(TRUE) promedio para SUPPORTS vs no-SUPPORTS
            supp = [r for r in results if r["case"]["expected"] == "SUPPORTS"]
            non_supp = [r for r in results if r["case"]["expected"] != "SUPPORTS"]

            if supp and non_supp:
                avg_pt_supp = sum(r["probabilities"].get("TRUE", 0) for r in supp) / len(supp)
                avg_pt_non = sum(r["probabilities"].get("TRUE", 0) for r in non_supp) / len(non_supp)
                delta = avg_pt_supp - avg_pt_non

                print(f"\n  Grammar: {grammar_name}", flush=True)
                print(f"    P(TRUE | SUPPORTS)     = {avg_pt_supp:.4f} (n={len(supp)})", flush=True)
                print(f"    P(TRUE | non-SUPPORTS) = {avg_pt_non:.4f} (n={len(non_supp)})", flush=True)
                print(f"    Delta                  = {delta:+.4f}", flush=True)

                if delta > 0.10:
                    print(f"    -> HAY señal semantica (delta > 0.10)", flush=True)
                elif delta > 0.05:
                    print(f"    -> Señal debil (0.05 < delta < 0.10)", flush=True)
                else:
                    print(f"    -> NO hay señal semantica (delta < 0.05)", flush=True)

        print(f"\n  Wall time: {wall:.0f}s ({wall/60:.1f} min)", flush=True)

        # ==================== Guardar ====================
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUTPUT_DIR / "bitnet_semantic_discrimination.json"
        report = {
            "experiment": "EXP-024 Semantic Discrimination x Decoding",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": "BitNet-b1.58-2B-4T",
            "seed": SEED,
            "temperature": TEMPERATURE,
            "n_probs": N_PROBS,
            "benchmark": "semantic_discrimination_v1.json",
            "grammar_conditions": list(GRAMMARS.keys()),
            "case_count": len(all_cases),
            "total_llm_calls": len(all_cases) * len(GRAMMARS),
            "wall_time_s": round(wall, 1),
            "results_by_grammar": {
                gname: [
                    {
                        "case": r["case"],
                        "grammar": r["grammar"],
                        "raw_output": r["raw_output"],
                        "greedy_token": r["greedy_token"],
                        "probabilities": r["probabilities"],
                        "log_probs": r["log_probs"],
                        "first_token_logprobs": r["first_token_logprobs"],
                        "latency_s": r["latency_s"],
                    }
                    for r in results
                ]
                for gname, results in all_results.items()
            },
        }
        with out.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n  Reporte: {out}", flush=True)

    finally:
        stop_server()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
