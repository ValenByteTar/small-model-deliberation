"""
EXP-023: Grammar Sensitivity Probe for BitNet 1.58b.

Descubierto en EXP-022: los grammars GBNF permisivos (con espacios y
variantes de caso) cambian la tokenizacion de BitNet, invirtiendo su
comportamiento. Esto afecta retrospectivamente la interpretacion de
EXP-017-022.

Hipotesis: La gramatica de decodificacion es una variable experimental
de primer orden para BitNet. Diferentes niveles de restriccion
gramatical pueden cambiar la distribucion de outputs y la accuracy.

Diseño:
  - Mismo modelo, prompt, temperature, seed, casos, num_predict
  - Variar EXCLUSIVAMENTE el grammar:
    G1 = grammar estricto (solo forma canonica)
    G2 = grammar permisivo (espacios + variantes de caso)
    G3 = sin grammar + parser
  - Medir:
    accuracy, output distribution, logprobs, token elegido,
    latency, tasa de outputs invalidos

El prompt usado es el NLI 3a de EXP-018 (TRUE/FALSE/CANNOT_TELL),
que es donde se observo la inversion de comportamiento.

Uso:
    python -u runners/run_bitnet_grammar_sensitivity.py
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

ROOT = Path(__file__).resolve().parent.parent

BITNET_ROOT = os.environ.get("BITNET_ROOT", os.path.expanduser("~/BitNet"))
MODEL_PATH = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
SERVER_EXE = "build/bin/Release/llama-server.exe"

BENCHMARK = ROOT / "benchmarks" / "semantic_assessment_v2.json"
OUTPUT_DIR = ROOT / "results" / "raw"

PORT = 8120
SEED = 42  # seed fijo para reproducibilidad
N_PROBS = 10  # top-10 logprobs para analisis detallado
NUM_PREDICT = 6
TEMPERATURE = 0.0
NUM_THREAD = 4

# ----------------------------- Grammars -----------------------------

# G1: estricto - solo forma canonica, sin espacios ni variantes
GRAMMAR_G1_STRICT = 'root ::= "TRUE" | "FALSE" | "CANNOT_TELL"'

# G2: permisivo - espacios + variantes de caso (como EXP-019/020/021)
GRAMMAR_G2_PERMISSIVE = (
    'root ::= "TRUE" | "FALSE" | "CANNOT_TELL" '
    '| " TRUE" | " FALSE" | " CANNOT_TELL" '
    '| " True" | " False" | " Cannot_tell" '
    '| " true" | " false" | " cannot_tell"'
)

# G3: sin grammar - el modelo genera libremente, parser extrae la etiqueta
GRAMMAR_G3_NONE = ""

# ----------------------------- Token Maps -----------------------------

# Mapa estricto: solo formas canonicas
MAP_STRICT = {
    "TRUE": "SUPPORTS",
    "FALSE": "CONTRADICTS",
    "CANNOT_TELL": "UNRELATED",
}

# Mapa permisivo: todas las variantes
MAP_PERMISSIVE = {
    "TRUE": "SUPPORTS", " TRUE": "SUPPORTS", "True": "SUPPORTS", " True": "SUPPORTS",
    "true": "SUPPORTS", " true": "SUPPORTS", " TRUE": "SUPPORTS",
    "FALSE": "CONTRADICTS", " FALSE": "CONTRADICTS", "False": "CONTRADICTS", " False": "CONTRADICTS",
    "false": "CONTRADICTS", " false": "CONTRADICTS", " FALSE": "CONTRADICTS",
    "CANNOT_TELL": "UNRELATED", " CANNOT_TELL": "UNRELATED", "Cannot_tell": "UNRELATED",
    " Cannot_tell": "UNRELATED", "cannot_tell": "UNRELATED", " cannot_tell": "UNRELATED",
    " CANNOT_TELL": "UNRELATED",
}

VALID_RELATIONS = ["SUPPORTS", "CONTRADICTS", "UNRELATED"]

# ----------------------------- Prompt (NLI 3a de EXP-018) -----------------------------

FEW_SHOT_NLI_3 = '''Task: Based on the EVIDENCE, determine if the CLAIM is TRUE, FALSE, or CANNOT_TELL.

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

# ----------------------------- Server Management -----------------------------

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

def call_with_logprobs(prompt: str, grammar: str, max_tokens: int = NUM_PREDICT) -> dict:
    url = f"http://127.0.0.1:{PORT}"
    payload: Dict[str, Any] = {
        "prompt": prompt,
        "stream": False,
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens,
        "repeat_penalty": 1.0,
        "seed": SEED,
        "n_probs": N_PROBS,
    }
    if grammar:
        payload["grammar"] = grammar

    try:
        resp = requests.post(f"{url}/completion", json=payload, timeout=90)
        resp.raise_for_status()
        data = resp.json()
        return {
            "content": data.get("content", "").strip(),
            "probs": data.get("completion_probabilities", []),
            "timings": data.get("timings", {}),
        }
    except Exception as exc:
        return {"content": f"ERROR: {exc}", "probs": [], "timings": {}, "error": str(exc)}


def get_first_token_logprobs(probs: list) -> List[dict]:
    if not probs:
        return []
    return probs[0].get("top_logprobs", [])


# ----------------------------- Parsing -----------------------------

def parse_strict(raw: str) -> Tuple[str, bool]:
    """Parse estricto: solo acepta TRUE/FALSE/CANNOT_TELL exacto."""
    raw_stripped = raw.strip()
    for key, val in MAP_STRICT.items():
        if raw_stripped == key:
            return val, True
    # Intentar primera palabra
    first_word = raw_stripped.split()[0] if raw_stripped.split() else ""
    for key, val in MAP_STRICT.items():
        if first_word == key:
            return val, True
    return "UNRELATED", False  # invalid


def parse_permissive(raw: str) -> Tuple[str, bool]:
    """Parse permisivo: acepta cualquier variante de caso/espacio."""
    raw_lower = raw.strip().lower()
    for key, val in MAP_PERMISSIVE.items():
        if key.strip().lower() in raw_lower:
            return val, True
    return "UNRELATED", False  # invalid


def parse_no_grammar(raw: str) -> Tuple[str, bool]:
    """Parse para sin grammar: extraer primera etiqueta valida del output."""
    raw_lower = raw.strip().lower()
    # Buscar cualquiera de las tres etiquetas
    for keyword, label in [("cannot_tell", "UNRELATED"), ("true", "SUPPORTS"), ("false", "CONTRADICTS")]:
        if keyword in raw_lower:
            return label, True
    return "UNRELATED", False  # invalid


# ----------------------------- Logprob Aggregation -----------------------------

def aggregate_logprobs(token_logprobs: list, token_map: dict) -> Dict[str, float]:
    """Logsumexp de logprobs por etiqueta."""
    by_label = defaultdict(list)
    for tl in token_logprobs:
        tok = tl.get("token", "")
        lp = tl.get("logprob", -999)
        # Probar mapeos: exacto, stripped, lower, " " + stripped lower
        label = (token_map.get(tok) or token_map.get(tok.strip())
                 or token_map.get(tok.lower()) or token_map.get(" " + tok.strip().lower()))
        if label:
            by_label[label].append(lp)
    result = {}
    for label, lps in by_label.items():
        if lps:
            max_lp = max(lps)
            result[label] = max_lp + math.log(sum(math.exp(lp - max_lp) for lp in lps))
    return result


# ----------------------------- Conditions -----------------------------

CONDITIONS = [
    {
        "id": "G1_strict",
        "name": "G1: Grammar estricto",
        "grammar": GRAMMAR_G1_STRICT,
        "token_map": MAP_STRICT,
        "parser": parse_strict,
        "description": 'root ::= "TRUE" | "FALSE" | "CANNOT_TELL"',
    },
    {
        "id": "G2_permissive",
        "name": "G2: Grammar permisivo",
        "grammar": GRAMMAR_G2_PERMISSIVE,
        "token_map": MAP_PERMISSIVE,
        "parser": parse_permissive,
        "description": "TRUE |  TRUE |  True |  true | FALSE | ... (12 variantes)",
    },
    {
        "id": "G3_none",
        "name": "G3: Sin grammar + parser",
        "grammar": GRAMMAR_G3_NONE,
        "token_map": MAP_PERMISSIVE,  # usar mapa permisivo para logprobs
        "parser": parse_no_grammar,
        "description": "Sin restriccion gramatical, parser extrae etiqueta",
    },
]


# ----------------------------- Experiment -----------------------------

def run_condition(condition: dict, cases: list) -> dict:
    """Ejecuta una condicion (grammar) sobre todos los casos."""
    cond_id = condition["id"]
    cond_name = condition["name"]
    grammar = condition["grammar"]
    token_map = condition["token_map"]
    parser = condition["parser"]

    print(f"\n{'='*70}", flush=True)
    print(f"  Condicion: {cond_name}", flush=True)
    print(f"  Grammar: {condition['description']}", flush=True)
    print(f"{'='*70}", flush=True)

    results = []
    correct = 0
    invalid_count = 0
    token_counter = Counter()  # que token raw genero
    label_counter = Counter()  # a que etiqueta se mapeo
    latency_list = []
    all_logprobs = []  # para analisis de distribucion

    for i, case in enumerate(cases):
        claim = case["claim"]
        evidence = case["evidence"]
        expected = case["expected"]

        # Solo nos interesan SUPPORTS, CONTRADICTS, UNRELATED para este experimento
        # (PARTIAL no es mapeable en NLI 3a)
        if expected == "PARTIAL":
            # Para PARTIAL, el "correct" seria CANNOT_TELL (no soportado totalmente)
            # pero lo marcamos como N/A para no confundir
            pass

        prompt = f"{FEW_SHOT_NLI_3}CLAIM: {claim.strip()}\nEVIDENCE: {evidence.strip()}\nBased on the evidence, the claim is:"

        t0 = time.time()
        result = call_with_logprobs(prompt, grammar, max_tokens=NUM_PREDICT)
        latency = time.time() - t0

        raw = result["content"]
        lps = get_first_token_logprobs(result["probs"])
        agg = aggregate_logprobs(lps, token_map)

        # Parse
        parsed_label, valid = parser(raw)

        # Argmax from logprobs
        argmax_label = max(agg, key=agg.get) if agg else parsed_label

        # Greedy = parsed label
        greedy_label = parsed_label

        # Accuracy: comparar greedy con expected
        # Para PARTIAL, ninguno de los tres es correcto, pero lo contamos
        is_correct = (greedy_label == expected)
        if is_correct:
            correct += 1
        if not valid:
            invalid_count += 1

        # Contadores
        token_counter[raw[:20]] += 1  # primeros 20 chars para identificar
        label_counter[greedy_label] += 1
        latency_list.append(latency)
        all_logprobs.append(agg)

        status = "OK" if is_correct else "X"
        valid_str = "" if valid else " [INVALID]"
        print(f"  [{i+1:2d}/{len(cases)}] {case['id']:<8s} raw={raw[:15]:<15s} -> {greedy_label:<12s} (exp={expected:<12s}) {status}{valid_str} [{latency:.2f}s]", flush=True)

        results.append({
            "case_id": case["id"],
            "category": case["category"],
            "expected": expected,
            "raw_output": raw,
            "first_token_logprobs": [{"token": tl.get("token", ""), "logprob": tl.get("logprob", 0)} for tl in lps],
            "aggregated_logprobs": {k: round(v, 4) for k, v in agg.items()},
            "greedy_label": greedy_label,
            "argmax_label": argmax_label,
            "valid": valid,
            "correct": is_correct,
            "latency_s": round(latency, 4),
        })

    n = len(cases)
    accuracy = correct / n if n > 0 else 0.0
    invalid_rate = invalid_count / n if n > 0 else 0.0
    avg_latency = sum(latency_list) / len(latency_list) if latency_list else 0.0
    p50_latency = sorted(latency_list)[len(latency_list) // 2] if latency_list else 0.0

    # Accuracy por categoria
    by_cat = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        by_cat[r["category"]]["total"] += 1
        if r["correct"]:
            by_cat[r["category"]]["correct"] += 1

    # Accuracy excluyendo PARTIAL (que NLI 3a no puede mapear)
    non_partial = [r for r in results if r["expected"] != "PARTIAL"]
    non_partial_correct = sum(1 for r in non_partial if r["correct"])
    non_partial_acc = non_partial_correct / len(non_partial) if non_partial else 0.0

    print(f"\n  --- Resumen {cond_id} ---", flush=True)
    print(f"  Accuracy (all):          {accuracy:.1%} ({correct}/{n})", flush=True)
    print(f"  Accuracy (excl PARTIAL): {non_partial_acc:.1%} ({non_partial_correct}/{len(non_partial)})", flush=True)
    print(f"  Invalid rate:            {invalid_rate:.1%} ({invalid_count}/{n})", flush=True)
    print(f"  Avg latency:             {avg_latency:.3f}s", flush=True)
    print(f"  P50 latency:             {p50_latency:.3f}s", flush=True)
    print(f"  Label distribution:      {dict(label_counter)}", flush=True)
    print(f"  Top raw tokens:", flush=True)
    for tok, count in token_counter.most_common(5):
        print(f"    '{tok}' x{count}", flush=True)

    return {
        "condition_id": cond_id,
        "condition_name": cond_name,
        "grammar": condition["description"],
        "n": n,
        "accuracy": round(accuracy, 4),
        "accuracy_excl_partial": round(non_partial_acc, 4),
        "correct": correct,
        "invalid_count": invalid_count,
        "invalid_rate": round(invalid_rate, 4),
        "avg_latency_s": round(avg_latency, 4),
        "p50_latency_s": round(p50_latency, 4),
        "label_distribution": dict(label_counter),
        "top_raw_tokens": dict(token_counter.most_common(10)),
        "by_category": {
            cat: {"total": s["total"], "correct": s["correct"],
                  "accuracy": round(s["correct"] / s["total"], 4) if s["total"] > 0 else 0.0}
            for cat, s in sorted(by_cat.items())
        },
        "cases": results,
    }


# ----------------------------- Main -----------------------------

def main():
    print("=" * 75, flush=True)
    print("EXP-023: GRAMMAR SENSITIVITY PROBE FOR BITNET 1.58b", flush=True)
    print("=" * 75, flush=True)
    print(f"\nModel: BitNet-b1.58-2B-4T", flush=True)
    print(f"Seed: {SEED}", flush=True)
    print(f"Temperature: {TEMPERATURE}", flush=True)
    print(f"num_predict: {NUM_PREDICT}", flush=True)
    print(f"n_probs: {N_PROBS}", flush=True)
    print(f"Prompt: NLI 3a (TRUE/FALSE/CANNOT_TELL) de EXP-018", flush=True)
    print(f"\nCondiciones:", flush=True)
    for cond in CONDITIONS:
        print(f"  {cond['id']}: {cond['description']}", flush=True)

    # Cargar benchmark
    with BENCHMARK.open("r", encoding="utf-8") as f:
        cases = json.load(f)["cases"]
    print(f"\nBenchmark: {len(cases)} casos", flush=True)

    # Iniciar servidor
    if not start_server():
        print("ERROR: No se pudo iniciar el servidor", flush=True)
        return 1

    try:
        all_results = []
        t_total = time.time()

        for cond in CONDITIONS:
            cond_result = run_condition(cond, cases)
            all_results.append(cond_result)

        wall = time.time() - t_total

        # ==================== Comparacion ====================
        print(f"\n{'='*75}", flush=True)
        print("COMPARACION DE CONDICIONES", flush=True)
        print(f"{'='*75}", flush=True)
        print(f"\n{'Condicion':<25s} {'Accuracy':>10s} {'Excl.PART':>10s} {'Invalid':>10s} {'AvgLat':>10s} {'P50Lat':>10s}", flush=True)
        print(f"{'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}", flush=True)
        for r in all_results:
            print(f"{r['condition_id']:<25s} {r['accuracy']:>10.1%} {r['accuracy_excl_partial']:>10.1%} {r['invalid_rate']:>10.1%} {r['avg_latency_s']:>10.3f}s {r['p50_latency_s']:>10.3f}s", flush=True)

        print(f"\nDistribucion de labels:", flush=True)
        print(f"{'Condicion':<25s} {'SUPPORTS':>10s} {'CONTRADICTS':>12s} {'UNRELATED':>10s}", flush=True)
        print(f"{'-'*25} {'-'*10} {'-'*12} {'-'*10}", flush=True)
        for r in all_results:
            dist = r["label_distribution"]
            print(f"{r['condition_id']:<25s} {dist.get('SUPPORTS',0):>10d} {dist.get('CONTRADICTS',0):>12d} {dist.get('UNRELATED',0):>10d}", flush=True)

        print(f"\nTop raw tokens por condicion:", flush=True)
        for r in all_results:
            print(f"\n  {r['condition_id']}:", flush=True)
            for tok, count in list(r["top_raw_tokens"].items())[:5]:
                print(f"    '{tok}' x{count}", flush=True)

        print(f"\nAccuracy por categoria:", flush=True)
        cats = sorted(all_results[0]["by_category"].keys())
        header = f"{'Categoria':<24s}"
        for r in all_results:
            header += f" {r['condition_id']:>15s}"
        print(header, flush=True)
        print("-" * len(header), flush=True)
        for cat in cats:
            row = f"{cat:<24s}"
            for r in all_results:
                s = r["by_category"].get(cat, {})
                acc = s.get("accuracy", 0.0)
                row += f" {acc:>15.1%}"
            print(row, flush=True)

        # ==================== Test estadistico ====================
        # McNemar's test para pares de condiciones
        print(f"\nMcNemar's test (pares):", flush=True)
        for i in range(len(all_results)):
            for j in range(i + 1, len(all_results)):
                r1 = all_results[i]
                r2 = all_results[j]
                # Solo casos no-PARTIAL
                cases1 = [c for c in r1["cases"] if c["expected"] != "PARTIAL"]
                cases2 = [c for c in r2["cases"] if c["expected"] != "PARTIAL"]
                # b = r1 correct, r2 wrong; c = r1 wrong, r2 correct
                b = sum(1 for c1, c2 in zip(cases1, cases2) if c1["correct"] and not c2["correct"])
                c = sum(1 for c1, c2 in zip(cases1, cases2) if not c1["correct"] and c2["correct"])
                # McNemar (sin correccion de continuidad)
                if b + c > 0:
                    mcnemar = (b - c) ** 2 / (b + c)
                else:
                    mcnemar = 0.0
                print(f"  {r1['condition_id']} vs {r2['condition_id']}: b={b} c={c} chi2={mcnemar:.2f} {'(significativo)' if mcnemar > 3.841 else ''}", flush=True)

        print(f"\nWall time total: {wall:.0f}s ({wall/60:.1f} min)", flush=True)

        # ==================== Guardar ====================
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUTPUT_DIR / "bitnet_grammar_sensitivity.json"
        report = {
            "experiment": "EXP-023 Grammar Sensitivity Probe",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": "BitNet-b1.58-2B-4T",
            "seed": SEED,
            "temperature": TEMPERATURE,
            "num_predict": NUM_PREDICT,
            "n_probs": N_PROBS,
            "prompt": "NLI 3a (TRUE/FALSE/CANNOT_TELL) de EXP-018",
            "benchmark": "semantic_assessment_v2.json",
            "case_count": len(cases),
            "wall_time_s": round(wall, 1),
            "conditions": all_results,
        }
        with out.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n  Reporte: {out}", flush=True)

    finally:
        stop_server()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
