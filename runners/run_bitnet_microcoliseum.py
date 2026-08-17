"""
EXP-022: BitNet Microcoliseum - 4 Workers Especializados.

Adapta el microcoliseum para BitNet usando los hallazgos de
EXP-017 a EXP-021. Cada worker usa el regimen/prompt que mejores
resultados dio para su capacidad especifica:

  Worker A (entailment):   NLI 3a TRUE/FALSE/CANNOT_TELL (greedy)
    -> EXP-018: 12/12 SUPPORTS (dice TRUE para todo, acierta SUPPORTS)
  Worker B (skeptical):    NLI Cascading TRUE -> FULL/PARTIAL
    -> EXP-020: unico regimen que emite PARTIAL (54.5%)
  Worker C (contradiction): NLI 3a TRUE/FALSE/CANNOT_TELL (argmax)
    -> EXP-018: 16/16 CONTRADICTS en argmax (FALSE token fuerte)
  Worker D (context):      Relevance gate YES/NO
    -> EXP-019: 100% irrelevantes claros, wrong_subject 100%

Arquitectura:

  4 instancias llama-server (puertos 8101-8104) en paralelo
  Cada worker -> su propia instancia + prompt + grammar + mapping
  Phase 1: 4 workers en paralelo (ThreadPoolExecutor)
  Phase 2: logit ensemble (logsumexp) con pesos por worker
  Phase 3: debate si disagreement (GBNF, no JSON)
  Phase 4: judge deterministico (policy, no LLM)

La autoridad esta en el sistema: logprobs + policy determinista.
BitNet produce senales, no decisiones.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

ROOT = Path(__file__).resolve().parent.parent

BITNET_ROOT = os.environ.get("BITNET_ROOT", os.path.expanduser("~/BitNet"))
MODEL_PATH = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
SERVER_EXE = "build/bin/Release/llama-server.exe"

BENCHMARK = ROOT / "benchmarks" / "semantic_assessment_v2.json"
OUTPUT_DIR = ROOT / "results" / "raw"

BASE_PORT = 8101
NUM_WORKERS = 4
N_PROBS = 8

# ----------------------------- Grammars -----------------------------

# Grammars estrictos - sin espacios ni variantes de caso
# EXP-018 uso grammars estrictos y obtuvo resultados reproducibles
# Los espacios cambian la tokenizacion y BitNet prefiere " FALSE" sobre "TRUE"
GRAMMAR_NLI3 = 'root ::= "TRUE" | "FALSE" | "CANNOT_TELL"'
GRAMMAR_YES_NO = 'root ::= "YES" | "NO"'
GRAMMAR_FULL_PARTIAL = 'root ::= "FULL" | "PARTIAL"'
# ----------------------------- Token Maps -----------------------------

MAP_NLI3 = {
    "TRUE": "SUPPORTS", " TRUE": "SUPPORTS", "true": "SUPPORTS", " True": "SUPPORTS", " true": "SUPPORTS",
    "FALSE": "CONTRADICTS", " FALSE": "CONTRADICTS", "false": "CONTRADICTS", " False": "CONTRADICTS", " false": "CONTRADICTS",
    "CANNOT_TELL": "UNRELATED", " CANNOT_TELL": "UNRELATED", "cannot_tell": "UNRELATED", " Cannot_tell": "UNRELATED", " cannot_tell": "UNRELATED",
}

MAP_YES_NO = {
    "YES": "RELEVANT", " YES": "RELEVANT", "yes": "RELEVANT", " Yes": "RELEVANT", " yes": "RELEVANT",
    "NO": "IRRELEVANT", " NO": "IRRELEVANT", "no": "IRRELEVANT", " No": "IRRELEVANT", " no": "IRRELEVANT",
}

MAP_FULL_PARTIAL = {
    "FULL": "SUPPORTS", " FULL": "SUPPORTS", "full": "SUPPORTS", " Full": "SUPPORTS",
    "PARTIAL": "PARTIAL", " PARTIAL": "PARTIAL", "partial": "PARTIAL", " Partial": "PARTIAL",
}

# ----------------------------- Few-Shot Prompts -----------------------------

# Worker A: entailment analyst - NLI 3a (TRUE/FALSE/CANNOT_TELL)
# Exp-018: dice TRUE para todo -> acierta SUPPORTS (12/12)
# Usa los mismos few-shot examples del dominio NIST CSF que funcionaron en EXP-018
PROMPT_A = '''Task: Based on the EVIDENCE, determine if the CLAIM is TRUE, FALSE, or CANNOT_TELL.

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

# Worker B: skeptical analyst - NLI Cascading (TRUE -> FULL/PARTIAL)
# Exp-020: unico regimen que emite PARTIAL
PROMPT_B_STEP1 = '''Task: Based on the EVIDENCE, is the CLAIM TRUE or FALSE?

CLAIM: The system supports encryption at rest, encryption in transit, and access logging.
EVIDENCE: The system supports encryption at rest, encryption in transit, and access logging.
Based on the evidence, the claim is: TRUE

CLAIM: The system supports encryption at rest only.
EVIDENCE: The system supports encryption at rest, encryption in transit, and access logging.
Based on the evidence, the claim is: FALSE

'''

PROMPT_B_STEP2 = '''Task: The claim is partially supported by the evidence. Does the evidence fully or partially confirm the claim?

CLAIM: The system supports encryption at rest, encryption in transit, and access logging.
EVIDENCE: The system supports encryption at rest and encryption in transit.
Does the evidence fully or partially confirm the claim? PARTIAL

CLAIM: The system supports encryption at rest, encryption in transit, and access logging.
EVIDENCE: The system supports encryption at rest, encryption in transit, and access logging.
Does the evidence fully or partially confirm the claim? FULL

'''

# Worker C: contradiction analyst - NLI 3a (TRUE/FALSE/CANNOT_TELL)
# Exp-018: 16/16 CONTRADICTS en argmax (FALSE token fuerte)
# Mismo prompt que A pero con ejemplos orientados a contradiccion (dominio NIST)
PROMPT_C = '''Task: Based on the EVIDENCE, determine if the CLAIM is TRUE, FALSE, or CANNOT_TELL. Focus on finding contradictions.

CLAIM: The NIST CSF requires all organizations to use multi-factor authentication
EVIDENCE: The NIST Cybersecurity Framework provides guidance for managing cybersecurity risk.
Based on the evidence, the claim is: FALSE

CLAIM: The NIST CSF mandates specific encryption algorithms for data protection
EVIDENCE: The NIST Cybersecurity Framework provides guidance for managing cybersecurity risk.
Based on the evidence, the claim is: FALSE

CLAIM: The NIST CSF has five core functions
EVIDENCE: The Framework Core consists of five Functions: Identify, Detect, Protect, Respond, and Recover.
Based on the evidence, the claim is: TRUE

'''

# Worker D: context/entity analyst - Relevance gate (YES/NO)
# Exp-019: 100% irrelevantes claros, wrong_subject 100%
PROMPT_D = '''Task: Do the CLAIM and EVIDENCE discuss the same specific subject (same product, standard, framework, technique) in the same context (same environment, sector, lifecycle phase)? Answer YES or NO. Contradictions about the same subject are still relevant.

CLAIM: Product X supports encryption at rest.
EVIDENCE: Product X supports encryption at rest and in transit.
Do the claim and evidence discuss the same subject and context? YES

CLAIM: Product X supports encryption at rest.
EVIDENCE: Product Y supports encryption at rest and in transit.
Do the claim and evidence discuss the same subject and context? NO

CLAIM: The NIST framework requires multi-factor authentication.
EVIDENCE: The NIST framework does not require multi-factor authentication for low-risk systems.
Do the claim and evidence discuss the same subject and context? YES

'''

# Debate prompts: uno por worker, adaptado a su capacidad
# NO pedir razonamiento composicional (EXP-021: BitNet no puede)
# En cambio: re-preguntar con framing diferente + senal del otro worker

# Worker A debate: si alguien dijo CONTRADICTS, re-preguntar con framing de contradiccion
PROMPT_DEBATE_A = '''Task: Another analyst says the claim is FALSE based on the evidence. Re-examine: is the claim TRUE, FALSE, or CANNOT_TELL?

CLAIM: The NIST CSF has five core functions
EVIDENCE: The Framework Core consists of five Functions: Identify, Detect, Protect, Respond, and Recover.
Another analyst concluded: FALSE
Based on the evidence, the claim is: TRUE

CLAIM: {claim}
EVIDENCE: {evidence}
Another analyst concluded: FALSE

Based on the evidence, the claim is:'''

# Worker B debate: si alguien dijo SUPPORTS, re-preguntar FULL/PARTIAL
PROMPT_DEBATE_B = '''Task: Another analyst says the claim is fully supported. Re-examine: does the evidence FULLY or PARTIALLY confirm the claim?

CLAIM: The framework includes risk assessment, mitigation, and recovery guidance
EVIDENCE: The framework provides risk assessment and mitigation guidance for organizations.
Another analyst concluded: FULL support
Does the evidence fully or partially confirm the claim? PARTIAL

CLAIM: {claim}
EVIDENCE: {evidence}
Another analyst concluded: FULL support

Does the evidence fully or partially confirm the claim?'''

# Worker C debate: si alguien dijo SUPPORTS, re-preguntar TRUE/FALSE
PROMPT_DEBATE_C = '''Task: Another analyst says the claim is TRUE based on the evidence. Re-examine carefully for contradictions: is the claim TRUE, FALSE, or CANNOT_TELL?

CLAIM: The NIST CSF requires all organizations to use multi-factor authentication
EVIDENCE: The NIST Cybersecurity Framework provides guidance for managing cybersecurity risk.
Another analyst concluded: TRUE
Based on the evidence, the claim is: FALSE

CLAIM: {claim}
EVIDENCE: {evidence}
Another analyst concluded: TRUE

Based on the evidence, the claim is:'''

# Worker D debate: si alguien dijo que es relevante, re-preguntar relevance
PROMPT_DEBATE_D = '''Task: Another analyst says the claim and evidence are relevant to each other. Re-examine: do they discuss the same specific subject and context?

CLAIM: Product X supports encryption at rest.
EVIDENCE: Product Y supports encryption at rest and in transit.
Another analyst concluded: YES, same subject
Do the claim and evidence discuss the same subject and context? NO

CLAIM: {claim}
EVIDENCE: {evidence}
Another analyst concluded: YES, same subject

Do the claim and evidence discuss the same subject and context?'''

# ----------------------------- Worker Specs -----------------------------

WORKER_SPECS = [
    {
        "id": "A",
        "role": "entailment",
        "port": BASE_PORT,
        "prompt": PROMPT_A,
        "grammar": GRAMMAR_NLI3,
        "token_map": MAP_NLI3,
        "weight": 1.0,  # peso en el ensemble
        "description": "NLI 3a greedy: acierta SUPPORTS (TRUE para todo)",
    },
    {
        "id": "B",
        "role": "skeptical",
        "port": BASE_PORT + 1,
        "prompt": PROMPT_B_STEP1,
        "prompt_step2": PROMPT_B_STEP2,
        "grammar": GRAMMAR_NLI3,  # step1 usa TRUE/FALSE
        "grammar_step2": GRAMMAR_FULL_PARTIAL,  # step2 usa FULL/PARTIAL
        "token_map": MAP_NLI3,
        "token_map_step2": MAP_FULL_PARTIAL,
        "weight": 1.0,
        "description": "NLI Cascading: unico que emite PARTIAL",
    },
    {
        "id": "C",
        "role": "contradiction",
        "port": BASE_PORT + 2,
        "prompt": PROMPT_C,
        "grammar": GRAMMAR_NLI3,
        "token_map": MAP_NLI3,
        "weight": 1.0,
        "description": "NLI 3a argmax: 16/16 CONTRADICTS (FALSE token fuerte)",
    },
    {
        "id": "D",
        "role": "context",
        "port": BASE_PORT + 3,
        "prompt": PROMPT_D,
        "grammar": GRAMMAR_YES_NO,
        "token_map": MAP_YES_NO,
        "weight": 1.5,  # peso mayor: relevance gate es el mas confiable
        "description": "Relevance gate: 100% irrelevantes claros",
    },
]

# ----------------------------- Server Management -----------------------------

_servers: List[subprocess.Popen] = []


def start_server(port: int) -> bool:
    exe = os.path.join(BITNET_ROOT, SERVER_EXE)
    model = os.path.join(BITNET_ROOT, MODEL_PATH)
    if not os.path.exists(exe) or not os.path.exists(model):
        print(f"ERROR: Binario o modelo no encontrado", flush=True)
        return False
    try:
        r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
        if r.status_code == 200:
            print(f"  llama-server ya activo en puerto {port}", flush=True)
            return True
    except Exception:
        pass
    cmd = [exe, "-m", model, "--host", "127.0.0.1", "--port", str(port),
           "-t", "2", "-c", "2048", "-ngl", "0",
           "--override-kv", "tokenizer.ggml.pre=str:llama3"]
    print(f"  Iniciando llama-server en puerto {port}...", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            creationflags=0x08000000)
    _servers.append(proc)
    t0 = time.time()
    while time.time() - t0 < 60:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if r.status_code == 200:
                print(f"  Puerto {port} listo", flush=True)
                return True
        except Exception:
            pass
        time.sleep(1)
    print(f"  ERROR: Timeout en puerto {port}", flush=True)
    return False


def start_all_servers() -> bool:
    print(f"Iniciando {NUM_WORKERS} instancias llama-server...", flush=True)
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {pool.submit(start_server, BASE_PORT + i): i for i in range(NUM_WORKERS)}
        results = {f.result() for f in as_completed(futures)}
    return all(results)


def stop_all_servers():
    global _servers
    for proc in _servers:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    _servers = []
    print("Servidores detenidos.", flush=True)


# ----------------------------- LLM Calls -----------------------------

def call_with_logprobs(prompt: str, grammar: str, port: int, max_tokens: int = 6) -> dict:
    url = f"http://127.0.0.1:{port}"
    try:
        resp = requests.post(
            f"{url}/completion",
            json={"prompt": prompt, "stream": False, "temperature": 0.0,
                  "max_tokens": max_tokens, "repeat_penalty": 1.0,
                  "grammar": grammar, "n_probs": N_PROBS},
            timeout=90,
        )
        resp.raise_for_status()
        data = resp.json()
        return {"content": data.get("content", "").strip(),
                "probs": data.get("completion_probabilities", [])}
    except Exception as exc:
        return {"content": f"ERROR: {exc}", "probs": []}


def get_token_logprobs(probs: list) -> List[dict]:
    if not probs:
        return []
    return probs[0].get("top_logprobs", [])


def aggregate_logprobs(token_logprobs: list, token_map: dict) -> Dict[str, float]:
    by_label = defaultdict(list)
    for tl in token_logprobs:
        tok = tl.get("token", "")
        lp = tl.get("logprob", -999)
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


def map_raw(raw: str, token_map: dict) -> str:
    raw_lower = raw.strip().lower()
    for key, val in token_map.items():
        if key.strip().lower() in raw_lower:
            return val
    vals = list(set(token_map.values()))
    return vals[0] if vals else "UNRELATED"


# ----------------------------- Worker Execution -----------------------------

@dataclass
class WorkerResult:
    worker_id: str
    role: str
    greedy: str  # etiqueta final (SUPPORTS/PARTIAL/CONTRADICTS/UNRELATED)
    argmax: str
    logprobs: Dict[str, float] = field(default_factory=dict)
    raw: str = ""
    # Para Worker B (cascading)
    step1_greedy: str = ""
    step2_greedy: str = ""
    step1_logprobs: Dict[str, float] = field(default_factory=dict)
    step2_logprobs: Dict[str, float] = field(default_factory=dict)


def run_worker_a(spec: dict, claim: str, evidence: str) -> WorkerResult:
    """Worker A: entailment - NLI 3a TRUE/FALSE/CANNOT_TELL."""
    prompt = f"{spec['prompt']}CLAIM: {claim.strip()}\nEVIDENCE: {evidence.strip()}\nBased on the evidence, the claim is:"
    result = call_with_logprobs(prompt, spec["grammar"], spec["port"], max_tokens=6)
    raw = result["content"]
    lps = get_token_logprobs(result["probs"])
    agg = aggregate_logprobs(lps, spec["token_map"])
    greedy = map_raw(raw, spec["token_map"])
    argmax = max(agg, key=agg.get) if agg else greedy
    return WorkerResult(
        worker_id=spec["id"], role=spec["role"],
        greedy=greedy, argmax=argmax, logprobs=agg, raw=raw,
    )


def run_worker_b(spec: dict, claim: str, evidence: str) -> WorkerResult:
    """Worker B: skeptical - NLI Cascading TRUE -> FULL/PARTIAL."""
    # Step 1: TRUE/FALSE
    prompt1 = f"{spec['prompt']}CLAIM: {claim.strip()}\nEVIDENCE: {evidence.strip()}\nBased on the evidence, the claim is:"
    result1 = call_with_logprobs(prompt1, spec["grammar"], spec["port"], max_tokens=4)
    raw1 = result1["content"]
    lps1 = get_token_logprobs(result1["probs"])
    agg1 = aggregate_logprobs(lps1, spec["token_map"])
    greedy1 = map_raw(raw1, spec["token_map"])

    # Step 2: si TRUE -> FULL/PARTIAL
    if greedy1 == "SUPPORTS":
        prompt2 = f"{spec['prompt_step2']}CLAIM: {claim.strip()}\nEVIDENCE: {evidence.strip()}\nDoes the evidence fully or partially confirm the claim?"
        result2 = call_with_logprobs(prompt2, spec["grammar_step2"], spec["port"], max_tokens=4)
        raw2 = result2["content"]
        lps2 = get_token_logprobs(result2["probs"])
        agg2 = aggregate_logprobs(lps2, spec["token_map_step2"])
        greedy2 = map_raw(raw2, spec["token_map_step2"])
        final_greedy = greedy2  # SUPPORTS o PARTIAL
        # Combinar logprobs: usar step2 si disponible
        final_logprobs = agg2 if agg2 else agg1
    else:
        # FALSE -> CONTRADICTS
        final_greedy = "CONTRADICTS"
        final_logprobs = agg1
        greedy2 = ""
        agg2 = {}

    argmax = max(final_logprobs, key=final_logprobs.get) if final_logprobs else final_greedy
    return WorkerResult(
        worker_id=spec["id"], role=spec["role"],
        greedy=final_greedy, argmax=argmax, logprobs=final_logprobs,
        raw=f"{raw1}" + (f" -> {raw2}" if greedy2 else ""),
        step1_greedy=greedy1, step2_greedy=greedy2,
        step1_logprobs=agg1, step2_logprobs=agg2,
    )


def run_worker_c(spec: dict, claim: str, evidence: str) -> WorkerResult:
    """Worker C: contradiction - NLI 3a con prompt orientado a contradiccion."""
    prompt = f"{spec['prompt']}CLAIM: {claim.strip()}\nEVIDENCE: {evidence.strip()}\nBased on the evidence, the claim is:"
    result = call_with_logprobs(prompt, spec["grammar"], spec["port"], max_tokens=6)
    raw = result["content"]
    lps = get_token_logprobs(result["probs"])
    agg = aggregate_logprobs(lps, spec["token_map"])
    greedy = map_raw(raw, spec["token_map"])
    argmax = max(agg, key=agg.get) if agg else greedy
    return WorkerResult(
        worker_id=spec["id"], role=spec["role"],
        greedy=greedy, argmax=argmax, logprobs=agg, raw=raw,
    )


def run_worker_d(spec: dict, claim: str, evidence: str) -> WorkerResult:
    """Worker D: context - Relevance gate YES/NO."""
    prompt = f"{spec['prompt']}CLAIM: {claim.strip()}\nEVIDENCE: {evidence.strip()}\nDo the claim and evidence discuss the same subject and context?"
    result = call_with_logprobs(prompt, spec["grammar"], spec["port"], max_tokens=4)
    raw = result["content"]
    lps = get_token_logprobs(result["probs"])
    agg = aggregate_logprobs(lps, spec["token_map"])
    greedy_raw = map_raw(raw, spec["token_map"])  # RELEVANT o IRRELEVANT

    # Convertir a etiqueta final: IRRELEVANT -> UNRELATED, RELEVANT -> SUPPORTS (default)
    # El logprob de RELEVANT se mapea a SUPPORTS, IRRELEVANT a UNRELATED
    final_logprobs = {}
    if "RELEVANT" in agg:
        final_logprobs["SUPPORTS"] = agg["RELEVANT"]
    if "IRRELEVANT" in agg:
        final_logprobs["UNRELATED"] = agg["IRRELEVANT"]

    if greedy_raw == "IRRELEVANT":
        greedy = "UNRELATED"
    else:
        greedy = "SUPPORTS"  # si es relevante, default a SUPPORTS (el aggregator decide)

    argmax = max(final_logprobs, key=final_logprobs.get) if final_logprobs else greedy
    return WorkerResult(
        worker_id=spec["id"], role=spec["role"],
        greedy=greedy, argmax=argmax, logprobs=final_logprobs, raw=raw,
    )


WORKER_FUNCS = {
    "A": run_worker_a,
    "B": run_worker_b,
    "C": run_worker_c,
    "D": run_worker_d,
}


# ----------------------------- Aggregation -----------------------------

VALID_RELATIONS = ["SUPPORTS", "PARTIAL", "CONTRADICTS", "UNRELATED"]


def ensemble_logsumexp(worker_results: List[WorkerResult]) -> Tuple[str, float, dict]:
    """Logit ensemble: logsumexp de logprobs de todos los workers, ponderado."""
    combined: Dict[str, float] = defaultdict(lambda: 0.0)

    for wr in worker_results:
        spec = next(s for s in WORKER_SPECS if s["id"] == wr.worker_id)
        weight = spec["weight"]
        for label, lp in wr.logprobs.items():
            if label in VALID_RELATIONS:
                combined[label] += weight * lp

    if not combined:
        return "UNRELATED", 0.0, {"combined_logprobs": {}}

    final_relation = max(combined, key=combined.get)
    # Confidence: softmax sobre los logprobs combinados
    max_lp = max(combined.values())
    exp_vals = {k: math.exp(v - max_lp) for k, v in combined.items()}
    total_exp = sum(exp_vals.values())
    confidence = exp_vals[final_relation] / total_exp if total_exp > 0 else 0.0

    return final_relation, round(confidence, 4), {"combined_logprobs": {k: round(v, 3) for k, v in combined.items()}}


def has_disagreement(worker_results: List[WorkerResult]) -> bool:
    rels = {wr.greedy for wr in worker_results if wr.greedy in VALID_RELATIONS}
    return len(rels) > 1


# ----------------------------- Debate Phase -----------------------------

# Mapa de prompts de debate por worker
DEBATE_PROMPTS = {
    "A": (PROMPT_DEBATE_A, GRAMMAR_NLI3, MAP_NLI3),
    "B": (PROMPT_DEBATE_B, GRAMMAR_FULL_PARTIAL, MAP_FULL_PARTIAL),
    "C": (PROMPT_DEBATE_C, GRAMMAR_NLI3, MAP_NLI3),
    "D": (PROMPT_DEBATE_D, GRAMMAR_YES_NO, MAP_YES_NO),
}


def run_debate_round(worker_results: List[WorkerResult], claim: str, evidence: str) -> List[dict]:
    """Phase 3: debate adaptado por worker.

    En lugar de pedir KEEP/CHANGE (razonamiento composicional que
    BitNet no puede hacer), cada worker re-evalua con un framing
    diferente que incorpora la senal del worker en desacuerdo.

    Worker A (SUPPORTS): si alguien dijo CONTRADICTS/UNRELATED,
      re-preguntar con framing de contradiccion
    Worker B (PARTIAL): si alguien dijo SUPPORTS,
      re-preguntar FULL/PARTIAL
    Worker C (CONTRADICTS): si alguien dijo SUPPORTS,
      re-preguntar TRUE/FALSE con framing de contradiccion
    Worker D (relevance): si alguien dijo relevante,
      re-preguntar relevance

    El debate produce nuevas logprobs, no KEEP/CHANGE.
    """
    debate_results = []
    cl = claim.strip()[:400]
    ev = evidence.strip()[:800]

    for wr in worker_results:
        spec = next(s for s in WORKER_SPECS if s["id"] == wr.worker_id)
        prompt_template, grammar, token_map = DEBATE_PROMPTS[wr.worker_id]

        prompt = prompt_template.format(claim=cl, evidence=ev)
        result = call_with_logprobs(prompt, grammar, spec["port"], max_tokens=6)
        raw = result["content"]
        lps = get_token_logprobs(result["probs"])
        agg = aggregate_logprobs(lps, token_map)

        # Mapear a etiqueta final
        if wr.worker_id == "D":
            # Relevance: IRRELEVANT -> UNRELATED, RELEVANT -> SUPPORTS
            debate_logprobs = {}
            if "RELEVANT" in agg:
                debate_logprobs["SUPPORTS"] = agg["RELEVANT"]
            if "IRRELEVANT" in agg:
                debate_logprobs["UNRELATED"] = agg["IRRELEVANT"]
            debate_greedy = "UNRELATED" if map_raw(raw, token_map) == "IRRELEVANT" else "SUPPORTS"
        else:
            debate_logprobs = agg
            debate_greedy = map_raw(raw, token_map)

        # Determinar si cambio comparando greedy inicial vs debate
        changed = debate_greedy != wr.greedy

        debate_results.append({
            "worker": wr.worker_id,
            "initial": wr.greedy,
            "debate_greedy": debate_greedy,
            "debate_logprobs": debate_logprobs,
            "changed": changed,
            "raw": raw,
        })
    return debate_results


# ----------------------------- Judge (Deterministic) -----------------------------

def deterministic_judge(
    worker_results: List[WorkerResult],
    ensemble_relation: str,
    ensemble_confidence: float,
    ensemble_meta: dict,
    debate_results: Optional[List[dict]] = None,
) -> Tuple[str, float, str]:
    """Judge deterministico: combina ensemble + debate + policy.

    Policy (ordenada por confianza experimental):
    1. Worker D (relevance gate) greedy UNRELATED -> override UNRELATED
       (EXP-019: 100% irrelevantes claros)
    2. Worker D debate tambien UNRELATED -> override mas fuerte
    3. Worker B (skeptical) greedy PARTIAL -> boost PARTIAL
       (EXP-020: unico que emite PARTIAL)
    4. Worker B debate tambien PARTIAL -> boost mas fuerte
    5. Worker C (contradiction) argmax CONTRADICTS -> boost CONTRADICTS
       (EXP-018: 16/16 CONTRADICTS en argmax)
    6. Worker C debate tambien CONTRADICTS -> boost mas fuerte
    7. Worker A (entailment) greedy SUPPORTS -> boost SUPPORTS
       (EXP-018: 12/12 SUPPORTS en greedy)
    8. Recalcular con logprobs + debate logprobs (logsumexp)
    """
    worker_d = next((wr for wr in worker_results if wr.worker_id == "D"), None)
    worker_b = next((wr for wr in worker_results if wr.worker_id == "B"), None)
    worker_c = next((wr for wr in worker_results if wr.worker_id == "C"), None)
    worker_a = next((wr for wr in worker_results if wr.worker_id == "A"), None)

    combined_lp = dict(ensemble_meta.get("combined_logprobs", {}))
    reason_parts = []

    # Policy 1+2: Worker D (relevance gate)
    d_greedy_unrelated = worker_d and worker_d.greedy == "UNRELATED"
    d_debate = next((d for d in (debate_results or []) if d["worker"] == "D"), None)
    d_debate_unrelated = d_debate and d_debate.get("debate_greedy") == "UNRELATED"

    if d_greedy_unrelated and d_debate_unrelated:
        return "UNRELATED", 0.95, "Worker D confirmo irrelevante en Phase 1 y debate. Override."
    if d_greedy_unrelated:
        return "UNRELATED", 0.9, "Worker D (relevance gate) detecto irrelevante. Override a UNRELATED."

    # Incorporar debate logprobs al combined_lp (logsumexp)
    if debate_results:
        for d in debate_results:
            wid = d["worker"]
            spec = next(s for s in WORKER_SPECS if s["id"] == wid)
            weight = spec["weight"] * 0.5  # debate tiene menos peso que phase 1
            for label, lp in d.get("debate_logprobs", {}).items():
                if label in VALID_RELATIONS:
                    combined_lp[label] = combined_lp.get(label, 0.0) + weight * lp
        reason_parts.append("Debate logprobs integrados (peso 0.5)")

    # Policy 3+4: Worker B (skeptical) PARTIAL
    b_greedy_partial = worker_b and worker_b.greedy == "PARTIAL"
    b_debate = next((d for d in (debate_results or []) if d["worker"] == "B"), None)
    b_debate_partial = b_debate and b_debate.get("debate_greedy") == "PARTIAL"

    if b_greedy_partial:
        combined_lp["PARTIAL"] = combined_lp.get("PARTIAL", 0.0) + (3.0 if b_debate_partial else 2.0)
        reason_parts.append(f"Worker B PARTIAL ({'confirmado en debate' if b_debate_partial else 'phase 1'})")

    # Policy 5+6: Worker C (contradiction) CONTRADICTS
    c_argmax_contradicts = worker_c and worker_c.argmax == "CONTRADICTS"
    c_debate = next((d for d in (debate_results or []) if d["worker"] == "C"), None)
    c_debate_contradicts = c_debate and c_debate.get("debate_greedy") == "CONTRADICTS"

    if c_argmax_contradicts:
        combined_lp["CONTRADICTS"] = combined_lp.get("CONTRADICTS", 0.0) + (2.5 if c_debate_contradicts else 1.5)
        reason_parts.append(f"Worker C CONTRADICTS argmax ({'confirmado en debate' if c_debate_contradicts else 'phase 1'})")

    # Policy 7: Worker A (entailment) SUPPORTS
    a_greedy_supports = worker_a and worker_a.greedy == "SUPPORTS"
    if a_greedy_supports:
        combined_lp["SUPPORTS"] = combined_lp.get("SUPPORTS", 0.0) + 1.0
        reason_parts.append("Worker A SUPPORTS greedy (12/12 en EXP-018)")

    # Recalcular final
    if combined_lp:
        final = max(combined_lp, key=combined_lp.get)
        max_lp = max(combined_lp.values())
        exp_vals = {k: math.exp(v - max_lp) for k, v in combined_lp.items()}
        total_exp = sum(exp_vals.values())
        conf = exp_vals[final] / total_exp if total_exp > 0 else 0.0
    else:
        final = ensemble_relation
        conf = ensemble_confidence

    if not reason_parts:
        reason_parts.append(f"Ensemble logsumexp: {final}")
    return final, round(conf, 4), "; ".join(reason_parts)


# ----------------------------- Case Execution -----------------------------

@dataclass
class CaseResult:
    case_id: str
    category: str
    ground_truth: str
    worker_results: List[WorkerResult] = field(default_factory=list)
    ensemble_relation: str = ""
    ensemble_confidence: float = 0.0
    ensemble_meta: dict = field(default_factory=dict)
    disagreement: bool = False
    debate_results: List[dict] = field(default_factory=list)
    final_relation: str = ""
    final_confidence: float = 0.0
    judge_reason: str = ""
    initial_correct: bool = False
    final_correct: bool = False
    latency_s: float = 0.0


def run_case(case: dict, mode: str) -> CaseResult:
    claim = case["claim"]
    evidence = case["evidence"]
    result = CaseResult(case_id=case["id"], category=case["category"], ground_truth=case["expected"])
    t0 = time.time()

    # Phase 1: 4 workers en paralelo
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {}
        for spec in WORKER_SPECS:
            fn = WORKER_FUNCS[spec["id"]]
            futures[pool.submit(fn, spec, claim, evidence)] = spec["id"]
        worker_results = []
        for f in as_completed(futures):
            wr = f.result()
            worker_results.append(wr)
    # Ordenar por worker_id
    worker_results.sort(key=lambda w: w.worker_id)
    result.worker_results = worker_results

    # Phase 2: Ensemble (logsumexp)
    ens_rel, ens_conf, ens_meta = ensemble_logsumexp(worker_results)
    result.ensemble_relation = ens_rel
    result.ensemble_confidence = ens_conf
    result.ensemble_meta = ens_meta
    result.initial_correct = (ens_rel == result.ground_truth)

    # Phase 3: Debate si disagreement
    result.disagreement = has_disagreement(worker_results)
    if mode == "debate-on-disagreement" and result.disagreement:
        result.debate_results = run_debate_round(worker_results, claim, evidence)
    elif mode == "debate-all":
        result.debate_results = run_debate_round(worker_results, claim, evidence)

    # Phase 4: Judge deterministico
    final_rel, final_conf, judge_reason = deterministic_judge(
        worker_results, ens_rel, ens_conf, ens_meta,
        result.debate_results if result.debate_results else None,
    )
    result.final_relation = final_rel
    result.final_confidence = final_conf
    result.judge_reason = judge_reason
    result.final_correct = (final_rel == result.ground_truth)

    result.latency_s = round(time.time() - t0, 2)
    return result


# ----------------------------- Metrics -----------------------------

def compute_metrics(results: List[CaseResult], mode: str) -> dict:
    n = len(results)
    initial_correct = sum(1 for r in results if r.initial_correct)
    final_correct = sum(1 for r in results if r.final_correct)
    initial_acc = initial_correct / n if n > 0 else 0.0
    final_acc = final_correct / n if n > 0 else 0.0
    delta = final_acc - initial_acc

    corrections = sum(1 for r in results if not r.initial_correct and r.final_correct)
    damage = sum(1 for r in results if r.initial_correct and not r.final_correct)
    stable = sum(1 for r in results if r.ensemble_relation == r.final_relation)

    by_cat = defaultdict(lambda: {"total": 0, "initial_correct": 0, "final_correct": 0})
    for r in results:
        by_cat[r.category]["total"] += 1
        if r.initial_correct:
            by_cat[r.category]["initial_correct"] += 1
        if r.final_correct:
            by_cat[r.category]["final_correct"] += 1

    # Worker-level accuracy
    worker_acc = {}
    for wid in ["A", "B", "C", "D"]:
        correct = sum(1 for r in results for wr in r.worker_results
                      if wr.worker_id == wid and wr.greedy == r.ground_truth)
        worker_acc[wid] = {"correct": correct, "total": n, "accuracy": round(correct / n, 4) if n > 0 else 0.0}

    return {
        "n": n, "mode": mode,
        "initial_accuracy": round(initial_acc, 4),
        "final_accuracy": round(final_acc, 4),
        "accuracy_delta": round(delta, 4),
        "initial_correct": initial_correct,
        "final_correct": final_correct,
        "corrections": corrections,
        "damage": damage,
        "net": corrections - damage,
        "stability_rate": round(stable / n, 4) if n > 0 else 0.0,
        "by_category": {cat: {"total": s["total"],
                              "initial_accuracy": round(s["initial_correct"] / s["total"], 4) if s["total"] > 0 else 0.0,
                              "final_accuracy": round(s["final_correct"] / s["total"], 4) if s["total"] > 0 else 0.0}
                        for cat, s in sorted(by_cat.items())},
        "worker_accuracy": worker_acc,
    }


# ----------------------------- Main -----------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="BitNet Microcoliseum - 4 Workers Especializados")
    parser.add_argument("--mode", default="debate-on-disagreement",
                        choices=["independent", "debate-on-disagreement", "debate-all"])
    args = parser.parse_args()

    print("=" * 75, flush=True)
    print("EXP-022: BITNET MICROCOLISEUM - 4 WORKERS ESPECIALIZADOS", flush=True)
    print("=" * 75, flush=True)
    print(f"\nWorkers:")
    for spec in WORKER_SPECS:
        print(f"  Worker {spec['id']} ({spec['role']}): port={spec['port']}, {spec['description']}", flush=True)
    print(f"\nMode: {args.mode}", flush=True)

    # Start servers
    if not start_all_servers():
        print("ERROR: No se pudieron iniciar los servidores", flush=True)
        return 1

    try:
        with BENCHMARK.open("r", encoding="utf-8") as f:
            cases = json.load(f)["cases"]
        n = len(cases)
        print(f"\nBenchmark: {n} casos\n", flush=True)

        results: List[CaseResult] = []
        t_total = time.time()

        for i, case in enumerate(cases):
            cr = run_case(case, args.mode)
            results.append(cr)
            workers_str = " ".join(f"{wr.worker_id}={wr.greedy[:4]}" for wr in cr.worker_results)
            status = f"ens={cr.ensemble_relation[:4]}({'OK' if cr.initial_correct else 'X'}) "
            status += f"final={cr.final_relation[:4]}({'OK' if cr.final_correct else 'X'})"
            debate_str = f" debate({sum(1 for d in cr.debate_results if d.get('changed'))}ch)" if cr.debate_results else ""
            print(f"  [{i+1:2d}/{n}] {case['id']:<8s} ({case['category']:<22s}) [{workers_str}] {status}{debate_str} [{cr.latency_s:.1f}s]", flush=True)

        wall = time.time() - t_total
        metrics = compute_metrics(results, args.mode)

        print(f"\n{'='*75}", flush=True)
        print("RESULTS SUMMARY", flush=True)
        print(f"{'='*75}", flush=True)
        print(f"  Initial (ensemble):  {metrics['initial_accuracy']:.1%} ({metrics['initial_correct']}/{n})", flush=True)
        print(f"  Final (judge):       {metrics['final_accuracy']:.1%} ({metrics['final_correct']}/{n})", flush=True)
        print(f"  Delta:               {metrics['accuracy_delta']:+.1%}", flush=True)
        print(f"  Corrections:         {metrics['corrections']}", flush=True)
        print(f"  Damage:              {metrics['damage']}", flush=True)
        print(f"  Net:                 {metrics['net']:+d}", flush=True)
        print(f"  Stability:           {metrics['stability_rate']:.1%}", flush=True)
        print(f"\n  Worker accuracy (greedy vs ground truth):", flush=True)
        for wid in ["A", "B", "C", "D"]:
            wa = metrics["worker_accuracy"][wid]
            spec = next(s for s in WORKER_SPECS if s["id"] == wid)
            print(f"    Worker {wid} ({spec['role']:<14s}): {wa['correct']}/{wa['total']} ({wa['accuracy']:.1%})", flush=True)
        print(f"\n  By category:", flush=True)
        print(f"    {'Categoria':<24} {'N':>4} {'Init':>8} {'Final':>8} {'Delta':>8}", flush=True)
        for cat, s in sorted(metrics["by_category"].items()):
            d = s["final_accuracy"] - s["initial_accuracy"]
            print(f"    {cat:<24} {s['total']:>4} {s['initial_accuracy']:>8.1%} {s['final_accuracy']:>8.1%} {d:>+8.1%}", flush=True)
        print(f"\n  Wall time: {wall:.0f}s ({wall/60:.1f} min)", flush=True)

        # Guardar
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUTPUT_DIR / "bitnet_microcoliseum_specialized.json"
        report = {
            "experiment": "EXP-022 BitNet Microcoliseum Specialized",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "mode": args.mode,
            "wall_time_s": round(wall, 1),
            "worker_specs": [{"id": s["id"], "role": s["role"], "port": s["port"],
                              "description": s["description"], "weight": s["weight"]}
                             for s in WORKER_SPECS],
            "metrics": metrics,
            "cases": [{
                "case_id": r.case_id, "category": r.category, "ground_truth": r.ground_truth,
                "worker_results": [{"worker_id": wr.worker_id, "role": wr.role,
                                    "greedy": wr.greedy, "argmax": wr.argmax,
                                    "logprobs": wr.logprobs, "raw": wr.raw}
                                   for wr in r.worker_results],
                "ensemble_relation": r.ensemble_relation,
                "ensemble_confidence": r.ensemble_confidence,
                "ensemble_meta": r.ensemble_meta,
                "disagreement": r.disagreement,
                "debate_results": r.debate_results,
                "final_relation": r.final_relation,
                "final_confidence": r.final_confidence,
                "judge_reason": r.judge_reason,
                "initial_correct": r.initial_correct,
                "final_correct": r.final_correct,
                "latency_s": r.latency_s,
            } for r in results],
        }
        with out.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n  Reporte: {out}", flush=True)

    finally:
        stop_all_servers()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
