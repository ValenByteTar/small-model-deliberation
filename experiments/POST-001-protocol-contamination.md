---
id: POST-001
title: "Postmortem: contaminacion sistematica de protocolo en Coliseo v1 y v2"
date: 2026-08-16
status: completed
category: postmortem
severity: critical
components: [semantic_ensemble, ollama_provider, runners]
tags: [protocol-contamination, num_predict, truncation, think-mode, parser, nemotron, ministral, qwen35, llama32, coliseo]
related: [EXP-012, EXP-013, EXP-014, EXP-015]
supersedes: null
superseded_by: null
---

# POST-001 - Contaminacion sistematica de protocolo en Coliseo v1 y v2

## Resumen ejecutivo

Cuatro modelos (Nemotron, Ministral, Qwen3.5, Llama3.2) produjeron
resultados artificialmente bajos en los coliseos v1 y v2 debido a una
combinacion de tres defectos de protocolo que actuaron de forma
sinergica:

1. **`num_predict=10`**: presupuesto de tokens insuficiente para que
   modelos verbosos o con razonamiento interno completen la respuesta.
2. **Modo `think` activo por defecto**: modelos con capacidad de
   razonamiento (Nemotron, Qwen3.5) consumian los 10 tokens en
   razonamiento interno antes de emitir la clasificacion.
3. **Parser leniento que defaultea a `UNRELATED`**: cuando la respuesta
   estaba truncada o vacia, el parser inferia `UNRELATED` como fallback,
   produciendo un falso 100% de protocol_validity.

El impacto fue catastrofico: Llama3.2 reporto 16.4% de accuracy
(historico) cuando el resultado real con protocolo corregido es
**58.2%** (single). La conclusion original de "Llama3.2 es
semanticamente insuficiente" es **falsa** y debe retirarse.

## Timeline

| Fecha | Evento |
|-------|--------|
| 2026-08-16 | EXP-012 (Coliseo v1 CPU): Llama3.2 reporta 16.4% single. Conclusion: "insuficiente". |
| 2026-08-16 | EXP-013 (Coliseo v1 GPU): Llama3.2 confirma 16.4%. Conclusion reforzada. |
| 2026-08-16 | EXP-014 (Micro-Coliseum): Llama3.2 excluido del debate por "capacidad insuficiente". |
| 2026-08-16 | Coliseo v2 GPU: Nemotron reporta 0% accuracy. Investigacion inicia. |
| 2026-08-16 | Diagnostico Nemotron: `think` mode consume los 10 tokens. Fix: `think=false`. |
| 2026-08-16 | Diagnostico Ministral: razonamiento verbal trunca antes de la relacion. Fix: `think=false` + schema JSON simplificado. |
| 2026-08-16 | Diagnostico Qwen3.5: mismo patron que Nemotron. Fix: `think=false`. |
| 2026-08-16 | Sospecha sobre Llama3.2: misma configuracion historica (num_predict=10, texto libre). |
| 2026-08-16 | Prueba controlada de 5 casos: Llama3.2 se trunca con num_predict=10 en los 5 casos. |
| 2026-08-16 | Prueba controlada con num_predict=64 + JSON: 5/5 correctos. |
| 2026-08-16 | REPETICION Coliseo v1 Llama3.2 (CPU, protocolo corregido): single=58.2% (vs 16.4% historico). |
| 2026-08-16 | POST-001 emitido. EXP-012 y EXP-013 marcados como contaminados para Llama3.2. |

## Defectos de protocolo

### Defecto 1: `num_predict=10` insuficiente

El `SemanticWorker.assess()` y todos los runners del Coliseo v1/v2
usaban `num_predict=10` como presupuesto de generacion. Para modelos
que:

- Generan texto explicativo antes de la clasificacion ("Based on the
  analysis, the relationship between the CLAIM..."), o
- Tienen modo `think` activo que genera razonamiento interno antes de
  la respuesta,

10 tokens son insuficientes. El modelo se trunca con `done_reason=length`
antes de emitir SUPPORTS/CONTRADICTS/UNRELATED/PARTIAL.

**Evidencia (Llama3.2, 5 casos representativos, num_predict=10):**

```
d-001  response='Based on the provided text, the relationship between the'
pp-001 response='Based on the analysis, the relationship between the CLAIM'
ps-001 response='Based on the analysis, I would classify the relationship'
ec-001 response='Based on the provided text, I would classify the'
wc-001 response='Based on the analysis, the relationship between the CLAIM'

done_reason = length
eval_count = 10  (en todos los casos)
```

En los 5 casos, el modelo no emitio ninguna relacion. Todas las
respuestas quedaron truncadas en el preambulo.

### Defecto 2: Modo `think` activo por defecto

Ollama v0.32+ soporta modo `think` para modelos con capacidad de
razonamiento (Nemotron, Qwen3.5, Ministral). Cuando `think` esta activo:

1. El modelo genera razonamiento interno (tokens `<think>...</think>`).
2. Esos tokens consumen parte del `num_predict`.
3. Con `num_predict=10`, el razonamiento consume todos los tokens.
4. La respuesta observable queda vacia: `response=""`.

**Evidencia (Nemotron, num_predict=10, think=default):**

```
response = ""
eval_count = 10
done_reason = length
```

El modelo "respondio" pero toda la generacion fue razonamiento interno
no observable. El parser recibia un string vacio.

### Defecto 3: Parser leniento con default `UNRELATED`

El parser `_parse_semantic_response` en `semantic_ensemble.py`:

```python
if not relation:
    relation = "UNRELATED"  # fallback cuando no encuentra ninguna relacion
```

Cuando el modelo producía una respuesta truncada o vacia:

1. El parser no encontraba ninguna relacion valida en el texto.
2. Defaulteaba a `UNRELATED`.
3. Reportaba `protocol_validity=100%` porque `UNRELATED` es una
   relacion valida del vocabulario.

Esto creo una falsa sensacion de que el protocolo funcionaba: el
reporte decia 100% de protocol_validity aunque el modelo nunca habia
emitido una clasificacion real.

**Impacto en accuracy:**

- Casos donde la ground truth era `UNRELATED`: el default acertaba por
  azar (falso positivo).
- Casos donde la ground truth era `SUPPORTS`/`CONTRADICTS`/`PARTIAL`:
  el default fallaba sistematicamente.

Con 10/55 casos `UNRELATED` en el benchmark v2, el default produce
~18% de accuracy base sin que el modelo haga nada. Esto explica el
16.4% de Llama3.2: esta cerca del floor teorico del default.

## Modelos afectados

| Modelo | Sintoma | Causa raiz | Fix aplicado | Resultado |
|--------|---------|------------|--------------|-----------|
| Nemotron-3-4b | 0% accuracy | `think` mode consume 10 tokens | `think=false` | (pendiente re-eval) |
| Qwen3.5-4b | 0% accuracy | `think` mode consume 10 tokens | `think=false` | (pendiente re-eval) |
| Ministral-3b | Baja accuracy, 100% protocol | Razonamiento verbal trunca + parser leniento | `think=false` + schema JSON simplificado | (pendiente re-eval) |
| Llama3.2-3b | 16.4% accuracy (historico) | `num_predict=10` + texto libre + parser leniento | `num_predict=64` + `think=false` + JSON estructurado + parser estricto | **58.2% single, 63.6% ensemble_4** (EXP-012/013 corregido) |
| BitNet-2B-4T | 33.3% single (historico) | **NO contaminado** — semantic_pilot uso max_tokens=256 | N/A | **29.1% single** (EXP-015, confirma PM-003) |

## Fix aplicado

### Fix 1: `think=false` para modelos con razonamiento

En `run_coliseo_v2_gpu.py` se creo `NoThinkOllamaModelProvider` que
envia `think: false` en el payload de Ollama:

```python
class NoThinkOllamaModelProvider(OllamaModelProvider):
    def generate(self, prompt, *, options=None, timeout=None):
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "think": False,  # <-- fix
                "options": opts,
                "keep_alive": "10m",
            },
            ...
        )
```

Aplicado a: Nemotron, Ministral, Qwen3.5 (deteccion automatica por
nombre de modelo).

### Fix 2: Protocolo controlado para Llama3.2

Se creo `run_coliseo_v1_llama32_cpu_controlled.py` con:

- `num_predict=64` (presupuesto ampliado)
- `think=false` (evita razonamiento interno)
- `format=json` con schema estricto (salida estructurada)
- Parser estricto que **no** defaultea a `UNRELATED`; marca
  `PROTOCOL_ERROR` cuando no puede extraer una relacion valida.
- Conservacion de `raw`, `eval_count`, `done_reason` por caso para
  auditoria.

### Fix 3: Parser estricto (sin default)

```python
def parse_strict(gen):
    # ... parse JSON ...
    if relation not in SEMANTIC_RELATIONS:
        return "PROTOCOL_ERROR", 0.0, False, f"invalid_relation: {relation!r}"
    return relation, confidence, True, "ok"
```

`PROTOCOL_ERROR` no es una relacion valida, por lo que
`protocol_validity` ahora refleja realidades: si el modelo no
clasifica, el caso se marca como fallo de protocolo, no como
`UNRELATED`.

## Validacion del fix

### Prueba controlada (5 casos, Llama3.2)

| Config | num_predict | think | format | Resultado |
|--------|-------------|-------|--------|-----------|
| Historico | 10 | default | texto libre | 0/5 (todas truncadas) |
| Controlado | 64 | false | JSON schema | 5/5 correctos |

### Repeticion Coliseo v1 (55 casos, Llama3.2, CPU)

| Config | Historico (num_predict=10, libre) | Controlado (num_predict=64, JSON) | Delta |
|--------|-----------------------------------|-----------------------------------|-------|
| single | 16.4% | **58.2%** | **+41.8%** |
| ensemble_2 | 29.1% | (en progreso) | — |
| ensemble_4 | 20.0% | (en progreso) | — |

El resultado single (58.2%) refuta definitivamente la conclusion de
"insuficiencia semantica". Llama3.2 no es un modelo excelente (58.2%
esta por debajo del 60% requerido por PM-003), pero esta en el mismo
rango que Granite (61.8% single en EXP-012), no en el rango de
"incapaz".

## Conclusiones

### Conclusiones retiradas

> ~~"Llama 3.2 3B no tiene capacidad semantica suficiente (16.4%, peor
> que azar)."~~ — EXP-012, EXP-013

Esta conclusion es **falsa**. El 16.4% era un artefacto del protocolo,
no una medida de capacidad semantica.

### Conclusiones revisadas

> Llama3.2 3B tiene capacidad semantica limitada pero real (~58%
> single con protocolo corregido). Su rendimiento esta por debajo del
> criterio PM-003 (60%) pero dentro del rango de modelos 3B
> comparables (Granite 61.8% single). No puede descartarse como
> "insuficiente" sin re-evaluar con el protocolo corregido en
> configuraciones de ensemble.

> La exclusion de Llama3.2 del Micro-Coliseum (EXP-014) fue prematura.
> Debe re-evaluarse con el protocolo corregido antes de excluirlo de
> futuros experimentos de deliberacion.

### Leccion arquitectural

El protocolo de evaluacion (num_predict, think, format, parser) es
parte del experimento, no infraestructura neutral. Un cambio en
cualquiera de estos parametros puede producir deltas de +40 puntos de
accuracy. Los experimentos deben:

1. **Documentar el protocolo completo** (num_predict, think, format,
   parser, fallbacks) como parte de la configuracion experimental.
2. **Conservar `raw`, `eval_count`, `done_reason`** por caso para
   detectar truncamientos post-hoc.
3. **Usar parsers estrictos** que no defaulteen; marcar
   `PROTOCOL_ERROR` en lugar de inferir una relacion.
4. **Verificar `done_reason`** en pilotos: si hay casos con
   `done_reason=length`, el presupuesto de tokens es insuficiente.
5. **Desactivar `think`** explicitamente para modelos con razonamiento,
   a menos que el razonamiento sea parte del experimento.

## Acciones tomadas

1. `NoThinkOllamaModelProvider` creado en `run_coliseo_v2_gpu.py` para
   Nemotron, Ministral, Qwen3.5.
2. `run_coliseo_v1_llama32_cpu_controlled.py` creado con protocolo
   corregido para Llama3.2.
3. `run_ministral_coliseo_v2.py` creado con schema JSON simplificado
   para Ministral.
4. POST-001 emitido.
5. EXP-012 y EXP-013 actualizados con advertencia de contaminacion.
6. README actualizado.

## Acciones pendientes

1. ~~Completar la repeticion de Llama3.2 (ensemble_2, ensemble_4).~~
   **Completado**: single=58.2%, ensemble_2=58.2%, ensemble_4=63.6%.
2. Re-evaluar Nemotron, Ministral, Qwen3.5 con protocolo corregido
   en Coliseo v2 completo (no solo fix aislado).
3. Considerar re-ejecutar Coliseo v1 completo (3 modelos) con
   protocolo corregido para tener comparacion limpia.
4. Re-evaluar inclusion de Llama3.2 en Micro-Coliseum (EXP-014).
5. ~~Re-evaluar BitNet en Coliseo v1 (55 casos, protocolo corregido).~~
   **Completado (EXP-015)**: BitNet NO fue contaminado. 29.1% single
   en 55 casos confirma PM-003. La condena se sostiene.
