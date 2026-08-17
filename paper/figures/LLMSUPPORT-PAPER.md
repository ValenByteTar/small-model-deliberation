# Hacia un LLMSupport Local: Evaluación Sistemática de Modelos LLM Pequeños para Semantic Assessment en un Pipeline Agentic RAG

## Abstract

Este documento reporta la investigación completa realizada en el proyecto `small-model-deliberation` con el objetivo de construir un **LLMSupport**: un componente de soporte semántico local, de bajo costo, que asista al pipeline principal de Agentic RAG en la evaluación de relaciones entre claims y evidencia. La motivación original incluía la posibilidad de evolucionar hacia un **Mixture-of-Experts (MoE)** si se presentaba una serie interesante de candidatos con capacidades complementarias. Evaluamos 9 modelos LLM de entre 2B y 4B parámetros bajo un protocolo experimental controlado de micro-coliseo deliberativo, con 55 casos balanceados en 10 categorías semánticas. **Qwen3.5 4B (Q4_K_M)** fue el mejor candidato evaluado bajo el protocolo experimental actual, alcanzando 85.5% en modo independiente y 89.1% bajo deliberación. Documentamos los fixes técnicos necesarios para cada modelo, el hallazgo metodológico de contaminación por protocolo (POST-001), y el cierre de BitNet b1.58-2B-4T como candidato tras 8 experimentos progresivos que demostraron ausencia de señal semántica discriminativa en su distribución de probabilidades. La conclusión arquitectónica central es que un modelo pequeño puede aportar señales útiles para Semantic Assessment local, pero ni el output correcto ni una mejora de accuracy constituyen evidencia suficiente de que la arquitectura de soporte esté produciendo la señal semántica correcta. Por ello, LLMSupport queda subordinado a contratos y policy determinística.

---

## 1. Introducción

### 1.1 Motivación: el problema del Semantic Assessment

Un pipeline de Agentic RAG que evalúa afirmaciones contra evidencia recuperada necesita realizar **Semantic Assessment**: determinar si una pieza de evidencia soporta, contradice, parcialmente soporta, o no está relacionada con un claim. Las cuatro relaciones canónicas son:

```
SUPPORTS     — la evidencia demuestra el claim
PARTIAL      — la evidencia soporta parte del claim pero no todo
CONTRADICTS  — la evidencia es incompatible con el claim
UNRELATED    — la evidencia no trata sobre el mismo tema/contexto
```

Esta evaluación debe ocurrir múltiples veces por consulta. Usar un modelo grande (7B+) para cada evaluación es costoso en un entorno local con hardware limitado (6 GB VRAM). La pregunta central de esta investigación fue: **¿existe un modelo pequeño (2B-4B) que pueda producir señales útiles para Semantic Assessment dentro de un pipeline donde la autoridad decisional permanece en el sistema?**

### 1.2 Visión arquitectónica: LLMSupport y MoE

La visión original (RES-016) planteaba una arquitectura donde múltiples modelos pequeños especializados podrían componerse:

```
                Semantic Assessment
                       │
             ┌─────────┼─────────┐
             │         │         │
        Relevance  Entailment  Granularity
             │         │         │
          Modelo A   Modelo B   Modelo C
             │         │         │
             └─────────┼─────────┘
                       │
              deterministic policy
```

Si los modelos demostraban capacidades complementarias — por ejemplo, un modelo bueno en detección de contradicciones y otro bueno en detección de relevancia — la arquitectura **podría** ser un MoE de expertos semánticos, donde un router determinístico enviaría cada caso al especialista apropiado. Sin embargo, esta investigación demostró experimentalmente una regla que se aplica también al MoE: **no convertir una hipótesis arquitectónica en arquitectura antes de demostrar la señal que la justifica.** EXP-023 y EXP-024 descubrieron exactamente este error con BitNet: confundir un comportamiento observado (TRUE x55 bajo grammar estricto) con una capacidad arquitectónica demostrada (entailment). El MoE queda como hipótesis pendiente de validación fuera de muestra.

El componente **LLMSupport** (ADR-0031) fue propuesto como fase inicial: un observador paralelo pasivo que genera hipótesis sin bloquear el pipeline. BitNet b1.58-2B-4T era el candidato ideal para este rol por su tamaño (~1.1 GB RAM) y su capacidad de correr en CPU sin competir con el modelo principal en GPU.

### 1.3 Contribuciones de este trabajo

1. **Evaluación sistemática de 9 modelos** bajo protocolo controlado de micro-coliseo deliberativo
2. **Documentación de fixes técnicos** necesarios para cada modelo en infraestructura local
3. **Hallazgo metodológico de contaminación por protocolo** (POST-001): `num_predict=10` + `think` mode + parser permisivo produjeron mediciones falsas en 4 modelos
4. **Cierre de BitNet** tras 8 experimentos (EXP-017 a EXP-024) que demostraron ausencia de señal semántica discriminativa
5. **Identificación de Qwen3.5 4B** como el mejor candidato evaluado bajo el protocolo actual
6. **Evidencia de heterogeneidad de performance por categoría** entre Qwen3.5 4B y Qwen3 4B RAG, compatible con una hipótesis de especialización (no demostrada como caso MoE)

### 1.4 Limitaciones del protocolo experimental

Las conclusiones de este trabajo están acotadas por las siguientes variables no controladas:

- **Un solo benchmark** (55 casos, dominio ciberseguridad/NIST)
- **N pequeño por categoría** (5-6 casos por categoría; un solo error cambia 100% a 83.3%)
- **Un único dominio** (no hay evaluación out-of-domain)
- **Seed limitada** (no hay evaluación de estabilidad entre corridas)
- **Una sola familia de prompts** (no se evaluó sensibilidad al prompt template)
- **No hay benchmark adversarial independiente**
- **No hay comparación contra un 7B** bajo exactamente el mismo protocolo
- **No hay evaluación del router** en el supuesto MoE

Por lo tanto, las formulaciones de este documento distinguen entre **"mejor candidato evaluado"** (sí) y **"mejor modelo pequeño para semantic assessment"** (todavía no verificable con la evidencia disponible).

---

## 2. Infraestructura Experimental

### 2.1 Hardware

- **GPU:** 6 GB VRAM (restricción crítica para selección de modelos)
- **CPU:** multi-core, usado para BitNet y experimentos controlados
- **RAM:** suficiente para múltiples instancias de modelos cuantizados

### 2.2 Software

- **Ollama** como backend principal para modelos 3B-4B (API `/api/generate`)
- **llama-server (bitnet.cpp)** como backend para BitNet b1.58-2B-4T (API `/completion` compatible con OpenAI)
- **Multi-instancia Ollama** (PAT-001): instancias separadas en puertos diferentes con modelos pinneados para evitar model thrashing en experimentos paralelos

### 2.3 Benchmark

`semantic_assessment_v2.json`: 55 casos balanceados en 10 categorías:

| Categoría | N | Relación esperada dominante |
|-----------|---|-----------------------------|
| direct_evidence | 6 | SUPPORTS |
| paraphrase | 6 | SUPPORTS |
| partial_support | 6 | PARTIAL |
| over_specificity | 5 | PARTIAL |
| explicit_contradiction | 5 | CONTRADICTS |
| implicit_contradiction | 5 | CONTRADICTS |
| negation | 6 | CONTRADICTS / PARTIAL |
| wrong_subject | 5 | UNRELATED |
| wrong_context | 5 | UNRELATED |
| adversarial | 6 | PARTIAL / CONTRADICTS |

**Resolución estadística:** con N=55, cada caso representa 1.82 puntos porcentuales. Diferencias de 2-4 puntos (como las observadas entre independent y debate) corresponden a 1-2 casos y deben interpretarse con cautela.

### 2.4 Micro-coliseo deliberativo

Cada modelo se evaluó bajo tres configuraciones:

1. **Independent**: 4 workers con roles especializados (entailment, skeptical, contradiction, context) evalúan independientemente; voting determina el resultado
2. **Debate-on-disagreement**: si los workers no están de acuerdo, se inicia un debate estructurado
3. **Debate-all**: todos los casos pasan por debate, independientemente del acuerdo inicial

Los 4 roles especializados son:

| Worker | Rol | Prompt focus |
|--------|-----|-------------|
| A | Entailment | ¿La evidencia demuestra el claim? |
| B | Skeptical | ¿Hay información ausente o no verificada? |
| C | Contradiction | ¿La evidencia contradice el claim? |
| D | Context | ¿El claim y la evidencia tratan sobre el mismo tema? |

---

## 3. Modelos Evaluados

### 3.1 Inventario completo

| Modelo | Parámetros | Cuantización | Backend | Tipo |
|--------|-----------|-------------|---------|------|
| BitNet-b1.58-2B-4T | 2B | 1.58 bits (ternario) | llama-server | Cuantización extrema |
| Llama 3.2 3B | 3B | Q4_K_M | Ollama | Estándar |
| Granite 4.1 3B | 3B | Q4_K_M | Ollama | Estándar |
| Gemma 3 4B | 4B | Q4_K_M | Ollama | Estándar |
| Qwen3 4B Base | 4B | Q4_K_M | Ollama | Estándar |
| Qwen3 4B RAG | 4B | Q4_K_M | Ollama | Fine-tuned RAG |
| Qwen3.5 4B | 4B | Q4_K_M | Ollama | Razonamiento |
| Nemotron-3 Nano 4B | 4B | Q4_K_M | Ollama | Razonamiento |
| Ministral 3B | 3B | Q4_K_M | Ollama | Razonamiento |

### 3.2 Fixes técnicos por modelo

Cada modelo requirió ajustes específicos para funcionar correctamente bajo el protocolo experimental. La omisión de estos fixes produjo mediciones falsas que podrían haberse interpretado como incapacidad del modelo.

#### 3.2.1 BitNet b1.58-2B-4T: fix de tokenizer

**Problema:** llama-server no reconocía correctamente el tokenizer del modelo BitNet, produciendo outputs incoherentes.

**Fix:** Override del tokenizer predecessor en todos los runners de BitNet:

```python
"--override-kv", "tokenizer.ggml.pre=str:llama3"
```

Este fix se aplicó en 13 archivos de runner diferentes. Sin este parámetro, el modelo no produce texto coherente.

**Resultado:** El modelo funciona y produce outputs, pero como se documenta en la Sección 5, la capacidad semántica resultante es inexistente.

#### 3.2.2 Llama 3.2 3B: fix de contaminación por protocolo (POST-001)

**Problema:** Accuracy histórica de 16.4%, aparentemente indicando capacidad insuficiente. La causa real era contaminación por protocolo con tres defectos sinérgicos:

1. `num_predict=10`: presupuesto de tokens insuficiente para que el modelo emitiera la clasificación antes de ser truncado
2. `think` mode activo por defecto: el modelo consumía tokens en razonamiento interno antes de emitir output visible
3. Parser permisivo que defaulteaba a `UNRELATED` cuando el output estaba vacío o truncado, enmascarando el problema

**Fix:**

```python
NUM_PREDICT = 64              # expandido desde 10
# + think=false
# + format=json con schema estricto
# + parser estricto que retorna PROTOCOL_ERROR en vez de defaultear a UNRELATED
```

**Resultado:** 16.4% → **58.2%** single, **63.6%** ensemble_4. Un delta de +41.8 puntos porcentuales. La conclusión anterior de "capacidad insuficiente" era falsa: el modelo era competente pero el protocolo lo truncaba.

#### 3.2.3 Qwen3.5 4B, Nemotron-3 Nano 4B, Ministral 3B: fix de think mode

**Problema:** Estos tres modelos son modelos de razonamiento que activan `think` mode por defecto en Ollama. Con `num_predict` limitado, el razonamiento interno consume todo el presupuesto de tokens antes de emitir output visible.

**Fix:** Implementación de `NoThinkOllamaModelProvider` que desactiva explícitamente el razonamiento:

```python
class NoThinkOllamaModelProvider(OllamaModelProvider):
    def generate(self, prompt, *, options=None, timeout=None):
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "think": False,    # fix: desactivar razonamiento
                "options": opts,
                "keep_alive": "10m",
            },
            ...
        )
```

Detección automática: cualquier modelo cuyo nombre contenga "nemotron", "ministral", o "qwen3.5" usa automáticamente este provider.

**Resultado para Qwen3.5:** de 0% (con think activo) a **85.5%** (con think desactivado).

#### 3.2.4 Ministral 3B: fix adicional de schema simplificado

**Problema:** Además del think mode, Ministral no producía JSON válido con el schema completo (que incluía `relation` + `confidence`).

**Fix:** Schema simplificado con solo el campo `relation`:

```python
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "relation": {"type": "string", "enum": list(RELATIONS)},
    },
    "required": ["relation"],
    "additionalProperties": False,
}
```

**Resultado:** 72.7% independent. Sin embargo, Ministral sufre daño catastrófico con debate (-12.7%, net -8 damage), un comportamiento outlier.

#### 3.2.5 Qwen3 4B Base: fix de CUDA OOM

**Problema:** El modelo base de Qwen3 4B causaba CUDA Out of Memory con el contexto por defecto de Ollama (40960 tokens), que intenta allocar ~24GB de KV cache.

**Fix:** Limitar el contexto a 4096 tokens:

```python
"num_ctx": 4096    # previene CUDA OOM en 6GB VRAM
```

#### 3.2.6 Granite 4.1 3B: estrategia de cuantización

**Problema:** La cuantización Q4 producía comportamiento errático: 20% de chunks vacíos, 5% de parse errors, relaciones básicas en lugar de ricas.

**Fix:** Migración de Q4_K_M a Q6_K como default. Q6 elimina los chunks vacíos y parse errors, con mejor calidad de relaciones, manteniéndose dentro de los 6GB VRAM.

#### 3.2.7 Multi-instancia Ollama (PAT-001): fix de model thrashing

**Problema:** Con `OLLAMA_MAX_LOADED_MODELS=1` (default), múltiples experimentos paralelos competían por la misma instancia de Ollama. Cada request disparaba unload/load del modelo, produciendo inestabilidad.

**Fix:** Patrón de multi-instancia con modelos pinneados:

```bash
# Instancia CPU en puerto 11435 con llama3.2:3b
python runners/ollama_instances.py start --port 11435 --cpu --model llama3.2:3b

# Instancia GPU en puerto 11436 con gemma3:4b
python runners/ollama_instances.py start --port 11436 --gpu --model gemma3:4b-it-q4_K_M
```

Cada instancia tiene PID, puerto, asignación de GPU/CPU y modelo pinneado independientes. El estado se persiste en `runners/.ollama-instances.json`.

### 3.3 Tabla resumen de fixes

| Modelo | Problema | Fix | Impacto |
|--------|----------|-----|---------|
| BitNet 2B | Tokenizer incompatibilidad | `--override-kv tokenizer.ggml.pre=str:llama3` | Habilita output coherente |
| Llama 3.2 3B | Contaminación por protocolo | `num_predict=64` + `think=false` + JSON + parser estricto | 16.4% → 58.2% (+41.8%) |
| Qwen3.5 4B | Think mode consume tokens | `think=false` | 0% → 85.5% |
| Nemotron-3 4B | Think mode consume tokens | `think=false` | 0% → 60.0% |
| Ministral 3B | Think mode + schema complejo | `think=false` + schema simplificado | 0% → 72.7% |
| Qwen3 4B Base | CUDA OOM | `num_ctx=4096` | Previene crash |
| Granite 3B | Q4 errático | Migración a Q6 | 0% empty chunks, mejor calidad |
| Todos | Model thrashing en paralelo | Multi-instancia Ollama (PAT-001) | Estabilidad experimental |

---

## 4. Resultados: Comparación de Modelos

### 4.1 Accuracy por modelo y configuración

Los resultados se reportan sobre el benchmark de 55 casos. Los datos de Qwen3.5 en `debate-on-disagreement` incluyen una corrección manual: el caso `pp-001` dio timeout durante la corrida pero el modelo había acertado (ground truth: SUPPORTS, output correcto: SUPPORTS). Este caso se corrige a correcto en los resultados reportados.

| Modelo | Independent | Debate-on-disagreement | Debate-all | Mejor config |
|--------|------------|----------------------|------------|-------------|
| **Qwen3.5 4B** | **85.5%** (47/55) | **89.1%** (49/55) | **89.1%** (49/55) | **Debate** |
| Qwen3 4B RAG | 76.4% (42/55) | 76.4% (42/55) | 74.6% (41/55) | Independent |
| Ministral 3B | 72.7% (40/55) | 60.0% (33/55) | 60.0% (33/55) | Independent |
| Granite 4.1 3B | 69.1% (38/55) | 69.1% (38/55) | 80.0% (44/55) | Debate-all |
| Qwen3 4B Base | 65.5% (36/55) | 65.5% (36/55) | 69.1% (38/55) | Debate-all |
| Llama 3.2 3B | 63.6% (35/55) | 60.0% (33/55) | 65.5% (36/55) | Debate-all |
| Nemotron-3 4B | 60.0% (33/55) | 61.8% (34/55) | 61.8% (34/55) | Debate |
| Gemma 3 4B | 52.7% (29/55) | 63.6% (35/55) | 56.4% (31/55) | Debate-on-disagr. |
| BitNet 2B | 27.3% (15/55) | 0.0% (0/55) | — | Independent |

**Nota sobre resolución:** con N=55, cada caso representa 1.82 puntos. La diferencia entre Qwen3.5 independent (47/55) y debate (49/55) son **2 casos**. Esto se reporta como "el debate recuperó 2 casos sobre 55 en esta corrida", no como evidencia estadística fuerte.

### 4.2 Gráfica: ranking de candidatos evaluados

```
Accuracy en Semantic Assessment (55 casos, mejor configuración por modelo)

Qwen3.5 4B    ████████████████████████████████████████████████████████████████ 89.1% (49/55)  ★
Granite 4.1   ████████████████████████████████████████████████████████████░░░ 80.0% (44/55)  (debate-all)
Qwen3 4B RAG  ██████████████████████████████████████████████████████████░░░░░ 76.4% (42/55)
Ministral 3B  ███████████████████████████████████████████████████████░░░░░░░ 72.7% (40/55)  (independent)
Qwen3 4B Base ████████████████████████████████████████████████████████░░░░░░ 69.1% (38/55)  (debate-all)
Llama 3.2 3B  ███████████████████████████████████████████████████████░░░░░░░ 65.5% (36/55)  (debate-all)
Nemotron-3 4B ██████████████████████████████████████████████████████░░░░░░░░ 61.8% (34/55)  (debate)
Gemma 3 4B    █████████████████████████████████████████████████████░░░░░░░░░ 63.6% (35/55)  (debate-on-disagr.)
BitNet 2B     ████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 27.3% (15/55)

              └────────────────────────────────────────────────────────────┘
              0%                          45%                         100%

              ★ = Mejor candidato evaluado bajo el protocolo actual
              Umbral mínimo observado: ~60%
              Nota: no implica "mejor modelo pequeño para semantic assessment" (ver §1.4)
```

### 4.3 Accuracy por categoría (top 4 modelos, modo independent)

| Categoría | N | Qwen3.5 4B | Qwen3 4B RAG | Ministral 3B | Granite 3B |
|-----------|---|-----------|-------------|-------------|-----------|
| direct_evidence | 6 | **100%** (6/6) | **100%** (6/6) | 83.3% (5/6) | 83.3% (5/6) |
| paraphrase | 6 | **100%** (6/6) | 66.7% (4/6) | **100%** (6/6) | 66.7% (4/6) |
| explicit_contradiction | 5 | **100%** (5/5) | **100%** (5/5) | **100%** (5/5) | **100%** (5/5) |
| implicit_contradiction | 5 | **100%** (5/5) | 80.0% (4/5) | **100%** (5/5) | 60.0% (3/5) |
| negation | 6 | **100%** (6/6) | 83.3% (5/6) | **100%** (6/6) | 66.7% (4/6) |
| wrong_subject | 5 | **100%** (5/5) | 40.0% (2/5) | 20.0% (1/5) | 80.0% (4/5) |
| over_specificity | 5 | 60.0% (3/5) | **100%** (5/5) | 60.0% (3/5) | 60.0% (3/5) |
| partial_support | 6 | 66.7% (4/6) | **100%** (6/6) | 83.3% (5/6) | 66.7% (4/6) |
| wrong_context | 5 | 40.0% (2/5) | 0.0% (0/5) | 0.0% (0/5) | 60.0% (3/5) |
| adversarial | 6 | 83.3% (5/6) | 83.3% (5/6) | 66.7% (4/6) | 50.0% (3/6) |

**Cautela sobre N por categoría:** cada categoría tiene 5-6 casos. Un solo error convierte 100% en 83.3% (6/6→5/6) o 80% (5/5→4/5). Los porcentajes por categoría tienen resolución gruesa y deben interpretarse como evidencia direccional, no como estimaciones robustas de capacidad.

### 4.4 Heterogeneidad de performance entre Qwen3.5 y Qwen3 RAG

La tabla anterior revela perfiles diferentes entre Qwen3.5 4B y Qwen3 4B RAG:

| Capacidades | Qwen3.5 4B | Qwen3 4B RAG | Diferencia |
|------------|-----------|-------------|-----------|
| direct_evidence | 100% (6/6) | 100% (6/6) | Ambos fuertes |
| paraphrase | 100% (6/6) | 66.7% (4/6) | Qwen3.5 superior |
| explicit_contradiction | 100% (5/5) | 100% (5/5) | Ambos fuertes |
| implicit_contradiction | 100% (5/5) | 80.0% (4/5) | Qwen3.5 superior |
| negation | 100% (6/6) | 83.3% (5/6) | Qwen3.5 superior |
| wrong_subject | 100% (5/5) | 40.0% (2/5) | Qwen3.5 muy superior |
| **over_specificity** | **60.0% (3/5)** | **100% (5/5)** | **Qwen3 RAG superior** |
| **partial_support** | **66.7% (4/6)** | **100% (6/6)** | **Qwen3 RAG superior** |
| wrong_context | 40.0% (2/5) | 0.0% (0/5) | Ambos débiles |
| adversarial | 83.3% (5/6) | 83.3% (5/6) | Ambos aceptables |

**Esto es evidencia de heterogeneidad de performance por categoría, compatible con una hipótesis de especialización.** No es evidencia de "caso MoE genuino". Tres problemas impiden esa conclusión:

1. **N extremadamente pequeño por categoría:** partial_support tiene 6 casos. 100% significa 6/6. Un solo error cambia eso a 83.3%. No alcanza para establecer una capacidad especializada robusta.

2. **La categoría puede estar correlacionada con el dataset:** Qwen3 RAG puede acertar estos 6 ejemplos específicos sin que necesariamente exista una capacidad general de "mejor en partial support".

3. **Falta evaluar el router:** el verdadero objeto arquitectónico no es "Qwen3.5 vs Qwen3 RAG" sino "Router → modelo seleccionado → resultado final". El router puede equivocarse, y su error no se ha medido.

El techo teórico de ~92% "si el router fuera perfecto" es matemáticamente interesante pero arquitectónicamente poco útil sin evaluar el sistema completo:

```
                    Query
                      │
                      ▼
                Router
                 /    \
                /      \
               ▼        ▼
          Qwen3.5    Qwen3 RAG
               \      /
                \    /
                 ▼  ▼
              Evidence
              assessment
```

El verdadero experimento futuro debería medir el sistema completo (router + modelos + aggregation), no solo los componentes individuales.

### 4.5 Efecto del debate deliberativo

| Modelo | Independent | Mejor debate | Casos recuperados | Casos dañados |
|--------|------------|-------------|-------------------|--------------|
| **Qwen3.5 4B** | 47/55 (85.5%) | **49/55 (89.1%)** | **2** | **0** |
| Granite 4.1 3B | 38/55 (69.1%) | 44/55 (80.0%) | 6 | — |
| Gemma 3 4B | 29/55 (52.7%) | 35/55 (63.6%) | 6 | — |
| Nemotron-3 4B | 33/55 (60.0%) | 34/55 (61.8%) | 1 | 0 |
| Qwen3 4B Base | 36/55 (65.5%) | 38/55 (69.1%) | 2 | — |
| Llama 3.2 3B | 35/55 (63.6%) | 36/55 (65.5%) | 1 | — |
| Qwen3 4B RAG | 42/55 (76.4%) | 42/55 (76.4%) | 0 | 0 |
| Ministral 3B | 40/55 (72.7%) | 33/55 (60.0%) | 0 | **8** |
| BitNet 2B | 15/55 (27.3%) | 0/55 (0.0%) | 0 | — |

**Formulación correcta de los resultados de debate:**

Para Qwen3.5 4B: el debate produjo una mejora observada de 2/55 casos en esta corrida, sin conversiones de correct→incorrect. Esto es interesante pero no constituye evidencia estadística fuerte (N=55, 2 casos = 3.6 puntos porcentuales).

**"Cero daño" no debe generalizarse:** es correcto para la corrida reportada de Qwen3.5, pero no implica "el debate no daña". Ministral 3B demuestra precisamente que el debate puede ser destructivo (-12.7%, 8 casos dañados).

**Conclusión sobre debate:** el debate no es monotónicamente beneficioso. Es otro componente que debe estar sometido a evidencia:

```
LLM
 ↓
deliberation
 ↓
¿mejoró?
```

No es una presuposición de "más reasoning = mejor". El debate es un mecanismo que puede agregar valor cuando el modelo base es competente y estable, pero puede destruir performance cuando el modelo cambia de opinión de forma inestable.

---

## 5. El Caso BitNet: Investigación Exhaustiva y Cierre

### 5.1 Motivación original

BitNet b1.58-2B-4T era el candidato ideal para LLMSupport por sus características únicas:

- **2B parámetros con pesos ternarios** {-1, 0, +1}: ~1.1 GB RAM
- **Corre en CPU** (4 threads): no compite con GPU del pipeline principal
- **Cuantización nativa 1.58 bits**: no es post-quantization, es entrenamiento ternario

Si BitNet tuviera capacidad semántica suficiente, podría correr como observador paralelo pasivo generando hipótesis sin costo de GPU.

### 5.2 Cadena experimental falsacionista

Realizamos 8 experimentos progresivos (EXP-017 a EXP-024). La cadena no fue "BitNet dio mal → descartemos BitNet", sino una progresión falsacionista donde cada experimento testó una hipótesis específica sobre la causa del mal desempeño:

| EXP | Hipótesis testada | Resultado | Siguiente pregunta |
|-----|-------------------|-----------|---------------------|
| EXP-017 | ¿Baseline con labels artificiales? | 29.1%, SUPPORTS wall (0/12) | ¿El problema es el label? |
| EXP-018 | ¿NLI reframing rompe la wall? | 40.0%, 12/12 SUPPORTS | ¿Es capacidad real o artefacto? |
| EXP-019 | ¿Relevance gate + ensemble? | 43.6% peak | ¿Es semántica o overlap lexical? |
| EXP-020 | ¿Atomic decomposition? | 33.3%, keyword matching | ¿Puede descomponer composicionalmente? |
| EXP-021 | ¿Detección de ausencia? | 0/12 FALSE | ¿Puede verificar negativamente? |
| EXP-022 | ¿Microcoliseum especializado? | 29.1% | ¿Specialization ayuda? |
| EXP-023 | ¿Grammar afecta comportamiento? | **Inversión completa** TRUE↔FALSE | ¿Es el grammar o la distribución? |
| EXP-024 | ¿Hay señal semántica en logprobs? | **Delta = +0.0017: NO** | Cierre |

Esta progresión convierte el rechazo en una **decisión de ingeniería defendible**: no se descartó BitNet por dar mal, sino porque se agotaron las hipótesis sobre por qué daba mal y la última (ausencia de señal en la distribución) se confirmó.

### 5.3 Hallazgo metodológico: grammar como variable de primer orden (EXP-023)

EXP-023 descubrió que el grammar GBNF de decodificación es una variable experimental de primer orden para BitNet, capaz de invertir completamente el comportamiento:

| Condición | Token dominante | Distribución |
|-----------|-----------------|--------------|
| Grammar estricto | `TRUE` x55 | 55/55 SUPPORTS |
| Grammar permisivo | `False` x40, `FALSE` x13 | 53/55 CONTRADICTS |
| Sin grammar | `False` x39, `FALSE` x13 | 53/55 CONTRADICTS |

El mecanismo es la tokenización con espacio: `"TRUE"` (sin espacio) y `" FALSE"` (con espacio) son tokens diferentes con probabilidades diferentes. El grammar estricto fuerza el token sin espacio, cambiando la preferencia del modelo.

Esto significó que los 12/12 SUPPORTS de EXP-018 — interpretados inicialmente como "BitNet tiene entailment" — eran en realidad un artefacto del grammar estricto forzando TRUE universal.

### 5.4 Resultado definitivo: no hay señal semántica discriminativa (EXP-024)

EXP-024 controló la variable de grammar y midió directamente si la distribución de probabilidades cambia según la relación semántica, usando 20 pares mínimos (mismo evidence, claims que difieren en un elemento semántico):

| Expected | P(TRUE) | P(FALSE) | P(CANNOT) |
|----------|---------|----------|-----------|
| SUPPORTS | 0.4000 | 0.6000 | 0.0000 |
| CONTRADICTS | 0.3983 | 0.6017 | 0.0000 |
| PARTIAL | 0.4668 | 0.5332 | 0.0000 |
| UNRELATED | 0.3941 | 0.6059 | 0.0000 |

**Delta P(TRUE|SUPPORTS) - P(TRUE|CONTRADICTS) = +0.0017**

La probabilidad de TRUE es idéntica independientemente de la relación semántica. BitNet no discrimina semánticamente. No es "parcialmente sabiendo pero no podemos extraerlo": la distribución es plana respecto a la semántica.

### 5.5 La excepción: lexical separability

El 100% de relevance detection en casos claros (EXP-019: wrong_subject) sigue siendo un resultado medido. Pero la interpretación correcta es:

**BitNet puede responder diferencialmente cuando existe una separación lexical extremadamente fuerte entre los dos textos** (Product A vs Product B, framework X vs framework Y). Si la evidencia comparte poco o ningún vocabulario relevante con el claim, BitNet produce una respuesta diferente.

Esto **no constituye evidencia de semantic relevance general**. Es compatible con:

```
overlap alto → una respuesta (TRUE o FALSE, según grammar)
overlap bajo  → otra respuesta
```

Por lo tanto, esto se clasifica como **lexical separability**, no como **semantic relevance detection**. EXP-024 confirma esto: en los pares mínimos de subject donde el overlap es alto pero la entidad difiere (NIST vs ISO 27001), la distribución P(TRUE)/P(FALSE) no cambia.

### 5.6 Mapa final de comportamiento de BitNet

```
                    BitNet 1.58B
                         │
              ┌──────────┴──────────┐
              │                     │
       comportamiento         capacidades
        observable             semánticas
              │                     │
      ┌───────┼────────┐            │
      │       │        │            │
    TRUE    FALSE    grammar      ¿señal?
      │       │        │            │
      │       │        │           NO
      │       │        │            │
      └───────┴────────┴────────────┘
                         │
                  lexical behavior
                         │
              ┌──────────┴──────────┐
              │                     │
       alto overlap             bajo overlap
              │                     │
        respuestas             separación
        sesgadas               aparente
```

### 5.7 Veredicto arquitectónico para BitNet

```
BitNet
   │
   X Semantic Assessment
   X Judge
   X Worker
   X Evidence classifier
   X MoE semantic expert
   │
   └── descartado
```

BitNet queda afuera no porque "sea demasiado chico" sino porque **no demostró producir la señal que el contrato necesita**. La distribución de probabilidades no contiene señal semántica discriminativa explotable.

---

## 6. Tres Niveles de Análisis: Capability, Signal, Authority

El proyecto descubrió que es necesario distinguir tres niveles diferentes al evaluar un modelo para Semantic Assessment:

### Nivel 1 — Capability

¿El modelo puede producir una clasificación correcta?

```
Qwen3.5:    YES — 85.5% de los casos
BitNet:     aparentemente ocasionalmente, pero no de forma confiable
            y no por discriminación semántica (EXP-024)
```

### Nivel 2 — Signal

¿Existe una señal correlacionada con la propiedad que queremos detectar?

```
Qwen3.5:    probablemente YES — la heterogeneidad por categoría sugiere
            que el modelo responde a propiedades semánticas del input
BitNet:     NO EVIDENCE — la distribución P(TRUE)/P(FALSE) es plana
            respecto a la relación semántica (delta = +0.0017)
```

### Nivel 3 — Authority

¿El modelo puede decidir?

```
Respuesta arquitectónica: NO.

Incluso Qwen3.5 no debería tener autoridad decisional.
La autoridad permanece en el sistema (policy determinística).
```

Esto es conceptualmente mucho más sólido que "LLM judge → decision":

```
                Evidence
                    │
                    ▼
             LLM Support
                    │
             hypotheses/signals
                    │
                    ▼
          deterministic policy
                    │
                    ▼
                decision
```

La distinción entre estos tres niveles es el verdadero avance conceptual del proyecto. BitNet fue rechazado en el Nivel 2 (no produce señal), no solo en el Nivel 1 (no clasifica correctamente). Qwen3.5 pasa el Nivel 1 y probablemente el Nivel 2, pero el Nivel 3 queda fuera de su alcance por diseño arquitectónico.

---

## 7. Evolución Arquitectónica

### 7.1 De RAG a arquitectura con contratos determinísticos

El proyecto refleja una evolución conceptual clara:

**RAG original:**
```
retrieve → LLM → answer
```

**Agentic RAG:**
```
Planner → retrieval → evidence → assessment → policy → answer
```

**Arquitectura actual:**
```
                 ┌───────────────┐
                 │ Deterministic │
                 │   Contracts   │
                 └───────┬───────┘
                         │
                         ▼
Query ──► Retrieval ──► Evidence
                         │
                         ▼
                  ┌──────────────┐
                  │  LLMSupport  │
                  │              │
                  │ Qwen3.5      │
                  │ Qwen3 RAG    │
                  └──────┬───────┘
                         │
                       signals
                         │
                         ▼
                  Deterministic
                     Policy
                         │
                         ▼
                     Decision
```

### 7.2 Por qué esta arquitectura

La separación signal → policy → decision resuelve tres problemas:

1. **Trazabilidad:** cada decisión se puede auditar. El LLM produce una señal; la policy produce la decisión. Si la decisión es incorrecta, se puede determinar si el error estuvo en la señal o en la policy.

2. **Estabilidad:** la policy determinística no cambia entre corridas. El LLM puede variar, pero la policy agrega de forma reproducible.

3. **Ownership:** el sistema es responsable de la decisión, no el LLM. Esto alinea con el principio de "determinismo en control, razonamiento en lenguaje" (P9).

BitNet queda afuera de esta arquitectura no por tamaño sino por falta de señal. Qwen3.5 entra como productor de señales, no como juez.

---

## 8. Discusión

### 8.1 Qwen3.5 4B: mejor candidato evaluado, no "mejor modelo"

Qwen3.5 4B fue el mejor candidato evaluado bajo el protocolo experimental actual. Domina en 6 de 10 categorías y es el único modelo que alcanza 100% en contradicciones (explícita e implícita), negación, y wrong_subject simultáneamente.

Sin embargo, las siguientes variables no controladas impiden generalizar a "mejor modelo pequeño para semantic assessment":

- Un solo benchmark, un solo dominio
- N=55 (resolución gruesa, 1.82 puntos por caso)
- Seed limitada (no hay evaluación de estabilidad entre corridas)
- Una sola familia de prompts
- No hay comparación contra 7B bajo el mismo protocolo
- No hay evaluación out-of-domain

### 8.2 El supuesto MoE: hipótesis, no conclusión

La heterogeneidad de performance entre Qwen3.5 4B y Qwen3 4B RAG es evidencia compatible con una hipótesis de especialización. Pero:

1. **N pequeño por categoría** (5-6 casos): un solo error cambia 100% a 83.3%
2. **Posible correlación con el dataset**: Qwen3 RAG puede acertar estos ejemplos específicos sin capacidad general
3. **El router no se evaluó**: el techo teórico de ~92% asume un router perfecto, que no existe

El verdadero experimento futuro debería medir el sistema completo: router + modelos + aggregation, no solo los componentes individuales.

### 8.3 El debate: no monotónicamente beneficioso

El debate deliberativo no es una presuposición de "más reasoning = mejor". Es un componente sometido a evidencia:

- **Qwen3.5 4B:** recuperó 2/55 casos, 0 daños en esta corrida
- **Ministral 3B:** destruyó 8/55 casos, -12.7%

El debate agrega valor cuando el modelo base es competente y estable. Puede destruir performance cuando el modelo cambia de opinión de forma inestable. "Cero daño" es correcto para la corrida reportada de Qwen3.5 pero no debe generalizarse.

### 8.4 Lecciones metodológicas

1. **La contaminación por protocolo puede producir mediciones falsas** (POST-001). Llama 3.2 3B pasó de "incapaz" (16.4%) a "competente" (58.2%) con un fix de protocolo. Siempre verificar que el protocolo no esté truncando o enmascarando el output.

2. **El grammar de decodificación es una variable experimental de primer orden** para modelos cuantizados (EXP-023). Debe reportarse junto con temperature y seed.

3. **La distribución de probabilidades puede no contener señal incluso cuando el greedy parece funcionar** (EXP-024). BitNet parecía tener entailment (12/12 SUPPORTS) pero la distribución era plana respecto a la semántica.

4. **Los modelos de razonamiento requieren `think=false` explícito** en tareas de clasificación estructurada con presupuesto de tokens limitado.

5. **La separabilidad lexical no es discriminación semántica** (EXP-019 vs EXP-024). Un modelo puede detectar ausencia total de overlap sin comprender la relación semántica.

6. **Distinguir Capability, Signal y Authority** como tres niveles independientes de evaluación. Un modelo puede tener capability sin signal (BitNet bajo grammar estricto) o signal sin authority (Qwen3.5).

---

## 9. Conclusiones

### 9.1 Resultado principal

La evaluación experimental demostró que un modelo pequeño puede aportar señales útiles para Semantic Assessment local, pero también que ni el output correcto ni una mejora de accuracy constituyen evidencia suficiente de que una arquitectura de soporte esté produciendo la señal semántica correcta. Por ello, LLMSupport queda subordinado a contratos y policy determinística, mientras que BitNet fue rechazado después de demostrar ausencia de señal discriminativa bajo un protocolo específicamente diseñado para aislarla.

### 9.2 Mejor candidato evaluado

Qwen3.5 4B (Q4_K_M) fue el mejor candidato evaluado bajo el protocolo experimental actual, alcanzando 85.5% (47/55) en modo independiente y 89.1% (49/55) bajo deliberación. La mejora del debate fue de 2/55 casos en esta corrida, sin conversiones de correct→incorrect. Esta conclusión está acotada por las limitaciones enumeradas en §1.4.

### 9.3 Hipótesis MoE

Existe heterogeneidad de performance por categoría entre Qwen3.5 4B y Qwen3 4B RAG, compatible con una hipótesis de especialización. Esto no constituye un caso MoE demostrado: se requiere evaluar el sistema completo (router + modelos + aggregation) con un benchmark más grande y múltiples dominios.

El estado epistemológico correcto del MoE es:

```
                    LLMSupport
                        │
               ┌────────┴────────┐
               │                 │
          Qwen3.5 4B       Qwen3 RAG
               │                 │
            probado            probado
               │                 │
               └───────┬─────────┘
                       │
                 complementariedad
                       │
                       ▼
                HIPÓTESIS MoE
                       │
                       X
                todavía no validada
```

Esto evita repetir exactamente el error que EXP-023/024 descubrieron con BitNet: confundir un comportamiento observado con una capacidad arquitectónica demostrada.

### 9.4 Regla metodológica

Esta investigación no solamente encontró un candidato para LLMSupport. También produjo una regla metodológica para todo el Agentic RAG:

> **Primero demostrar la señal. Después asignarle ownership. Recién entonces construir la arquitectura alrededor de ella.**

BitNet fue el caso negativo: se asignó ownership arquitectónico (LLMSupport, observador paralelo, potencial MoE expert) antes de demostrar la señal. EXP-024 reveló que la señal no existía. El MoE de Qwen3.5 + Qwen3 RAG corre el mismo riesgo si se construye antes de validar la complementariedad fuera de muestra.

### 9.5 Cierre de BitNet

BitNet b1.58-2B-4T fue rechazado tras 8 experimentos que aislaron progresivamente las variables de confusión. EXP-024 demostró que la distribución de probabilidades no contiene señal semántica discriminativa (delta = +0.0017). El comportamiento observable es consistente con matching holístico por overlap lexical (lexical separability), no con discriminación semántica. El rechazo es una decisión de ingeniería defendible: se agotaron las hipótesis sobre la causa del mal desempeño y la última (ausencia de señal) se confirmó.

### 9.6 Arquitectura final propuesta

```
                 ┌───────────────┐
                 │ Deterministic │
                 │   Contracts   │
                 └───────┬───────┘
                         │
                         ▼
Query ──► Retrieval ──► Evidence
                         │
                         ▼
                  ┌──────────────┐
                  │  LLMSupport  │
                  │              │
                  │ Qwen3.5      │
                  │ (Qwen3 RAG?) │
                  └──────┬───────┘
                         │
                       signals
                         │
                         ▼
                  Deterministic
                     Policy
                         │
                         ▼
                     Decision
```

La autoridad permanece en el sistema (policy determinística). Los modelos LLM producen señales que la policy agrega. BitNet no participa. Qwen3 RAG entra como candidato a segundo experto solo si el router se evalúa y demuestra valor incremental.

---

## 10. Trabajo Futuro

1. **Evaluar el router MoE:** implementar y medir el sistema completo (router + Qwen3.5 + Qwen3 RAG + aggregation) contra los componentes individuales
2. **Benchmark más grande:** expandir a 200+ casos para reducir la resolución gruesa por categoría
3. **Múltiples dominios:** evaluar out-of-domain para verificar generalización
4. **Estabilidad entre corridas:** múltiples seeds para medir varianza
5. **Comparación contra 7B:** evaluar un modelo 7B bajo exactamente el mismo protocolo
6. **Benchmark adversarial independiente:** construido por un equipo diferente para evitar sesgo de diseño
7. **Sensibilidad al prompt:** evaluar múltiples familias de prompts

---

## 11. Artefactos Experimentales

| Artefacto | Descripción |
|-----------|-------------|
| `benchmarks/semantic_assessment_v2.json` | Benchmark principal (55 casos, 10 categorías) |
| `benchmarks/semantic_discrimination_v1.json` | Pares mínimos para EXP-024 (60 casos) |
| `runners/run_microcoliseum_all.py` | Runner de micro-coliseo para todos los modelos |
| `runners/run_coliseo_v2_gpu.py` | Coliseo v2 con NoThinkOllamaModelProvider |
| `runners/run_coliseo_v1_llama32_cpu_controlled.py` | Llama 3.2 con protocolo corregido |
| `runners/debate.py` | Implementación del debate deliberativo |
| `runners/lib/ollama_provider.py` | Provider base de Ollama |
| `runners/ollama_instances.py` | Multi-instancia Ollama (PAT-001) |
| `experiments/POST-001-protocol-contamination.md` | Documentación de contaminación por protocolo |
| `experiments/PAT-001-ollama-multi-instance.md` | Patrón de multi-instancia |
| `experiments/EXP-001-bitnet/PM-003-bitnet-semantic-capacity.md` | Cierre de BitNet |
| `experiments/EXP-001-bitnet/EXP-017` a `EXP-024` | Cadena experimental de BitNet |
| `experiments/EXP-001-bitnet/BITNET-PAPER-DRAFT.md` | Paper detallado de la investigación BitNet |
| `docs/adr-0031-llmsupport.md` | ADR de LLMSupport (deprecado para BitNet) |
| `docs/res-016-bitnet-vision.md` | Visión arquitectónica MoE + BitNet |
| `docs/res-007-model-strategy.md` | Estrategia de modelos (Granite Q4 vs Q6) |

---

## 12. Tabla de Experimentos por Modelo

| Modelo | EXP | Configuración | Accuracy | Notas |
|--------|-----|--------------|----------|-------|
| BitNet 2B | EXP-015 | single | 29.1% | Protocolo corregido |
| BitNet 2B | EXP-015 | ensemble_2 | 36.4% | |
| BitNet 2B | EXP-015 | ensemble_4 | 30.9% | |
| BitNet 2B | EXP-018 | NLI 3a + logit ensemble | 40.0% | Artefacto de grammar (EXP-023) |
| BitNet 2B | EXP-019 | relevance gate + ensemble | 43.6% | Lexical separability, no semántica |
| BitNet 2B | EXP-022 | microcoliseum especializado | 29.1% | |
| BitNet 2B | EXP-024 | semantic discrimination | delta=0.0017 | **No hay señal discriminativa** |
| Llama 3.2 3B | EXP-012 | single (contaminado) | 16.4% | POST-001 |
| Llama 3.2 3B | EXP-013 | single (corregido) | **58.2%** | +41.8% por fix de protocolo |
| Llama 3.2 3B | EXP-013 | ensemble_4 (corregido) | **63.6%** | |
| Llama 3.2 3B | EXP-016 | independent | 63.6% | |
| Llama 3.2 3B | EXP-016 | debate-all | 65.5% | |
| Granite 3B | EXP-011 | single (12 casos) | 91.7% | |
| Granite 3B | EXP-011 | ensemble (12 casos) | 100% | |
| Granite 3B | EXP-012 | single (55 casos) | 61.8% | |
| Granite 3B | EXP-012 | ensemble_4 (55 casos) | 76.4% | |
| Granite 3B | EXP-016 | independent | 69.1% | |
| Granite 3B | EXP-016 | debate-all | 80.0% | +6 casos recuperados |
| Gemma 3 4B | EXP-016 | independent | 52.7% | |
| Gemma 3 4B | EXP-016 | debate-on-disagr. | 63.6% | +6 casos recuperados |
| Qwen3 4B Base | EXP-016 | independent | 65.5% | |
| Qwen3 4B Base | EXP-016 | debate-all | 69.1% | |
| Qwen3 4B RAG | EXP-016 | independent | 76.4% | |
| Qwen3 4B RAG | EXP-016 | debate-all | 74.6% | -1 caso |
| Qwen3.5 4B | EXP-016 | independent | **85.5%** (47/55) | |
| Qwen3.5 4B | EXP-016 | debate-on-disagr. | **89.1%** (49/55) | +2 casos, 0 daños en esta corrida |
| Qwen3.5 4B | EXP-016 | debate-all | **89.1%** (49/55) | +2 casos, 0 daños en esta corrida |
| Nemotron-3 4B | EXP-016 | independent | 60.0% | |
| Nemotron-3 4B | EXP-016 | debate | 61.8% | +1 caso |
| Ministral 3B | EXP-016 | independent | 72.7% | |
| Ministral 3B | EXP-016 | debate | 60.0% | **-8 casos dañados** |

---

## 13. Referencias internas

- ADR-0031: LLMSupport Fase 1 (deprecated para BitNet)
- RES-016: Tutor LLM con MoE y BitNet (visión arquitectónica)
- RES-007: Estrategia de modelos (Granite Q4 vs Q6)
- POST-001: Contaminación por protocolo
- PAT-001: Multi-instancia Ollama
- PM-003: BitNet Semantic Capacity (REJECTED)
- EXP-017 a EXP-024: Cadena experimental BitNet
- EXP-016: Microcoliseum extendido (todos los modelos)

---

## Apéndice A: Configuración experimental estándar

```
Benchmark:       semantic_assessment_v2.json (55 casos, 10 categorías)
Workers:         4 (A=entailment, B=skeptical, C=contradiction, D=context)
Temperature:     0.0 (greedy)
num_predict:     60 (debate) / 10-64 (single, según modelo)
Modos:           independent / debate-on-disagreement / debate-all
Backend:         Ollama (modelos 3B-4B) / llama-server (BitNet)
GPU:             6 GB VRAM, num_gpu=99 para modelos Ollama
Multi-instancia: PAT-001 (puertos separados por modelo)
```

## Apéndice B: Glosario

| Término | Definición |
|---------|------------|
| **LLMSupport** | Componente de soporte semántico local que produce señales subordinadas a policy determinística |
| **Semantic Assessment** | Determinación de la relación entre un claim y una evidencia (SUPPORTS/PARTIAL/CONTRADICTS/UNRELATED) |
| **Capability** | Nivel 1: ¿el modelo puede producir una clasificación correcta? |
| **Signal** | Nivel 2: ¿existe una señal correlacionada con la propiedad semántica a detectar? |
| **Authority** | Nivel 3: ¿el modelo puede decidir? (Respuesta arquitectónica: NO) |
| **Micro-coliseo** | Arquitectura de 4 workers especializados que evalúan independientemente o debaten |
| **Debate deliberativo** | Mecanismo donde los workers discuten y revisan sus posiciones ante desacuerdos |
| **Contaminación por protocolo** | Mediciones falsas causadas por configuración experimental deficiente (POST-001) |
| **Lexical separability** | Capacidad de responder diferencialmente cuando dos textos no comparten vocabulario |
| **Semantic relevance detection** | Capacidad de determinar relación temática independientemente del overlap lexical |
| **Grammar estricto** | GBNF que solo permite la forma canónica del token (sin espacios ni variantes) |
| **Grammar permisivo** | GBNF que permite espacios y variantes de caso |
| **Think mode** | Modo de razonamiento interno de modelos como Qwen3.5, Nemotron, Ministral |
| **Model thrashing** | Competencia entre experimentos por la misma instancia de Ollama |
