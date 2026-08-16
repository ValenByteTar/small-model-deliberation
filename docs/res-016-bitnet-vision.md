---
id: RES-016
category: research
status: draft
created: 2026-08-15
updated: 2026-08-15
author: human
components: [tutor-llm, knowledge-model, student-model, agentic-rag, moe, bitnet, model-strategy, assessment, pedagogical-policy]
tags: [tutor-llm, moe, bitnet, vision-largo-plazo, student-model, knowledge-model, pedagogical-loop, local-first, model-strategy]
related: [RES-003, RES-004, RES-007, ADR-0004, ADR-0007, ADR-0011, ADR-0015, ADR-0018, ADR-0020, ADR-0021, ADR-0022]
supersedes: null
superseded_by: null
---

# RES-016 - Tutor LLM: vision de arquitectura con MoE y BitNet

## Topic

Evolucion del Agentic RAG hacia un Tutor LLM local: un sistema compuesto que
usa conocimiento verificable para diagnosticar, ensenar, evaluar y adaptar su
intervencion pedagogica, explorando Mixture-of-Experts (MoE) y BitNet b1.58
como vias para capacidad especializada dentro de un presupuesto local.

## Sources

- Vision original: "De Agentic RAG a Tutor LLM: una vision de arquitectura con
  MoE y BitNet" (Boveda IA, 2026-08-14)
- Ma et al., "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"
  (arXiv:2402.17764)
- Wang, Ma y Wei, "BitNet a4.8: 4-bit Activations for 1-bit LLMs"
  (arXiv:2411.04965)
- Fedus, Zoph y Shazeer, "Switch Transformers: Scaling to Trillion Parameter
  Models with Simple and Efficient Sparsity" (arXiv:2101.03961)
- Zhou et al., "Mixture-of-Experts with Expert Choice Routing"
  (arXiv:2202.09368)
- RES-003 (Knowledge Consumer / LLMSupport)
- RES-004 (LLMSupport: observador paralelo de hipotesis)
- RES-007 (LLM Model Strategy: 3B vs 8B)
- ADR-0004 (Execution State) — base para Student Model
- ADR-0007 (ModelProvider) — inyeccion de modelos heterogeneos
- ADR-0011 (Local-first) — restriccion de operacion local
- ADR-0015 (Knowledge System) — contrato entre Consumer y Knowledge
- ADR-0018 (Builder/Consumer split) — Knowledge Model como artifact
- ADR-0020 (Ownership de decisiones) — separacion razonamiento/control
- BitNet b1.58-2B-4T: entorno bitnet.cpp verificado en CPU (121.5 t/s prompt,
  21.7 t/s generation, 4 threads, ~1.1GB RAM, 0% GPU)

---

## 1. Tesis central

El Agentic RAG actual no es el producto final. Es una **capacidad** del tutor:
la capa que encuentra y fundamenta el conocimiento necesario. El objetivo no es
construir un chatbot educativo con recuperacion documental, sino un sistema
local que pueda usar conocimiento verificable para diagnosticar, ensenar,
evaluar y adaptar su intervencion a una persona.

La transicion es:

```text
documentos -> conocimiento recuperable -> respuestas
```

a:

```text
documentos -> conocimiento recuperable y modelado
-> diagnostico -> intervencion -> evaluacion -> actualizacion del alumno
```

Esto requiere componentes que el RAG actual no tiene: Student Model, Assessment,
Politica pedagogica, Knowledge Model con granularidad pedagogica.

---

## 2. Por que RAG no es suficiente para ensenar

Un RAG competente localiza y sintetiza informacion. Eso no implica que sepa
ensenar. Ante una respuesta erronea de un alumno, responder con una explicacion
correcta puede ser insuficiente: hace falta distinguir entre:

- falta de prerequisitos
- confusion entre conceptos cercanos
- error de procedimiento
- lectura superficial
- formulacion ambigua de la consigna

Un LLM por si solo puede aproximar partes de esta conducta, pero no deberia ser
la unica fuente de estado ni el arbitro unico de su propia calidad.

---

## 3. Arquitectura del Tutor LLM

```text
Alumno -> Interaccion
              |
              v
     Diagnostico / Assessment
        |           |
        v           v
   Student Model  Knowledge Model
        |           |
        v           v
   Politica pedagogica
        |
        v
   Agentic RAG (recuperacion condicionada)
        |
        v
   LLM tutor (generacion condicionada por estado + evidencia)
        |
        v
   Respuesta + evento estructurado
        |
        v
   Observaciones: respuesta, tiempo, confianza, intentos
        |
        v
   Diagnostico (ciclo cerrado)
```

El LLM no trata cada turno como conversacion aislada. Recibe:
- representacion acotada del estado del alumno
- objetivo inmediato
- restricciones pedagogicas
- evidencia recuperada

Su salida incluye lenguaje natural **y eventos estructurados** que permiten
actualizar el estado sin inferirlo otra vez desde texto libre.

### 3.1 Knowledge Model (pedagogico)

El Knowledge Model actual (RES-002, ADR-0021) representa el dominio como
artifacts del Builder: entidades, relaciones, conceptos, document_roles. Para
el tutor, necesita evolucionar hacia:

- **Conceptos y habilidades**: unidades ensenables, con definiciones, ejemplos,
  contraejemplos y fuentes
- **Relaciones pedagogicas**: prerequisito, parte-de, confusion-frecuente-con,
  aplica-a, contradice, depende-de
- **Objetivos de dominio**: que evidencia observable cuenta como comprension,
  aplicacion o transferencia
- **Evidencia y procedencia**: cada afirmacion importante debe volver al
  documento, seccion o fragmento que la sustenta (ya cubierto por ADR-0022)
- **Granularidad pedagogica**: un chunk util para recuperar no siempre es una
  unidad util para ensenar; puede requerirse mapear chunks a conceptos

Los metadatos generados con Granite offline son un candidato natural para
bootstrap. Deben tratarse como **hipotesis revisables**, no como verdad
curricular.

### 3.2 Student Model

El Student Model es la memoria explicita del proceso de aprendizaje de una
persona. No pretende leer la mente del alumno: mantiene **estimaciones con
incertidumbre**.

Para cada concepto o habilidad puede registrar:

- nivel de dominio estimado y confianza de la estimacion
- evidencia observada: respuestas, pasos, intentos, aciertos por azar, tiempo,
  autoevaluacion
- errores o concepciones alternativas recurrentes
- prerequisitos posiblemente debiles
- intervenciones ya ofrecidas y su resultado
- preferencias de interaccion solo cuando sean relevantes y consentidas

**Separacion critica**: los **hechos observados** ("respondio X dos veces") se
separan de las **inferencias del sistema** ("no comprende Y"). Esta separacion
hace revisable el modelo, reduce la ilusion de precision y permite corregirlo.

Esto alinea con ADR-0004 (Execution State): el Student Model es un Execution
State especializado con persistencia entre sesiones.

### 3.3 Assessment y bucle pedagogico

La unidad de operacion no es una respuesta, sino un ciclo cerrado:

1. Seleccionar un objetivo pequeno y una hipotesis sobre el estado del alumno
2. Recuperar evidencia y construir una intervencion: pregunta diagnostica,
   explicacion, ejemplo trabajado, pista o practica
3. Observar la respuesta y evaluar tanto el resultado como, cuando exista, el
   procedimiento
4. Distinguir explicaciones alternativas del error antes de actualizar el
   Student Model
5. Elegir el siguiente paso: avanzar, practicar, cambiar representacion, volver
   a un prerequisito o pedir aclaracion

El assessment no mide solo "respuesta final correcta". Debe poder usar items de
transferencia, explicacion en palabras propias, justificacion de pasos y
contraste con contraejemplos. La automatizacion de correccion necesita
**umbrales de confianza y rutas de abstencion**: cuando no haya evidencia
suficiente, el tutor debe preguntar mas o marcar incertidumbre en vez de inventar
un diagnostico.

### 3.4 Rol del Agentic RAG en el tutor

El Agentic RAG sigue siendo esencial, pero con una funcion mas precisa:

- recuperar conocimiento relevante al **objetivo y al error observado**, no solo
  a la ultima pregunta
- contrastar fuentes y exponer procedencia cuando una explicacion depende de
  material especifico
- elegir el nivel de detalle y los ejemplos adecuados
- detectar vacios, conflicto entre fuentes o evidencia insuficiente
- servir al Knowledge Model y al tutor, no reemplazarlos

En un tutor, recuperar el pasaje mas parecido semanticamente puede ser una mala
decision. Tambien importa si ese pasaje presupone conocimientos que el alumno
aun no domina, si contiene el contraejemplo necesario o si ofrece una formulacion
apta para el nivel deseado. La recuperacion deberia progresar desde **ranking por
relevancia** hacia **ranking condicionado por concepto, prerequisitos, dificultad
y proposito pedagogico**.

---

## 4. Arquitectura de modelos: capacidad compartida y especializada

Una arquitectura plausible combina componentes con exigencias distintas, en vez
de exigir que un unico modelo grande haga todo:

| Funcion | Perfil de modelo deseable | Observacion |
| --- | --- | --- |
| Extraccion y metadata offline | Modelo local estable, batch-friendly | Granite ya cumple este papel (RES-007) |
| Recuperacion, reranking, verificacion | Encoders/rerankers especializados | Menor coste y evaluacion mas directa que LLM generativo |
| Tutor conversacional | Modelo generativo con buena instruccion y contexto | Debe estar condicionado por estado y evidencia |
| Diagnostico / actualizacion estructurada | Modelo pequeno o cabeza especializada + reglas | Conviene auditarlo y desacoplarlo de la prosa |
| Planificacion pedagogica | Politica explicita asistida por modelo | No debe confundirse con mera generacion de texto |
| Observador paralelo (hipotesis) | Modelo pequeno en CPU (BitNet) | RES-004 §8.1; bitnet.cpp verificado: 21.7 t/s en CPU |

El MoE es una posibilidad dentro del componente generativo, no un sustituto de
esta descomposicion. Un router no reemplaza ni el modelo de conocimiento ni el
modelo del alumno.

---

## 5. Por que MoE encaja con la idea

En un MoE, un router selecciona un subconjunto de expertos para cada token,
secuencia o tarea. Esto permite aumentar la capacidad total sin activar todos
los parametros en cada inferencia. En principio, encaja con un tutor porque el
sistema encuentra situaciones recurrentemente heterogeneas:

- explicacion conceptual vs resolucion procedimental
- diagnostico de errores vs generacion de ejercicios
- diferentes dominios, niveles de abstraccion o idiomas
- recuperacion, sintesis con evidencia y conversacion socratica
- casos que requieren prudencia, verificacion o derivacion

La intuicion atractiva no es "un experto por materia" en sentido rigido. La
especializacion util puede emerger por patron linguistico, tipo de razonamiento
o distribucion de tareas y no ser interpretable como una taxonomia humana.
Asignar de antemano etiquetas semanticas al router seria una **hipotesis a
probar**, no una propiedad garantizada.

### 5.1 Advertencias de MoE clasico

- Menor computo teorico por token no garantiza menor latencia real
- Routing, balanceo de carga, memoria, tamano de lote y comunicacion entre
  dispositivos pueden eliminar la ganancia
- Expertos subutilizados o colapsados son problemas de entrenamiento y
  operacion, no detalles menores
- Con batch pequeno (tipico de conversacion individual), la aceleracion de MoE
  puede ser inferior a la esperada

---

## 6. BitNet y el coste de inferencia

BitNet b1.58 propone pesos ternarios {-1, 0, +1}, llamados "1.58-bit" porque
codificar tres estados requiere log2(3) bits. El interes para esta vision no es
una promesa de rendimiento inmediato: es la posibilidad de reducir de forma
sustancial memoria, movimiento de datos y coste energetico de modelos entrenados
para esa representacion, especialmente en despliegues locales.

### 6.1 Verificacion empirica (2026-08-15)

Se configuro entorno bitnet.cpp en CPU con modelo BitNet-b1.58-2B-4T:

| Metrica | Valor | Hardware |
| --- | --- | --- |
| Prompt processing | 121.5 t/s | CPU 4 threads |
| Generation | 21.7 t/s | CPU 4 threads |
| Memoria modelo | ~1.1 GB | GGUF i2_s |
| Uso GPU | 0% | No compite con pipeline principal |

Estos numeros confirman que un modelo 2B ternario en CPU es viable para el rol
de observador paralelo (RES-004 §8.1). Para el rol de tutor conversacional
principal, la calidad de generacion de un modelo 2B requiere evaluacion
pedagogica antes de conclusiones.

### 6.2 Distincion critica

La cuantizacion posterior de un modelo convencional y entrenar un modelo
nativamente ternario **no son la misma cosa**. Las afirmaciones publicadas sobre
BitNet dependen de recetas de entrenamiento, kernels y condiciones experimentales
particulares; no se deben extrapolar automaticamente a un modelo tutor, a una GPU
concreta ni a una pila local existente. Ademas, las activaciones, el KV cache, el
contexto largo y el router siguen teniendo costes propios.

### 6.3 Conexion estructural MoE + BitNet

```text
MoE:    aumenta capacidad total con activacion escasa.
BitNet: reduce el coste de representar y ejecutar cada experto.
Combinacion: explorar mas capacidad especializada dentro de un presupuesto local.
```

No se sigue que la combinacion sea automaticamente eficiente. Un MoE de expertos
muy comprimidos puede estar dominado por overhead de routing y accesos dispersos;
el hardware y los kernels determinan si la aritmetica reducida se traduce en
latencia y throughput reales.

---

## 7. Hipotesis arquitectica: MoE + BitNet para tutor local

```text
Contexto: objetivo, evidencia RAG, estado acotado
    |
    v
Backbone compartido (atencion, capas sensibles en precision mayor)
    |
    v
Router ligero
    |
    +---> Experto: explicacion
    +---> Experto: diagnostico
    +---> Experto: practica
    +---> Experto: sintesis con evidencia
    |
    v
Decoder / salida
    |
    v
Respuesta + evento estructurado
```

Los nombres de expertos describen una **intencion de evaluacion**, no una
garantia de que el modelo se especialice asi. Una variante prudente mantiene
backbone, atencion, router y capas sensibles en precision mayor, y prueba
expertos ternarios o de baja precision solo donde haya kernels y metricas que lo
justifiquen. Tambien puede ser mas razonable un MoE pequeno para una subfuncion
(por ejemplo, clasificacion de intervencion) que un LLM MoE integral.

---

## 8. Ventajas potenciales

- **Capacidad condicional**: aplicar mas parametros totales sin ejecutar toda la
  red por token
- **Localidad economica**: la reduccion de memoria de pesos puede hacer viables
  experimentos o despliegues que un modelo denso equivalente no permite
- **Especializacion medible**: se puede evaluar si ciertos expertos mejoran
  diagnostico, generacion de practica o fidelidad a evidencia
- **Evolucion modular**: los componentes no generativos y el pipeline RAG
  conservan valor aunque MoE/BitNet no resulten convenientes
- **Privacidad y control**: un flujo local permite limitar exposicion de
  materiales y datos del alumno, sujeto a una politica de retencion apropiada
  (alinea con ADR-0011 local-first)

---

## 9. Trade-offs y riesgos

| Riesgo | Probabilidad | Impacto | Mitigacion |
| --- | --- | --- | --- |
| Complejidad multiplicada (MoE: router, balanceo, afinidad, empaquetado) | Alta | Alto | Empezar con MoE acotado a una subfuncion, no integral |
| Especializacion espuria (router explota correlaciones superficiales) | Media | Alto | Evaluar routing con metricas de balance y especializacion |
| Latencia impredecible con batch pequeno | Alta | Medio | Medir en condiciones reales (conversacion individual) |
| Pila inmadura (kernels, formatos, observabilidad para MoE+ternario) | Alta | Alto | Validar bitnet.cpp antes de comprometer arquitectura |
| Degradacion pedagogica silenciosa | Media | Alto | Evaluacion pedagogica antes de optimizar arquitectura |
| Falsa trazabilidad (citar fuentes no prueba que el diagnostico sea correcto) | Media | Alto | Separar diagnostico de generacion libre (RES-004 principio) |
| Datos sensibles (Student Model requiere minimizacion, consentimiento) | Media | Alto | Politica de retencion explicita, control del usuario |
| Coste de entrenamiento (inferencia barata no compensa entrenamiento inalcanzable) | Alta | Alto | Usar modelos pre-entrenados cuando sea posible; fine-tuning ligero |

---

## 10. Hipotesis tecnicas a validar

1. El Knowledge Model construido a partir de extraccion, chunking y metadatos
   mejora recuperacion pedagogica frente al RAG basado solo en similitud
2. Mantener un Student Model explicito mejora aprendizaje y calibracion del
   tutor, no solo la percepcion de personalizacion
3. Separar diagnostico estructurado de generacion libre reduce errores de
   actualizacion del estado
4. Un MoE pequeno mejora una metrica pedagogica concreta por unidad de
   latencia/memoria frente a un modelo denso del mismo presupuesto
5. El routing se mantiene balanceado y util en sesiones individuales de baja
   concurrencia, no solo en entrenamiento o batching alto
6. Expertos en baja precision entrenados adecuadamente preservan calidad de
   explicacion, evaluacion y seguimiento de instrucciones
7. La combinacion BitNet + MoE ofrece una mejora de extremo a extremo (memoria,
   latencia, energia y calidad) en el hardware objetivo; FLOPs y tamano de
   checkpoint no bastan
8. El sistema puede expresar incertidumbre y pedir evidencia adicional sin
   deteriorar de forma inaceptable la experiencia de aprendizaje

---

## 11. Roadmap conceptual

No es una secuencia obligatoria; ordena incertidumbres para no atribuir a MoE o
BitNet problemas que pertenecen a la pedagogia o al RAG.

1. **Base de conocimiento confiable.** Consolidar procedencia, calidad de
   extraccion, representacion de conceptos y evaluacion de recuperacion sobre el
   pipeline actual. (En curso: ADR-0022, RES-010, ka_v5.0.0)
2. **Tutor con estado explicito.** Probar un bucle diagnostico-intervencion-
   evaluacion con un modelo denso existente y un dominio acotado
3. **Evaluacion pedagogica.** Definir tareas, rubricas, conjuntos de errores y
   medidas longitudinales antes de optimizar arquitectura de modelos
4. **Descomposicion de funciones.** Comparar modelo unico, modelos pequenos
   especializados y politicas hibridas para diagnostico, planificacion y
   correccion (alinea con RES-004, RES-007)
5. **Experimentos de eficiencia.** Medir modelos locales compactos con el mismo
   contexto, corpus, hardware y objetivo pedagogico (bitnet.cpp verificado)
6. **MoE acotado.** Evaluar routing y especializacion solo si hay una funcion
   con heterogeneidad clara y una metrica de exito concreta
7. **BitNet / baja precision nativa.** Considerar esta linea cuando exista una
   ruta reproducible de entrenamiento o adaptacion y kernels adecuados para el
   hardware objetivo
8. **Co-diseno.** Solo entonces evaluar una combinacion MoE + BitNet como
   arquitectura integral o como componente puntual

---

## 12. Preguntas abiertas

1. Cual es la unidad minima util del Knowledge Model: concepto, habilidad,
   microobjetivo, grafo de prerequisitos o una combinacion?
2. Que evidencia permite actualizar dominio sin confundir un acierto casual con
   comprension?
3. Cuando el tutor debe explicar, preguntar, mostrar un ejemplo o abstenerse?
4. Como se evalua mejora educativa de forma etica y longitudinal, mas alla de
   satisfaccion conversacional?
5. Que funciones requieren realmente especializacion condicional y cuales se
   resuelven mejor con herramientas, reglas o modelos pequenos?
6. El routing debe operar por token, por turno, por objetivo pedagogico o por
   sesion? Que informacion puede usar sin sesgarse?
7. Que precision es aceptable por subcomponente (pesos, activaciones, KV cache,
   router) para conservar calidad pedagogica?
8. Que hardware local y que patron de concurrencia se pretende optimizar? La
   respuesta condiciona si MoE y BitNet tienen sentido practico
9. Como puede el alumno inspeccionar, corregir o borrar su modelo de progreso?

---

## 13. Criterio de exito

La vision tiene valor si el sistema puede demostrar, en un dominio delimitado,
que:

- usa fuentes correctas
- detecta incertidumbre
- adapta intervenciones con evidencia
- mejora resultados de aprendizaje o de dominio medido

Que sea capaz de generar texto convincente, que tenga muchos parametros o que
use pocos bits **no es un criterio suficiente**.

---

## 14. Relacion con la arquitectura existente

| Componente existente | Rol en el Tutor LLM |
| --- | --- |
| Knowledge Builder (ADR-0021) | Bootstrap del Knowledge Model pedagogico (conceptos, relaciones, evidencia) |
| Knowledge Consumer (ADR-0018, RES-003) | Recuperacion condicionada por concepto, prerequisitos, dificultad |
| Warm Artifacts (ADR-0022) | Fuente de verdad para Knowledge Model y Student Model |
| Execution State (ADR-0004) | Base para Student Model (estado persistente entre sesiones) |
| ModelProvider (ADR-0007) | Inyeccion de modelos heterogeneos (tutor, observador, diagnostico) |
| Policy Engine (ADR-0013) | Politica pedagogica explicita (no generacion libre) |
| LLMSupport (RES-004) | Observador paralelo de hipotesis pedagogicas (bitnet.cpp en CPU) |
| Model Strategy (RES-007) | Granite 3B/8B como tutor conversacional; BitNet 2B como observador |
| Canonical Document Contract (ADR-0022) | Procedencia y trazabilidad de evidencia pedagogica |
| Verification (ADR-0019) | Verificacion de respuestas del alumno contra evidencia |

El Tutor LLM **no descarta** la arquitectura existente. La amplifica: cada
componente tiene un rol pedagogico mas preciso.

---

## 15. Takeaways

1. **El Agentic RAG pasa de producto a capacidad.** El tutor lo usa para
   recuperar y fundamentar conocimiento, no como fin en si mismo
2. **El Knowledge Model necesita granularidad pedagogica.** Los chunks no son
   unidades de ensenanza; hace falta mapear chunks a conceptos y habilidades
3. **El Student Model es Execution State persistente.** Mantiene estimaciones
   con incertidumbre, separa hechos observados de inferencias
4. **El bucle pedagogico es la unidad de operacion.** No es una respuesta, es un
   ciclo diagnostico-intervencion-evaluacion
5. **MoE es una posibilidad, no una premisa.** La especializacion util puede
   emerger por patron, no por taxonomia humana; requiere validacion
6. **BitNet es estructuralmente atractivo pero no automaticamente eficiente.** La
   combinacion MoE + BitNet requiere co-diseno de hardware, kernels y metricas
7. **La evaluacion pedagogica va antes que la optimizacion arquitectonica.** No
   tiene sentido optimizar MoE/BitNet sin tareas, rubricas y medidas
   longitudinales
8. **La arquitectura existente se preserva.** Builder, Consumer, Warm Artifacts,
   Execution State, ModelProvider, Policy Engine: todos tienen un rol en el
   tutor
9. **No se implementa ahora.** Este research prepara la promocion futura a ADR,
   sujeto a validacion de las hipotesis tecnicas y pedagogicas

---

## 16. Criterio de promocion a ADR

Este research puede promoverse a ADR cuando se acuerde al menos:

- Tutor LLM como evolucion natural del Agentic RAG (no como producto separado)
- Knowledge Model pedagogico como extension del Knowledge Model actual (conceptos,
  habilidades, relaciones pedagogicas)
- Student Model como Execution State persistente con incertidumbre explicita
- Bucle pedagogico (diagnostico-intervencion-evaluacion) como unidad de operacion
- Politica pedagogica explicita asistida por modelo (no generacion libre)
- Recuperacion condicionada por concepto, prerequisitos, dificultad y proposito
- MoE como posibilidad exploratoria, no como decision arquitectonica
- BitNet como via de eficiencia local sujeta a validacion empirica
- Evaluacion pedagogica como prerequisito de cualquier optimizacion arquitectonica
- ADR separado para MoE si se valida la hipotesis #4 (MoE pequeno mejora metrica
  pedagogica por unidad de latencia/memoria)
- ADR separado para BitNet si se valida la hipotesis #7 (mejora de extremo a
  extremo en hardware objetivo)

Hasta entonces permanece como research de arquitectura de largo plazo.
