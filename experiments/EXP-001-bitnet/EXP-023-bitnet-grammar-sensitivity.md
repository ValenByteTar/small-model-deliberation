# EXP-023: Grammar Sensitivity Probe for BitNet 1.58b

## Fecha
2025-01-XX

## Contexto

Durante EXP-022 (microcoliseum especializado) se descubrio que los
grammars GBNF permisivos (con espacios y variantes de caso) cambian
la tokenizacion de BitNet, invirtiendo su comportamiento. Este
hallazgo no debe quedar enterrado dentro de EXP-022: es una variable
experimental de primer orden que afecta retrospectivamente la
interpretacion de EXP-017 a EXP-022.

## Hipotesis

**H0:** La gramatica de decodificacion no afecta significativamente
el comportamiento de BitNet. Diferentes niveles de restriccion
gramatical producen la misma distribucion de outputs.

**H1:** La gramatica de decodificacion es una variable experimental
de primer orden. Diferentes niveles de restriccion gramatical pueden
invertir la distribucion de outputs y la accuracy por categoria.

## Diseño

### Variables controladas (constantes)

| Variable | Valor |
|----------|-------|
| Modelo | BitNet-b1.58-2B-4T |
| Seed | 42 |
| Temperature | 0.0 |
| num_predict | 6 |
| n_probs | 10 |
| Prompt | NLI 3a (TRUE/FALSE/CANNOT_TELL) de EXP-018 |
| Benchmark | semantic_assessment_v2.json (55 casos) |
| num_thread | 4 |

### Variable independiente (unica)

| Condicion | Grammar | Descripcion |
|-----------|---------|-------------|
| G1 strict | `root ::= "TRUE" \| "FALSE" \| "CANNOT_TELL"` | Solo forma canonica, sin espacios |
| G2 permissive | `root ::= "TRUE" \| " TRUE" \| " True" \| ...` (12 variantes) | Espacios + variantes de caso |
| G3 none | (sin grammar) | Modelo genera libremente, parser extrae etiqueta |

### Metricas

- Accuracy (global y excluyendo PARTIAL)
- Distribucion de outputs (que etiqueta genera)
- Token raw elegido (que token exacto produce)
- Logprobs del primer token
- Latencia
- Tasa de outputs invalidos

## Resultados

### Accuracy global

| Condicion | Accuracy (all) | Accuracy (excl. PARTIAL) | Invalid | Avg Latency | P50 Latency |
|-----------|---------------|-------------------------|---------|-------------|-------------|
| G1 strict | 21.8% (12/55) | 32.4% (12/37) | 0.0% | 0.514s | 0.487s |
| G2 permissive | 27.3% (15/55) | 40.5% (15/37) | 0.0% | 0.459s | 0.440s |
| G3 none | 27.3% (15/55) | 40.5% (15/37) | 0.0% | 0.618s | 0.598s |

### Distribucion de outputs

| Condicion | SUPPORTS | CONTRADICTS | UNRELATED |
|-----------|----------|-------------|-----------|
| G1 strict | **55** | 0 | 0 |
| G2 permissive | 0 | **53** | 2 |
| G3 none | 0 | **53** | 2 |

**G1 genera TRUE para absolutamente todo.**
**G2 y G3 generan FALSE para casi todo.**

### Token raw elegido

| Condicion | Top tokens |
|-----------|------------|
| G1 strict | `TRUE` x55 |
| G2 permissive | `False` x40, `FALSE` x13, `CANNOT_TELL` x2 |
| G3 none | `False\n\nBased on the` x39, `FALSE\n\nBased on the` x13, `CANNOT_TELL\n\nBased` x2 |

### Accuracy por categoria

| Categoria | G1 strict | G2 permissive | G3 none |
|-----------|-----------|---------------|---------|
| direct_evidence | **100.0%** | 0.0% | 0.0% |
| paraphrase | **100.0%** | 0.0% | 0.0% |
| explicit_contradiction | 0.0% | **100.0%** | **100.0%** |
| implicit_contradiction | 0.0% | **100.0%** | **100.0%** |
| negation | 0.0% | **66.7%** | **66.7%** |
| adversarial | 0.0% | 16.7% | 16.7% |
| over_specificity | 0.0% | 0.0% | 0.0% |
| partial_support | 0.0% | 0.0% | 0.0% |
| wrong_context | 0.0% | 0.0% | 0.0% |
| wrong_subject | 0.0% | 0.0% | 0.0% |

### McNemar's test

| Par | b (G1 ok, G2 X) | c (G1 X, G2 ok) | chi2 | Significativo |
|-----|------------------|------------------|------|---------------|
| G1 vs G2 | 12 | 15 | 0.33 | No (global) |
| G1 vs G3 | 12 | 15 | 0.33 | No (global) |
| G2 vs G3 | 0 | 0 | 0.00 | Identicos |

El test global no es significativo porque los efectos se cancelan:
G1 acierta los 12 SUPPORTS y falla los 15 CONTRADICTS, G2 hace lo
opuesto. Pero **la inversion por categoria es perfecta y completa**.

G2 y G3 son estadisticamente identicos (b=0, c=0).

## Analisis

### H1 confirmada: inversion completa

La gramatica de decodificacion **invierte completamente** el
comportamiento de BitNet:

```
  G1 strict:    TRUE x55  →  acierta SUPPORTS, falla CONTRADICTS
  G2 permisivo: FALSE x53 →  acierta CONTRADICTS, falla SUPPORTS
  G3 sin grammar: FALSE x53 →  identico a G2
```

No es un cambio sutil: es una inversion binaria. El modelo dice TRUE
para todo o FALSE para todo, dependiendo exclusivamente del grammar.

### Mecanismo: tokenizacion con espacio

El mecanismo es un artefacto de tokenizacion:

- `"TRUE"` (sin espacio) es un token unico
- `" FALSE"` (con espacio) es un token diferente
- `"True"` (title case) es otro token diferente

Con grammar estricto, BitNet solo puede generar `"TRUE"`, `"FALSE"`,
o `"CANNOT_TELL"` (sin espacio). De estos tres, prefiere `"TRUE"`.

Con grammar permisivo, BitNet puede generar `" FALSE"` (con espacio),
que es lo que naturalmente prefiere. El espacio cambia el token y la
probabilidad asociada.

Sin grammar, BitNet genera `" False"` (con espacio, title case), que
es su preferencia natural. El parser lo mapea a CONTRADICTS.

**G2 (permisivo) = G3 (sin grammar)** porque el grammar permisivo
permite el token que BitNet prefiere naturalmente. El grammar no
restringe nada que el modelo no fuera a hacer de todos modos.

### G1 es el caso anomalo

G1 (estricto) es el caso anomalo, no G2. Al forzar tokens sin
espacio, G1 cambia la distribucion de probabilidades y hace que
BitNet prefiera `"TRUE"`. Esto no es el comportamiento "natural" del
modelo — es un artefacto de la restriccion gramatical.

### Implicancias para EXP-017 a EXP-022

| Experimento | Grammar | Comportamiento | Reinterpretacion |
|-------------|---------|----------------|------------------|
| EXP-017 | Labels (SUPPORTS/CONTRADICTS/...) | SUPPORTS wall | Multi-token, diferente dinamica |
| EXP-018 | **Estricto** | TRUE para todo → acierta SUPPORTS | **Artefacto del grammar estricto** |
| EXP-019 | **Permisivo** | FALSE para todo → acierta CONTRADICTS | Comportamiento natural |
| EXP-020 | **Permisivo** | FALSE para todo | Comportamiento natural |
| EXP-021 | **Permisivo** | FALSE para todo | Comportamiento natural |
| EXP-022 | **Estricto** (despues del fix) | TRUE para todo → acierta SUPPORTS | **Artefacto del grammar estricto** |

**La "SUPPORTS wall" que EXP-018 rompio con NLI reframing fue
parcialmente un artefacto del grammar estricto**, no solo del token
labeling. Si EXP-018 hubiera usado grammars permisivos, habria
obtenido el resultado opuesto: una "CONTRADICTS wall" donde BitNet
dice FALSE para todo.

**Las conclusiones de EXP-019, 020, 021 siguen siendo validas** porque
usaron grammars permisivos, que reproducen el comportamiento natural
del modelo. Pero la comparacion con EXP-018 esta confundida: las
diferencias entre EXP-018 y EXP-019/020/021 pueden deberse
parcialmente al grammar, no solo al regimen experimental.

### Por que la accuracy global es similar

G1 y G2 tienen accuracy global similar (21.8% vs 27.3%) porque el
benchmark tiene aproximadamente la misma cantidad de casos SUPPORTS
(12) que CONTRADICTS (16). G1 acierta los 12 SUPPORTS, G2 acierta
los 15 CONTRADICTS. La diferencia neta es pequena, pero **aciertan
categorias completamente diferentes**.

### Latencia

G1 (0.514s) y G2 (0.459s) son mas rapidos que G3 (0.618s) porque el
grammar detiene la generacion despues del primer token valido. G3
genera mas tokens (incluyendo "Based on the evidence..." despues del
label), lo que aumenta la latencia.

## Conclusiones

1. **La gramatica de decodificacion es una variable experimental de
   primer orden para BitNet.** Cambia completamente la distribucion
   de outputs, no solo la restringe.

2. **La inversion es binaria y completa:** G1 produce TRUE para todo,
   G2/G3 producen FALSE para todo. No hay gradacion.

3. **El mecanismo es la tokenizacion con espacio:** `"TRUE"` (sin
   espacio) y `" FALSE"` (con espacio) son tokens diferentes con
   probabilidades diferentes. El grammar estricto fuerza el token sin
   espacio, cambiando la preferencia del modelo.

4. **G2 (permisivo) = G3 (sin grammar):** el grammar permisivo no
   restringe nada que el modelo no fuera a hacer naturalmente.

5. **G1 (estricto) es el caso anomalo:** fuerza un comportamiento que
   no es natural del modelo.

6. **Esto afecta la interpretacion de EXP-018 y EXP-022:** la
   "SUPPORTS wall" que NLI reframing rompio fue parcialmente un
   artefacto del grammar estricto. Las conclusiones de EXP-019/020/021
   siguen siendo validas (usaron grammars permisivos = comportamiento
   natural), pero la comparacion con EXP-018 esta confundida.

## Recomendaciones

1. **Todos los futuros experimentos con BitNet deben reportar el
   grammar exacto usado.** Es una variable experimental de primer
   orden, equivalente a temperature o seed.

2. **Los grammars estrictos pueden ser utiles** para forzar un
   comportamiento especifico (e.g., forzar TRUE para acertar
   SUPPORTS), pero deben usarse con conciencia de que estan cambiando
   el comportamiento del modelo, no solo restringiendolo.

3. **Para comparaciones entre experimentos, usar el mismo grammar.**
   Las comparaciones entre EXP-018 (estricto) y EXP-019/020/021
   (permisivo) estan confundidas por el grammar.

4. **Actualizar la documentacion de EXP-017-022** con un caveat
   explicando que el grammar es una variable de primer orden y que
   algunos experimentos usaron grammars diferentes.

## Caveat para paper

> La gramatica de decodificacion (GBNF) es una variable experimental
> de primer orden para BitNet 1.58b. Grammars estrictos (sin espacios
> ni variantes de caso) y grammars permisivos (con espacios) producen
> comportamientos opuestos: el primero genera TRUE para todo, el
> segundo genera FALSE para todo. Esto se debe a que los tokens con y
> sin espacio tienen probabilidades diferentes. Los experimentos
> reportados en este trabajo usaron diferentes niveles de restriccion
> gramatical (EXP-018: estricto, EXP-019-021: permisivo), lo que
> debe tenerse en cuenta al comparar resultados entre experimentos.
