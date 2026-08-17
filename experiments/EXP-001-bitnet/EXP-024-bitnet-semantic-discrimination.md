# EXP-024: Semantic Discrimination x Decoding

## Fecha
2025-01-XX

## Contexto

EXP-023 descubrio que el grammar GBNF es una variable experimental de
primer orden para BitNet, invirtiendo completamente el comportamiento
(TRUE x55 bajo grammar estricto, FALSE x53 bajo grammar permisivo).
Esto confundio la interpretacion de EXP-017 a EXP-022: no podiamos
separar la capacidad semantica del sesgo de decodificacion.

EXP-024 busca responder la pregunta anterior a todas las demas:
**¿Existe señal semantica en la distribucion de probabilidades de
BitNet, antes de imponer cualquier interfaz de decision?**

## Hipotesis

**H0 (nula):** La distribucion P(TRUE)/P(FALSE) no cambia
sistematicamente segun la relacion semantica entre claim y evidence.
BitNet no tiene señal semantica explotable.

**H1 (alternativa):** La distribucion P(TRUE)/P(FALSE) cambia
sistematicamente: P(TRUE|SUPPORTS) > P(TRUE|CONTRADICTS). BitNet
tiene señal semantica aunque el clasificador final sea malo.

## Diseño

### Pares minimos

20 pares de claims que comparten el mismo evidence pero difieren en
un elemento semantico, en 4 grupos:

| Grupo | Descripcion | Pares |
|-------|-------------|-------|
| G1 | Entailment: soportado vs no-soportado | 5 |
| G2 | Contradiction: negacion explicita vs coincidencia | 5 |
| G3 | Absence: componente adicional ausente | 5 |
| G4 | Subject: mismo topico, diferente entidad | 5 |

Cada par tiene 3 variantes (A, B, C) → 60 casos totales.

### Condiciones factoriales

| Variable | Condiciones |
|----------|-------------|
| Grammar | strict / permissive / none |
| Seed | 42 (fijo) |
| Temperature | 0.0 |
| n_probs | 15 (top-15 logprobs) |

Total: 60 casos × 3 grammars = 180 LLM calls.

### Metrica principal

**P(TRUE | expected_relation)**: promedio de P(TRUE) sobre todos los
casos con una relacion semantica esperada.

Si P(TRUE|SUPPORTS) > P(TRUE|CONTRADICTS) + 0.05: hay señal.
Si P(TRUE|SUPPORTS) ≈ P(TRUE|CONTRADICTS): no hay señal.

### Metrica secundaria

**Sensibilidad a modificacion semantica minima**: para cada par,
comparar P(TRUE|A) vs P(TRUE|B) vs P(TRUE|C). Si A es SUPPORTS y B
no lo es, P(TRUE|A) deberia ser > P(TRUE|B).

## Resultados

### Distribucion por relacion semantica

| Expected | N | P(TRUE) | P(FALSE) | P(CANNOT) | P(T)-P(F) |
|----------|---|---------|----------|-----------|-----------|
| SUPPORTS | 22 | 0.4000 | 0.6000 | 0.0000 | -0.1999 |
| CONTRADICTS | 12 | 0.3983 | 0.6017 | 0.0000 | -0.2034 |
| PARTIAL | 11 | 0.4668 | 0.5332 | 0.0000 | -0.0663 |
| UNRELATED | 15 | 0.3941 | 0.6059 | 0.0000 | -0.2119 |

**Delta P(TRUE|SUPPORTS) - P(TRUE|CONTRADICTS) = +0.0017**

### Por grammar

| Grammar | P(T\|SUPPORTS) | P(T\|non-SUPPORTS) | Delta | Veredicto |
|---------|---------------|-------------------|-------|-----------|
| strict | 0.4000 | 0.4165 | -0.0164 | NO hay señal |
| permissive | 0.3994 | 0.4153 | -0.0158 | NO hay señal |
| none | 0.3994 | 0.4153 | -0.0158 | NO hay señal |

El delta es **negativo**: BitNet asigna ligeramente MAS probabilidad
a TRUE para casos no-soportados que para soportados. Esto es lo
opuesto a discriminacion semantica.

### Sensibilidad a modificacion semantica minima

| Grammar | Signal rate (P(T\|A) > P(T\|B/C) + 0.05) |
|---------|------------------------------------------|
| strict | 2/19 (10.5%) |
| permissive | 1/19 (5.3%) |
| none | 1/19 (5.3%) |

Solo 1-2 de 19 pares minimos muestran diferencia significativa. Esto
es consistente con azar.

### Pares minimos destacados

**G4 (subject pairs) — cero señal:**

| Pair | A (SUPPORTS) | P(T\|A) | B (UNRELATED) | P(T\|B) | A>B? |
|------|-------------|---------|---------------|---------|------|
| G4-P1 | Product X | 0.4635 | Product Y | 0.4289 | ~ |
| G4-P2 | NIST | 0.4697 | ISO 27001 | 0.4929 | ~ |
| G4-P3 | TLS 1.3 | 0.4104 | TLS 1.2 | 0.4383 | ~ |
| G4-P4 | AWS S3 | 0.4035 | Azure Blob | 0.4483 | ~ |
| G4-P5 | federal | 0.3377 | state | 0.3433 | ~ |

BitNet no puede distinguir "Product X" de "Product Y" ni "NIST" de
"ISO 27001" en la distribucion de probabilidades. Las diferencias son
< 0.05 en todos los casos.

**G2 (contradiction pairs) — señal invertida:**

| Pair | A (CONTRADICTS) | P(T\|A) | B (SUPPORTS) | P(T\|B) | A>B? |
|------|----------------|---------|-------------|---------|------|
| G2-P1 | contradice | 0.4094 | coincide | 0.3344 | Y |
| G2-P2 | contradice | 0.5080 | coincide | 0.3927 | Y |
| G2-P3 | contradice | 0.3265 | coincide | 0.2641 | Y |

En G2, BitNet asigna MAS probabilidad a TRUE para contradicciones que
para casos soportados. Esto es lo opuesto a discriminacion semantica.
El mecanismo probable: las contradicciones tienen mas overlap de
keywords (mismas palabras, diferente negacion), y BitNet responde al
overlap, no a la semantica.

### P(CANNOT_TELL) = 0.0000

En todos los casos, bajo todos los grammars, P(CANNOT_TELL) = 0.0000.
BitNet no tiene representacion de incertidumbre. El modelo nunca
considera que no puede determinar la respuesta.

## Analisis

### H0 confirmada: no hay señal semantica

La distribucion P(TRUE)/P(FALSE) no cambia sistematicamente segun la
relacion semantica. El delta P(TRUE|SUPPORTS) - P(TRUE|CONTRADICTS) =
+0.0017 es estadisticamente indistinguible de cero.

BitNet no esta "parcialmente sabiendo pero no podemos extraerlo".
**BitNet simplemente no tiene la señal.**

### El comportamiento es independiente del grammar

Los tres grammars (strict, permissive, none) producen distribuciones
de probabilidad casi identicas. Esto confirma EXP-023: el grammar
afecta el token greedy (TRUE vs FALSE), pero no afecta la
distribucion subyacente. La distribucion es siempre ~40% TRUE, ~60%
FALSE, independientemente de la semantica.

### El unico patron: PARTIAL ligeramente mas TRUE

P(TRUE|PARTIAL) = 0.4668 vs ~0.40 para los demas. Esto no es
discriminacion semantica: los casos PARTIAL tienen mas overlap de
keywords con el evidence (comparten algunos componentes pero no
todos), y BitNet responde al overlap. Es la misma explicacion que
EXP-020: comportamiento consistente con matching holistico.

### G2 (contradiction): señal invertida explica EXP-018

En G2, P(TRUE|CONTRADICTS) > P(TRUE|SUPPORTS). Las contradicciones
tienen mas overlap de keywords (mismas palabras, diferente negacion).
Bajo grammar estricto, esto se manifestaba como TRUE para
contradicciones — pero EXP-018 interpreto esos TRUE como SUPPORTS
porque el benchmark no tenia contradicciones en los casos SUPPORTS.

### P(CANNOT_TELL) = 0: sin representacion de incertidumbre

BitNet nunca asigna probabilidad a CANNOT_TELL. El modelo trata
siempre como si pudiera determinar la respuesta. Esto es consistente
con EXP-021: el modelo no reconoce ausencia de informacion como una
condicion valida.

## Conclusiones

1. **No hay señal semantica en la distribucion de probabilidades de
   BitNet.** P(TRUE|SUPPORTS) ≈ P(TRUE|CONTRADICTS) ≈ P(TRUE|UNRELATED).
   El delta es 0.0017, indistinguible de ruido.

2. **Esto es independiente del grammar.** Los tres grammars producen
   la misma distribucion. El grammar cambia el token greedy pero no
   la distribucion subyacente.

3. **BitNet no esta "parcialmente sabiendo".** No hay señal oculta
   en los logprobs que un mejor extractor pudiera recuperar. La
   distribucion es plana respecto a la semantica.

4. **El comportamiento es consistente con matching holistico por
   overlap de keywords**, no con discriminacion semantica. Los casos
   con mas overlap (PARTIAL, contradicciones con mismas palabras)
   tienen ligeramente mas P(TRUE), pero esto no es semantica.

5. **P(CANNOT_TELL) = 0.0000**: BitNet no tiene representacion de
   incertidumbre.

6. **EXP-018's "12/12 SUPPORTS" fue puramente un artefacto del
   grammar estricto.** No habia discriminacion semantica subyacente.

## Implicancias finales para BitNet

**BitNet no tiene caso de uso en el pipeline semantico.** No como
juez generalista, no como worker especializado, no como extractor
de señales elementales. La distribucion de probabilidades no
contiene señal semantica explotable.

La unica excepcion posible es relevance detection clara (wrong_subject
con entidades completamente diferentes), donde EXP-019 mostro 100%
de deteccion. Pero EXP-024 muestra que esto no se refleja en la
distribucion P(TRUE)/P(FALSE) — probablemente funciona por un
mecanismo diferente (ausencia total de overlap de keywords) que no
generaliza a distinciones semanticas finas.

**PM-003 cerrado definitivamente.** BitNet no es viable como
componente semantico en ningun rol.

## Implicancias para la arquitectura

```
                Semantic Assessment
                       |
             +---------+---------+
             |                   |
       deterministic         LLM signals
          contracts               |
             |             +-----+-----+
             |             |           |
             |          Qwen3.5    (BitNet: NO)
             |
             +-------------+---------+
                          |
                    deterministic policy
```

BitNet sale del pipeline semantico. La arquitectura no necesita un
modelo barato para señales elementales si esas señales no contienen
informacion semantica.
