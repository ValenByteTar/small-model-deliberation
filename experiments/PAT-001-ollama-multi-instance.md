---
id: PAT-001
title: "Patron: instancias Ollama multi-puerto para experimentacion paralela"
date: 2026-08-16
status: active
category: pattern
components: [ollama_provider, runners, ollama_instances]
tags: [ollama, multi-instance, parallel-execution, model-thrashing, gpu, cpu, isolation]
related: [POST-001, EXP-012, EXP-013, EXP-014]
supersedes: null
superseded_by: null
---

# PAT-001 - Instancias Ollama multi-puerto

## Problema

Ollama por defecto corre una sola instancia en `localhost:11434` con
`OLLAMA_MAX_LOADED_MODELS=1`. Cuando multiples experimentos corren en
paralelo contra la misma instancia:

1. Cada experimento pide un modelo diferente.
2. Ollama descarga el modelo actual, carga el nuevo, procesa el
   request, y al siguiente request del otro experimento, vuelve a
   descargar y cargar.
3. Este **model thrashing** dispara la latencia de 3s a 30s+ por
   request.
4. Los resultados experimentales se contaminan: las latencias medidas
   no reflejan el modelo, sino el overhead de carga/descarga.

**Caso real (POST-001):** Durante la repeticion de Llama3.2 en CPU, un
proceso en background compitio con el Micro-Coliseum en GPU por la
misma instancia Ollama. El model thrashing entre `llama3.2:3b` y
`gemma3:4b` elevo las latencias del Micro-Coliseum de ~3s a >30s por
caso.

## Solucion

Levantar multiples instancias de `ollama.exe serve` en puertos
diferentes, cada una con su propio modelo pinneado. Cada instancia:

- Su propio proceso `ollama.exe serve` (PID independiente).
- Su propio `OLLAMA_HOST` (puerto).
- `OLLAMA_MAX_LOADED_MODELS=1` (1 modelo pinneado, sin thrashing
  interno).
- `OLLAMA_NUM_GPU=0` (CPU) o `99` (GPU) segun asignacion.
- Modelo pre-cargado con `keep_alive=24h`.
- Comparte el store de modelos (`~/.ollama/models`) — lectura
  simultanea OK, no hay conflicto.

## Implementacion

### Gestor: `runners/ollama_instances.py`

Script CLI que gestiona el ciclo de vida de las instancias:

```bash
# Lanzar instancia CPU en puerto 11435 con llama3.2:3b
python runners/ollama_instances.py start --port 11435 --cpu --model llama3.2:3b

# Lanzar instancia GPU en puerto 11436 con gemma3:4b
python runners/ollama_instances.py start --port 11436 --gpu --model gemma3:4b-it-q4_K_M

# Ver estado de todas las instancias
python runners/ollama_instances.py status

# Detener instancia especifica
python runners/ollama_instances.py stop --port 11435

# Detener todas las instancias gestionadas
python runners/ollama_instances.py stop --all
```

El estado se persiste en `runners/.ollama-instances.json` para
sobrevivir entre invocaciones.

### Runners port-aware

Todos los runners aceptan `--port` (default: 11434):

| Runner | Flag |
|--------|------|
| `debate.py` | `--port 11434` |
| `run_microcoliseum_all.py` | `--port 11434` (propaga a debate.py) |
| `run_coliseo_v1_llama32_cpu_controlled.py` | `--port 11435` |
| `run_coliseo_v2_gpu.py` | `--port 11434` |
| `run_coliseo_v1_gpu.py` | `--port 11434` |
| `run_coliseo_benchmark_v2.py` | `--port 11435` |

### Provider subyacente

`OllamaModelProvider` ya acepta `base_url` desde su creacion (ADR-0007).
Los runners simplemente pasan `base_url=f"http://localhost:{args.port}"`
al constructor. No fue necesario modificar el provider.

## Topologias recomendadas

### Topologia 1: GPU + CPU (recomendada para laptop con 6GB VRAM)

```
Instancia A (:11434, GPU)  ← modelo X pinneado (experimento principal)
Instancia B (:11435, CPU)  ← modelo Y pinneado (experimento secundario)
```

Sin contencion de VRAM. La instancia CPU es mas lenta pero no
interfiere con la GPU.

### Topologia 2: GPU + GPU (solo si VRAM permite)

```
Instancia A (:11434, GPU)  ← modelo X (3B Q4, ~2.5GB VRAM)
Instancia B (:11436, GPU)  ← modelo Y (3B Q4, ~2.5GB VRAM)
```

Requiere >=5GB VRAM libres. Con 6GB (RTX 4050 Laptop), es marginal:
dos modelos Q4 3B caben pero sin margen para overhead.

### Topologia 3: Solo GPU (experimento unico)

```
Instancia A (:11434, GPU)  ← modelo X pinneado
```

Lo que se usaba antes. Sin paralelismo pero sin overhead de gestion.

## Restricciones

1. **VRAM**: cada instancia GPU pinnea un modelo en VRAM. Con 6GB,
   maximo 1-2 modelos Q4 3B simultaneos en GPU.
2. **RAM CPU**: cada instancia CPU pinnea un modelo en RAM. Un modelo
   3B Q4 ocupa ~2.5GB RAM. Con 16GB RAM, 3-4 instancias CPU son
   viables.
3. **Store compartido**: todas las instancias leen del mismo
   `~/.ollama/models`. Esto es seguro (read-only) pero significa que
   los modelos deben estar descargados previamente (`ollama pull`).
4. **Puertos**: usar puertos consecutivos (11434, 11435, 11436...) para
   simplicidad. Verificar con `netstat` antes de lanzar.

## Workflow recomendado

```bash
# 1. Lanzar instancias
python runners/ollama_instances.py start --port 11434 --gpu --model gemma3:4b-it-q4_K_M
python runners/ollama_instances.py start --port 11435 --cpu --model llama3.2:3b

# 2. Verificar
python runners/ollama_instances.py status

# 3. Correr experimentos en paralelo (terminales separadas)
python runners/run_microcoliseum_all.py --port 11434          # GPU
python runners/run_coliseo_v1_llama32_cpu_controlled.py --port 11435  # CPU

# 4. Al terminar
python runners/ollama_instances.py stop --all
```

## Lecciones

1. **El modelo thrashing es invisible hasta que no lo es**: las
   latencias se disparan de 3s a 30s+ sin warning. El primer sintoma
   es "el experimento va muy lento".
2. **`kill_shell` no mata procesos hijos**: cuando se lanza un
   experimento largo en background via `exec` con `timeout=0`, matar
   el shell wrapper no mata el proceso `python.exe` hijo. Usar
   `Stop-Process -Id <PID> -Force` directamente sobre el PID del
   proceso hijo.
3. **El aislamiento de instancias es la unica solucion robusta**:
   serializar experimentos (uno a la vez) evita el thrashing pero
   pierde paralelismo. La multi-instancia da ambos.
4. **Documentar el puerto como parte del experimento**: el puerto
   Ollama es una variable experimental. Dos runs del mismo experimento
   en puertos diferentes pueden tener latencias diferentes si una
   instancia esta compartida y la otra no.
