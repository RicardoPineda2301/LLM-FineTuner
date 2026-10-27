# LLM-FineTuner

**Fine-tuner local para modelos de lenguaje** - Entrena adaptadores LoRA sobre modelos base (Qwen2.5, Mistral, etc.) usando QLoRA con Unsloth, optimizado para bajo consumo de VRAM.

> **Estado: experimental** | Probado en Windows 11 con CUDA 12.x

---

## Qué hace

| Etapa | Descripción | Salida |
|---|---|---|
| **Ingesta** | Lee ejemplos de código y documentación en formato `.jsonl` crudo | `data/processed/*.json` |
| **Entrenamiento** | QLoRA: congelar modelo base, entrenar solo adaptadores LoRA (4-8 bits) | `data/output/adapters/*` |
| **Fusión** | Combina modelo base + adaptador LoRA en un solo modelo | `data/output/merged/*` |
| **Exportación** | Convierte a formato `.gguf` portable | `data/output/gguf/*` |
| **Empaquetado** | Genera `Modelfile` para Ollama | `data/output/ollama/*` |

---

## Stack

| Componente | Tecnología | Versión |
|---|---|---|
| Lenguaje | Python 3.11+ | Requerido |
| GPU | NVIDIA CUDA | 12.x+ |
| QLoRA | Unsloth | 2024.11+ |
| Modelos base | Qwen2.5, Mistral, Llama | HuggingFace |
| Conversión | llm-gguf | latest |
| Runtime | Ollama | reciente |

---

## Prerrequisitos

### Sistema
- **GPU NVIDIA** con **CUDA 12.x** y **cuDNN**
- **VRAM**: 8GB+ (12GB+ recomendado para 14B)
- **RAM**: 16GB+ (para procesamiento de datasets)

### Software
| Herramienta | Versión | Para qué |
|---|---|---|
| Python | 3.11-3.13 | Runtime principal |
| pip | 24+ | Gestor de paquetes |
| Git | 2.x | Clonar repositorios de modelos |
| Ollama | reciente | Servir modelos entrenados |

---

## Puesta en marcha

### 1. Entorno Python

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/macOS
# O en Windows:
venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Configuración

Copiar y editar el archivo de configuración:
```bash
cp config/config.example.yaml config/config.yaml
```

Editando las variables principales en `config/config.yaml`:
```yaml
model:
  base: "Qwen/Qwen2.5-7B-Instruct"  # o "mistralai/Mistral-7B-Instruct"
  output_dir: "data/output"
  quantization: "nf4"  # nf4, fp4, int8, int4

training:
  lora_rank: 16
  lora_alpha: 32
  max_seq_length: 4096
  epochs: 3
  batch_size: 2
  learning_rate: 2e-4

data:
  input_dir: "data/raw"
  processed_dir: "data/processed"
```

### 3. Ingestión de datos

Colocar archivos `.jsonl` en `data/raw/`. Formato esperado:
```json
{"text": "[INST] Explica este código... [/INST] \n```python\ndef hello():\n    return 'world'\n```"}
{"text": "[INST] Documenta esta función... [/INST] \n..."}
```

Procesar los datos:
```bash
python scripts/ingest.py
```

### 4. Entrenamiento QLoRA

```bash
python scripts/train.py
```

Este script:
1. Descarga automáticamente el modelo base de HuggingFace
2. Aplica QLoRA (4-bit) para reducir uso de VRAM
3. Entrena solo los adaptadores LoRA
4. Guarda el adaptador en `data/output/adapters/`

### 5. Fusión y Exportación

Fusionar modelo base + adaptador:
```bash
python scripts/merge.py
```

Exportar a `.gguf`:
```bash
python scripts/export.py
```

### 6. Empaquetado para Ollama

Generar Modelfile y registrar en Ollama:
```bash
python scripts/package_ollama.py
```

Esto crea:
- `data/output/ollama/finetuned-qwen2.5-7b/Modelfile`
- Ejecuta `ollama create finetuned-qwen2.5-7b -f Modelfile`

---

## Configuración detallada

### Parámetros de entrenamiento (QLoRA)

| Parámetro | Descripción | Valor por defecto | Recomendado |
|---|---|---|---|
| `lora_rank` | Rango LoRA (tamaño adaptador) | 16 | 8-64 |
| `lora_alpha` | Alpha de escalado LoRA | 32 | 2x lora_rank |
| `quantization` | Cuantización del modelo | nf4 | nf4, fp4, int8 |
| `max_seq_length` | Longitud máxima secuencia | 4096 | 2048-4096 |
| `epochs` | Número de épocas | 3 | 1-10 |
| `batch_size` | Tamaño de batch | 2 | 1-4 |
| `learning_rate` | Tasa de aprendizaje | 2e-4 | 1e-5 - 5e-4 |

### Modelos base soportados

| Modelo | Tamaño | VRAM (QLoRA 4-bit) | VRAM (QLoRA 8-bit) |
|---|---|---|---|
| Qwen2.5-7B-Instruct | 7B | ~6-8GB | ~8-10GB |
| Qwen2.5-14B-Instruct | 14B | ~12-14GB | ~14-16GB |
| Mistral-7B-Instruct | 7B | ~6-8GB | ~8-10GB |
| Llama3-8B-Instruct | 8B | ~7-9GB | ~9-11GB |

---

## Estructura del repositorio

```
LLM-FineTuner/
├── README.md                    # Este archivo
├── requirements.txt             # Dependencias Python
├── config/
│   ├── config.example.yaml      # Plantilla de configuración
│   └── config.yaml              # Configuración activa (gitignored)
├── data/
│   ├── raw/                     # Datos crudos (.jsonl)
│   │   └── examples.jsonl        # Ejemplo de dataset
│   ├── processed/               # Datos procesados
│   │   └── dataset.json         # Dataset listo para entrenamiento
│   ├── output/
│   │   ├── adapters/            # Adaptadores LoRA entrenados
│   │   │   └── qwen2.5-7b-lora/
│   │   │       ├── adapter_config.json
│   │   │       └── adapter_model.safetensors
│   │   ├── merged/               # Modelos fusionados
│   │   │   └── qwen2.5-7b-merged/
│   │   ├── gguf/                # Modelos en formato .gguf
│   │   │   └── qwen2.5-7b.gguf
│   │   └── ollama/               # Paquetes para Ollama
│   │       └── finetuned-qwen2.5-7b/
│   │           └── Modelfile
│   └── logs/                    # Logs de entrenamiento
├── scripts/
│   ├── ingest.py                 # Procesamiento de datos
│   ├── train.py                  # Entrenamiento QLoRA
│   ├── merge.py                  # Fusión modelo + adaptador
│   ├── export.py                 # Exportación a .gguf
│   └── package_ollama.py         # Empaquetado para Ollama
├── src/
│   └── utils/
│       ├── data_loader.py       # Cargador de datasets
│       ├── model_utils.py       # Utilidades de modelos
│       ├── ollama_utils.py      # Utilidades para Ollama
│       └── config_loader.py     # Cargador de configuración
└── examples/
    └── sample.jsonl              # Dataset de ejemplo
```

---

## Ejemplo completo

### 1. Preparar datos de ejemplo

```bash
# Usar el dataset de ejemplo
cp examples/sample.jsonl data/raw/

# O crear uno propio
python scripts/create_dataset.py --input code_samples/ --output data/raw/my_dataset.jsonl
```

### 2. Entrenar con QLoRA

```bash
python scripts/train.py \
  --config config/config.yaml \
  --model Qwen/Qwen2.5-7B-Instruct \
  --output_dir data/output/adapters/qwen2.5-7b-lora
```

### 3. Fusionar y exportar

```bash
python scripts/merge.py \
  --base Qwen/Qwen2.5-7B-Instruct \
  --adapter data/output/adapters/qwen2.5-7b-lora \
  --output data/output/merged/qwen2.5-7b-merged

python scripts/export.py \
  --model data/output/merged/qwen2.5-7b-merged \
  --output data/output/gguf/qwen2.5-7b.gguf \
  --quant int4
```

### 4. Registrar en Ollama

```bash
python scripts/package_ollama.py \
  --model_path data/output/gguf/qwen2.5-7b.gguf \
  --name finetuned-qwen2.5-7b \
  --template "qwen2.5"
```

### 5. Usar el modelo

```bash
ollama run finetuned-qwen2.5-7b
```

---

## Solución de problemas

### Error: "CUDA out of memory"

**Causa**: VRAM insuficiente para el modelo y batch size.

**Soluciones**:
1. Reducir `batch_size` a 1
2. Usar cuantización más agresiva (`int4` en lugar de `nf4`)
3. Reducir `max_seq_length`
4. Usar un modelo base más pequeño

### Error: "Module not found"

**Causa**: Entorno virtual no activado o dependencias no instaladas.

**Solución**:
```bash
source venv/bin/activate  # o venv\Scripts\activate en Windows
pip install -r requirements.txt
```

### Error: "Model not found"

**Causa**: Modelo base no descargado.

**Solución**:
```bash
# Unsloth descarga automáticamente, pero si falla:
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct
```

### Lento procesamiento de datos

**Causa**: Dataset grande siendo procesado en CPU.

**Soluciones**:
1. Procesar en lotes más pequeños
2. Usar `num_workers > 0` en el dataloader
3. Verificar que el dataset esté en formato `.jsonl` correcto

---

## Tips para datasets

### Formato de datos

Los archivos `.jsonl` deben contener objetos JSON, uno por línea:

```json
{"text": "[INST] {instrucción} [/INST] {respuesta}"}
```

### Ejemplo de dataset para fine-tuning

```json
{"text": "[INST] Escribe una función en Python que sume dos números. [/INST] def suma(a, b):\n    return a + b"}
{"text": "[INST] Explica qué hace este código:\n```python\nx = 5\ny = 10\nz = x + y\n``` [/INST] El código define dos variables x e y con valores 5 y 10 respectivamente, luego suma sus valores y almacena el resultado en la variable z."}
{"text": "[INST] Corrige este código Python:\n```python\ndef factorial(n):\n    if n == 0:\n        return 0\n    return n * factorial(n-1)\n``` [/INST] El error está en la condición base. Debería ser `if n == 0: return 1` para que el factorial de 0 sea 1 en lugar de 0."}
```

### Fuentes de datos recomendadas

1. **Código del cliente**: Extraer funciones/comentarios del repositorio
2. **Documentación**: Manuales, API docs, READMEs
3. **Ejemplos de uso**: Scripts de ejemplo, notebooks
4. **Q&A interno**: Preguntas frecuentes de la empresa
5. **Stack Overflow / GitHub**: Issues y soluciones relevantes

---

## Personalización avanzada

### Usar modelo local existente

Si ya tienes un modelo descargado:

```yaml
# En config.yaml
model:
  base: "/path/a/tu/modelo"
  local: true
```

### Ajustar hiperparámetros

```yaml
# En config.yaml
training:
  lora_rank: 32
  lora_alpha: 64
  lora_dropout: 0.1
  epochs: 5
  batch_size: 4
  learning_rate: 3e-4
  warmup_steps: 100
  weight_decay: 0.01
```

### Entrenamiento multi-GPU

```yaml
# En config.yaml
training:
  multi_gpu: true
  gradient_accumulation_steps: 2
```

---

## Rendimiento esperado

| Hardware | Modelo | VRAM usada | Tiempo/época (10k samples) |
|---|---|---|---|
| RTX 3060 (12GB) | Qwen2.5-7B | ~8GB | ~30-45 min |
| RTX 4070 (12GB) | Qwen2.5-7B | ~8GB | ~20-30 min |
| RTX 4090 (24GB) | Qwen2.5-14B | ~14GB | ~40-60 min |
| RTX 4090 (24GB) | Qwen2.5-7B | ~8GB | ~15-20 min |

---

## Contribuir

1. Forkear el repositorio
2. Crear una rama (`git checkout -b feature/nueva-funcionalidad`)
3. Commitear cambios (`git commit -m 'Añadir nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abrir Pull Request

---

## Licencia

MIT License - Ver archivo [LICENSE](LICENSE) para detalles.

---

## Agradecimientos

- [Unsloth](https://github.com/unslothai/unsloth) por QLoRA eficiente
- [HuggingFace](https://huggingface.co) por los modelos base
- [Ollama](https://ollama.com) por el runtime local
- [llm-gguf](https://github.com/ggerganov/llama.cpp) por la conversión a .gguf
