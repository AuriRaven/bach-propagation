# Bach Propagation

> Pipeline de investigación y aplicación generativa para el análisis y síntesis del lenguaje armónico barroco. El sistema ingiere partituras MIDI/MusicXML, extrae secuencias armónicas mediante un análisis simbólico multicapa, entrena modelos generativos (Transformer para el corpus completo barroco; LSTM como baseline para el subcorpus de instrumento solo) y expone los resultados a través de una API REST consumida por una interfaz interactiva en Next.js.

---

## Arquitectura del sistema

```
┌─────────────────────────────────────────────────────────────────────┐
│                          FRONTEND  (Next.js / React)                │
│                                                                     │
│  bach-workbench.tsx ──── LibraryView  (exploración del corpus)      │
│         │           ──── CompositionView  (generación interactiva)  │
│         │           ──── AnalysisView  (análisis armónico)          │
│         │           ──── SettingsView                               │
│         │                                                           │
│  useMidiPlayer  │  useAiChat  │  useGeneration  │  api-client.ts    │
└─────────────────────────┬───────────────────────────────────────────┘
                          │ HTTP / REST  (puerto 3000 → 8000)
┌─────────────────────────▼───────────────────────────────────────────┐
│                         BACKEND  (FastAPI / Python 3.11+)           │
│                                                                     │
│  POST /api/generate   ← MusicTransformer (transformer-v1)           │
│  GET  /api/corpus     ← Consulta y streaming del corpus curado      │
│  POST /api/analysis   ← Pipeline de análisis armónico por partitura │
│  POST /api/ai/chat    ← Chat musicológico asistido por LLM          │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────────┐
          ▼               ▼                   ▼
   MusicTransformer  BaroqueHarmonyLSTM   Corpus (Supabase)
   (corpus barroco)  (instrumento solo)   + ETL pipeline
```

### Modelos

| Modelo | Corpus | Descripción |
|---|---|---|
| `MusicTransformer` | Corpus barroco completo (multi-compositor) | Transformer generativo condicionado por función armónica; genera secuencias de acordes en estilo barroco con gramática restringida. |
| `BaroqueHarmonyLSTM` | Subcorpus de instrumento solo (Bach) | LSTM baseline de 2 capas para predicción del siguiente acorde; sirve como línea de referencia cuantitativa. |

---

## Pipeline de análisis armónico

```
MIDI / MusicXML
      │
      ▼
  ScoreLoader          ← parseo a representación interna Score
      │
      ▼
  Quantizer            ← ajuste de onsets/duraciones a rejilla rítmica (1/16)
      │
      ▼
  Windower             ← segmentación por compás, tiempo o ventana deslizante
      │
      ▼
  KeyEstimator         ← detección de tonalidad (Krumhansl-Schmuckler)
      │
      ▼
  ChordClassifier      ← detección de acordes por plantillas (media geométrica)
      │
      ▼
  RomanNumeralAnalyzer ← (Acorde, Tonalidad) → cifrado romano + dominantes secundarias
      │
      ▼
  FunctionLabeler      ← etiquetado T / PD / D con sobreescrituras contextuales
      │
      ▼
  ProlongationAnalyzer ← árbol de reducción schenkeriana en 3 niveles
      │
      ▼
  SequenceExtractor    ← codificación a tokens (vocabulario de 90)
      │
      ▼
  MusicTransformer / BaroqueHarmonyLSTM
```

---

## Estructura del repositorio

```
bach-propagation/
├── backend/                          # Servicio Python (FastAPI)
│   ├── main.py                       # Punto de entrada; registra routers y CORS
│   ├── pyproject.toml                # Dependencias y configuración (uv)
│   ├── routers/
│   │   ├── generation.py             # POST /api/generate — generación con Transformer
│   │   ├── analysis.py               # POST /api/analysis — análisis armónico
│   │   ├── corpus.py                 # GET  /api/corpus  — consulta del corpus curado
│   │   └── ai_chat.py                # POST /api/ai/chat — chat musicológico (LLM)
│   ├── src/
│   │   ├── core/                     # Tipos musicales y cargador de partituras
│   │   ├── temporal/                 # Cuantización métrica y ventanas de análisis
│   │   ├── harmonic/                 # Estimación de tonalidad, clasificación de acordes,
│   │   │                             #   numerales romanos, función armónica
│   │   ├── schenkerian/              # Árbol de prolongación schenkeriana (3 niveles)
│   │   ├── data/                     # Extractor de secuencias, encoder (90 tokens), dataset
│   │   ├── models/                   # MusicTransformer, BaroqueHarmonyLSTM, generador
│   │   ├── grammar/                  # Reglas y validador de gramática barroca
│   │   ├── evaluation/               # Exactitud, perplejidad, validez de transiciones
│   │   ├── corpus/                   # Matriz de transiciones
│   │   ├── analysis/                 # Análisis no supervisado del espacio de embeddings
│   │   └── utils/                    # Wrappers de music21, utilidades de red
│   ├── scripts/
│   │   ├── baroque_corpus_etl/       # Pipeline ETL completo (Bronze → Silver → Gold)
│   │   │   ├── extract/              # Descarga de KernScores, Mutopia, Kunst der Fuge, jsbach.net
│   │   │   ├── transform/            # Auditoría, decodificación de nombres, plan de renombrado
│   │   │   └── load/                 # Carga al corpus, metadatos, embeddings, Supabase
│   │   ├── train_transformer.py      # Entrenamiento del MusicTransformer
│   │   ├── train_baseline.py         # Entrenamiento del LSTM baseline
│   │   ├── evaluate_transformer.py   # Métricas: Transformer vs. baseline
│   │   ├── evaluate_baseline.py      # Métricas: LSTM vs. modelo aleatorio
│   │   ├── extract_all.py            # Extracción del corpus → sequences.json
│   │   ├── batch_harmonic_analysis.py
│   │   ├── analyze_embeddings.py     # Clustering + figuras del espacio de embeddings
│   │   └── validate_chorales.py      # Smoke-test del pipeline en 10 corales BWV
│   ├── checkpoints/
│   │   └── transformer-v1/           # Checkpoint activo del Transformer
│   ├── reports/                      # Auditorías del corpus (bronze_audit.csv, etc.)
│   └── tests/                        # Suite pytest (442+ tests)
│
├── frontend/                         # Aplicación Next.js / React
│   ├── app/                          # Rutas y layout (App Router)
│   ├── components/
│   │   ├── bach-workbench.tsx        # Componente integrador principal
│   │   ├── views/
│   │   │   ├── composition-view.tsx  # Generación interactiva con controles de modelo
│   │   │   ├── analysis-view.tsx     # Visualización del análisis armónico
│   │   │   ├── library-view.tsx      # Exploración del corpus curado
│   │   │   └── settings-view.tsx
│   │   └── ui/                       # Sistema de diseño (shadcn/ui)
│   ├── hooks/
│   │   ├── use-midi-player.ts        # Reproducción MIDI en el navegador
│   │   ├── use-ai-chat.ts            # Chat con el asistente musicológico
│   │   ├── use-generation.ts         # Llamada a POST /api/generate
│   │   └── use-load-into-workbench.ts
│   └── lib/
│       ├── api-client.ts             # Cliente tipado para todos los endpoints
│       └── app-state.tsx             # Estado global de la aplicación
│
└── docker-compose.yml                # Levanta backend + frontend en un comando
```

---

## Levantar el entorno

### Opción A — Docker (recomendado)

Requiere Docker Desktop y un archivo `.env` en la raíz con las variables necesarias:

```bash
# .env (en la raíz del repositorio)
SUPABASE_URL=...
SUPABASE_SERVICE_ROLE_KEY=...
OPENAI_API_KEY=...
```

```bash
docker compose up --build
```

| Servicio | URL local |
|---|---|
| Backend (FastAPI + docs) | http://localhost:8000/docs |
| Frontend (Next.js) | http://localhost:3000 |

Para detener los servicios:

```bash
docker compose down
```

---

### Opción B — Entorno local con `uv`

Requiere Python 3.11+ y [`uv`](https://github.com/astral-sh/uv).

```bash
# 1. Clonar el repositorio
git clone <repo-url>
cd bach-propagation

# 2. Instalar dependencias del backend
cd backend
uv sync              # dependencias de producción
uv sync --extra dev  # + pytest y herramientas de desarrollo

# 3. Iniciar el servidor FastAPI
uv run uvicorn main:app --reload --port 8000

# 4. En otra terminal — instalar y arrancar el frontend
cd ../frontend
pnpm install
pnpm dev             # → http://localhost:3000
```

---

## Flujos de ETL y entrenamiento

### ETL del corpus barroco (Bronze → Silver → Gold)

Los scripts en `backend/scripts/baroque_corpus_etl/` implementan un pipeline de medallón:

```bash
# Bronze — descarga de fuentes primarias
cd backend
uv run python scripts/baroque_corpus_etl/extract/download_kernscores_bach.py
uv run python scripts/baroque_corpus_etl/extract/download_mutopia_baroque.py
uv run python scripts/baroque_corpus_etl/extract/download_kunstderfuge_p1.py
uv run python scripts/baroque_corpus_etl/extract/download_kunstderfuge_p2.py
uv run python scripts/baroque_corpus_etl/extract/download_jsbach_net.py

# Silver — auditoría, normalización de nombres y enriquecimiento de metadatos
uv run python scripts/baroque_corpus_etl/transform/audit_bronze.py
uv run python scripts/baroque_corpus_etl/transform/rename_plan.py
uv run python scripts/baroque_corpus_etl/transform/execute_rename.py
uv run python scripts/baroque_corpus_etl/transform/enrich_audit.py

# Gold — carga al corpus curado y subida a Supabase
uv run python scripts/baroque_corpus_etl/load/load_corpus.py
uv run python scripts/baroque_corpus_etl/load/extract_metadata.py
uv run python scripts/baroque_corpus_etl/load/generate_embeddings.py
uv run python scripts/baroque_corpus_etl/load/upload_to_storage.py
```

Los reportes de auditoría se generan automáticamente en `backend/reports/`.

---

### Entrenamiento del MusicTransformer (corpus barroco completo)

```bash
cd backend

# 1. Extraer secuencias armónicas del corpus
uv run python scripts/extract_all.py \
    --midi-dir data/encoded \
    --output data/encoded/sequences.json \
    --augment          # transpone a los 12 tonos

# 2. Construir el vocabulario armónico
uv run python scripts/build_harmonic_vocab.py \
    --sequences data/encoded/sequences.json \
    --output data/encoded/vocab.json

# 3. Entrenar el Transformer
uv run python scripts/train_transformer.py \
    --sequences data/encoded/sequences.json \
    --vocab data/encoded/vocab.json \
    --output checkpoints/transformer-v1

# 4. Evaluar
uv run python scripts/evaluate_transformer.py \
    --checkpoint checkpoints/transformer-v1/best.pt \
    --sequences data/encoded/sequences.json
```

---

### Entrenamiento del LSTM baseline (instrumento solo)

```bash
cd backend

# Extraer secuencias del subcorpus de instrumento solo
uv run python scripts/extract_all.py \
    --midi-dir data/samples/01_bach \
    --output data/sequences.json

# Entrenar
uv run python scripts/train_baseline.py \
    --sequences data/sequences.json \
    --epochs 30 \
    --checkpoint model_best.pt

# Evaluar (LSTM vs. modelo aleatorio)
uv run python scripts/evaluate_baseline.py \
    --checkpoint model_best.pt \
    --sequences data/sequences.json

# Analizar espacio de embeddings
uv run python scripts/analyze_embeddings.py \
    --checkpoint model_best.pt \
    --sequences data/sequences.json \
    --output-dir results/embedding_analysis
```

---

## Tests

```bash
cd backend

uv run pytest                    # suite completa
uv run pytest tests/core/        # módulo específico
uv run pytest tests/harmonic/
uv run pytest tests/models/
uv run pytest --cov=src          # con cobertura
```

Los tests que dependen de archivos MIDI locales se saltan automáticamente si esos archivos no están presentes.

---

## Datos

| Fuente | Acceso | Uso |
|---|---|---|
| Corpus music21 integrado | Automático (sin descarga) | `validate_chorales.py`, tests de integración |
| Subcorpus MIDI instrumento solo | [Google Drive](https://drive.google.com/drive/folders/1xVFWoCIXrHlDUlijDBDBJGywb6344sfT?usp=sharing) → `backend/data/samples/` | Entrenamiento del LSTM baseline |
| Corpus barroco ETL | Scripts `baroque_corpus_etl/extract/` | Entrenamiento del MusicTransformer |

---

## Variables de entorno

| Variable | Descripción |
|---|---|
| `SUPABASE_URL` | URL del proyecto Supabase (almacenamiento del corpus Gold) |
| `SUPABASE_SERVICE_ROLE_KEY` | Clave de servicio con permisos de escritura |
| `OPENAI_API_KEY` | Requerida para el router `/api/ai/chat` |
| `FRONTEND_URL` | URL del frontend en producción (CORS); por defecto `http://localhost:3000` |

---

## Dependencias principales

| Paquete | Versión mínima | Uso |
|---|---|---|
| `fastapi` | 0.109 | API REST del backend |
| `music21` | 9.1 | Parseo y análisis de partituras |
| `torch` | 2.10 | Transformer y LSTM |
| `numpy` | 2.4.2 | Operaciones numéricas |
| `scikit-learn` | 1.5 | Clustering y métricas de embeddings |
| `scipy` | 1.14 | Correlación tonal (Mantel test) |
| `matplotlib` | 3.9 | Figuras de análisis |
| `supabase` | 2.28 | Almacenamiento del corpus Gold |
| `pandas` | 3.0 | Auditoría y reportes ETL |
| `openai` | 2.30 | LLM para el chat musicológico |
