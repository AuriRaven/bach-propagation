# bach-propagation

Pipeline de investigación para analizar y modelar el lenguaje armónico de Bach. El sistema toma archivos MIDI o MusicXML en bruto, extrae secuencias armónicas a través de un pipeline de análisis de múltiples etapas, entrena un LSTM como modelo base sobre esas secuencias y realiza un análisis no supervisado del espacio de embeddings de acordes aprendido.

---

## Visión general del pipeline

```
MIDI / MusicXML
      │
      ▼
  ScoreLoader          ← parseo a representación interna Score
      │
      ▼
  Quantizer            ← ajuste de onsets/duraciones a grilla rítmica
      │
      ▼
  Windower             ← segmentación en ventanas de análisis
      │
      ▼
  KeyEstimator         ← detección de tonalidad (Krumhansl-Schmuckler)
      │
      ▼
  ChordClassifier      ← detección de acordes por plantillas
      │
      ▼
  RomanNumeralAnalyzer ← Acorde + Tonalidad → cifrado romano
      │
      ▼
  FunctionLabeler      ← función armónica T / PD / D
      │
      ▼
  ProlongationAnalyzer ← árbol de reducción schenkeriana en 3 niveles
      │
      ▼
  SequenceExtractor    ← codificación de eventos a secuencias de tokens
      │
      ▼
  BaroqueHarmonyLSTM   ← modelo de predicción del siguiente acorde
      │
      ▼
  EmbeddingAnalysis    ← clustering no supervisado del espacio de embeddings
```

---

## Estructura del repositorio

```
bach-propagation/
├── src/
│   ├── core/
│   │   ├── data_structures.py       # Todos los tipos del dominio musical (tiempo con Fraction)
│   │   └── score_loader.py          # MIDI/MusicXML → Score
│   ├── temporal/
│   │   ├── meter_analyzer.py        # Grillas de peso métrico jerárquico
│   │   ├── quantizer.py             # Ajuste de onsets y duraciones
│   │   └── windower.py              # Segmentación (compás / tiempo / deslizante)
│   ├── harmonic/
│   │   ├── key_estimator.py         # Estimación de tonalidad global y local
│   │   ├── chord_classifier.py      # Detección de acordes por plantillas
│   │   ├── roman_numeral_analyzer.py
│   │   └── function_labeler.py      # Etiquetado T/PD/D con sobreescrituras contextuales
│   ├── schenkerian/
│   │   └── prolongation.py          # ProlongationAnalyzer (árbol de reducción en 3 niveles)
│   ├── data/
│   │   ├── sequence_extractor.py    # Pipeline → dicts de eventos codificados
│   │   ├── encoder.py               # Vocabulario de 90 tokens (acorde/función/tonalidad)
│   │   └── dataset.py               # HarmonyDataset, partición 80/20 train-val
│   ├── models/
│   │   ├── baseline_lstm.py         # BaroqueHarmonyLSTM (embed=64, hidden=128, 2 capas)
│   │   └── train.py                 # Bucle de entrenamiento con early stopping
│   ├── evaluation/
│   │   └── metrics.py               # Exactitud, perplejidad, validez de transiciones, ARI
│   ├── analysis/
│   │   └── embedding_analysis.py    # Análisis no supervisado del espacio de embeddings
│   └── utils/
│       └── music21_helpers.py       # Wrappers ligeros de music21
├── scripts/
│   ├── extract_all.py               # Extracción en lote del corpus → sequences.json
│   ├── train_baseline.py            # Entrenamiento del LSTM → model_best.pt
│   ├── evaluate_baseline.py         # Tabla de métricas: LSTM vs. línea base aleatoria
│   ├── analyze_embeddings.py        # Clustering de embeddings + figuras
│   └── validate_chorales.py         # Validación del pipeline en 10 corales BWV
├── tests/                           # Suite pytest (un subpaquete por módulo de src)
├── data/
│   └── samples/01_bach/             # Archivos MIDI locales (opcionales — ver más abajo)
├── results/                         # Salidas de los scripts (en .gitignore)
├── CONTEXT.md                       # Referencia técnica completa para sesiones con LLM
├── pyproject.toml
└── uv.lock
```

---

## Instalación

Requiere Python 3.11+ y [uv](https://github.com/astral-sh/uv).

```bash
git clone <repo-url>
cd bach-propagation
uv sync
```

Con dependencias de desarrollo (para correr los tests):

```bash
uv sync --extra dev
```

---

## Correr los tests

```bash
# Suite completa
uv run pytest

# Módulo individual
uv run pytest tests/core/
uv run pytest tests/harmonic/
uv run pytest tests/models/
uv run pytest tests/analysis/

# Con cobertura
uv run pytest --cov=src
```

Los tests que dependen de archivos MIDI locales se saltan automáticamente si esos archivos no están presentes.

---

## Correr los scripts

Todos los scripts se ejecutan desde la raíz del repositorio con `uv run`.

### 1 — Extraer secuencias armónicas de un corpus MIDI

```bash
uv run python scripts/extract_all.py \
    --midi-dir data/samples/01_bach \
    --output data/sequences.json

# Con aumentación de altura (transpone cada pieza a los 12 tonos)
uv run python scripts/extract_all.py \
    --midi-dir data/samples/01_bach \
    --output data/sequences.json \
    --augment
```

Produce `data/sequences.json` — una lista de secuencias de dicts de eventos usada para el entrenamiento.

### 2 — Entrenar el LSTM de línea base

```bash
uv run python scripts/train_baseline.py \
    --sequences data/sequences.json \
    --epochs 30 \
    --checkpoint model_best.pt
```

Guarda el checkpoint con mejor pérdida de validación en `model_best.pt`.

### 3 — Evaluar el modelo entrenado

```bash
uv run python scripts/evaluate_baseline.py \
    --checkpoint model_best.pt \
    --sequences data/sequences.json
```

Imprime una tabla comparativa (LSTM vs. línea base aleatoria) con exactitud de acorde, exactitud de función, validez de transiciones y perplejidad.

### 4 — Analizar el espacio de embeddings de acordes

```bash
uv run python scripts/analyze_embeddings.py \
    --checkpoint model_best.pt \
    --sequences data/sequences.json \
    --output-dir results/embedding_analysis
```

Salidas:

| Archivo | Descripción |
|---|---|
| `purity_results.json` | Pureza, ARI, varianza PCA, correlación con círculo de quintas |
| `embeddings_2d.csv` | Coordenadas 2D (PCA) para los 60 tokens de acorde |
| `figures/function_ground_truth.png` | Scatter coloreado por función armónica |
| `figures/clusters_k3/5/7.png` | Scatter coloreado por cluster k-means |
| `figures/combined_analysis.png` | Figura 2×2 para la tesis (funciones + clusters k=3/5 + barras de pureza) |

### 5 — Validar el pipeline en corales BWV

```bash
uv run python scripts/validate_chorales.py
```

Ejecuta el pipeline completo de análisis sobre 10 corales del corpus integrado de music21 e imprime estadísticas de tonalidad, acorde y función por coral.

---

## Vocabulario de tokens

El `Encoder` mapea eventos a un vocabulario entero de 90 tokens:

| Rango | Tokens | Descripción |
|---|---|---|
| 0 | 1 | PAD |
| 1 | 1 | START |
| 2 | 1 | END |
| 3–62 | 60 | Acorde (12 raíces × 5 calidades: mayor, menor, dis., aum., dom7) |
| 63–65 | 3 | Función armónica (Tónica, Subdominante, Dominante) |
| 66–89 | 24 | Tonalidad (12 tónicos × 2 modos) |

Fórmula del token de acorde: `3 + raíz × 5 + índice_calidad`.

Las calidades no canónicas se colapsan antes de codificar: `major7 → major`, `minor7 → minor`, `dim7/half-dim7 → diminished`.

---

## Decisiones de diseño clave

- **`fractions.Fraction` para todos los valores temporales** — sin errores de redondeo flotante en aritmética de onsets y duraciones.
- **`separate_voices=False` para MIDI solista** — MuseScore exporta incluso piezas monofónicas como 3–4 tracks MIDI; aplanar a voz 0 evita voces fantasma.
- **Puntuación de acordes con media geométrica** — `raw / sqrt(total × n_notas)` equilibra cobertura y completitud; los acordes de séptima ganan solo cuando las 4 notas están presentes.
- **Detección de dominantes secundarias** — un acorde se marca como V/x únicamente si `(raíz + 5) % 12` coincide con una tónica diatónica y la calidad del acorde difiere de la calidad diatónica esperada para ese grado.
- **Etiquetas de función por voto mayoritario** — `build_chord_function_map` del encoder asigna a cada token de acorde su función armónica más frecuente en el corpus de entrenamiento; usada como verdad de referencia en la evaluación no supervisada.
