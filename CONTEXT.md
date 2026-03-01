# bach-propagation — Project Context

Paste this file at the start of a new chat to restore full context.

---

## Project goal

Build a Bach-style music generation system. The pipeline goes from raw MIDI/MusicXML input through temporal analysis, harmonic analysis (key estimation, chord classification, Roman numeral labelling), Schenkerian reduction, and ultimately a generative model that can compose in Bach's style.

Current branch: `feature/core`. Main branch: `main`.

---

## Tech stack

- **Python** (3.11+)
- **music21** ≥ 9.1 — score parsing, corpus access
- **numpy** ≥ 2.4
- **pytest** ≥ 8.0 with `scope="module"` fixtures
- **uv** for package management (`uv run pytest`)

---

## Repository layout

```
bach-propagation/
├── src/
│   ├── core/
│   │   ├── data_structures.py   # All musical domain types
│   │   ├── score_loader.py      # MIDI/MusicXML → Score
│   │   └── __init__.py          # Exports all public types
│   ├── temporal/
│   │   ├── meter_analyzer.py    # Hierarchical beat grids
│   │   ├── quantizer.py         # Onset/duration snapping
│   │   ├── windower.py          # Score segmentation
│   │   └── __init__.py
│   └── utils/
│       ├── music21_helpers.py   # Thin music21 wrappers
│       └── __init__.py
├── tests/
│   ├── conftest.py              # Shared fixtures + helper factories
│   ├── core/
│   │   ├── test_data_structures.py
│   │   └── test_score_loader.py
│   └── temporal/
│       ├── test_meter_analyzer.py
│       ├── test_quantizer.py
│       └── test_windower.py
├── data/
│   └── samples/
│       └── 01_bach/
│           └── solo_instruments/
│               ├── cello/cello_suites/bwv1007_.../01_prelude.mid
│               └── violin/sonatas/bwv1001_.../01_adagio.mid
├── pyproject.toml
└── uv.lock
```

---

## What is built and tested (177/177 tests passing)

### `src/core/data_structures.py`

All temporal values use `fractions.Fraction`.

**Enums:** `Accidental`, `ProlongationLevel` (FOREGROUND/MIDDLEGROUND/BACKGROUND), `HarmonicFunction` (TONIC/PREDOMINANT/DOMINANT/AMBIGUOUS)

**Frozen dataclasses (hashable, usable as dict keys):**
- `TimeSignature(numerator, denominator)` — with `is_compound`, `beats_per_measure`, `beat_duration` properties
- `RomanNumeral(degree, quality, inversion, local_key_tonic, local_key_mode, ...)` — `.label` property renders standard notation (e.g. `"V7"`, `"viio6"`, `"V7/V"`)
- `KeyEstimate(tonic, mode, confidence, onset)` — `.label` → `"G major"`

**Mutable dataclasses:**
- `MusicalEvent` — atomic note/rest; fields: `pitch`, `onset`, `duration`, `velocity`, `measure`, `beat`, `beat_strength`, `voice`, `is_rest`; properties: `offset`, `pitch_class`; method: `transpose(semitones)`
- `Chord` — detected chord with `root`, `quality`, `pitches` (frozenset), `bass`, `inversion`
- `HarmonicSegment` — tonal section with `key_tonic`, `key_mode`, `chords`, `roman_numerals`, `prolongation_level`
- `Score` — top-level container; sorts events by `(onset, voice)` on construction; key methods: `get_time_signature_at(onset)`, `get_events_in_range(start, end)`, `get_events_by_voice(voice)`; properties: `duration`, `num_voices`, `pitch_range`

**`get_events_in_range` overlap semantics:** includes event if `event.onset < end AND event.offset > start`.

### `src/core/score_loader.py`

```python
# Monophonic — flattens to voice 0
ScoreLoader(separate_voices=False).load("file.mid")

# Multi-voice — each MIDI track → distinct voice number
ScoreLoader(separate_voices=True).load("file.mid")
```

**Important:** MuseScore-exported MIDIs for solo instruments have 3–4 MIDI tracks even for monophonic pieces (voices 2/3/4 are nearly empty notated voices). With `separate_voices=True`, `score.num_voices` equals the number of MIDI tracks, not 1.

Internal detail: `_get_time_signature_at_offset` caches the result of `_extract_time_signatures` per `id(m21_score)` to avoid O(n) traversals per note.

### `src/utils/music21_helpers.py`

Thin wrappers that isolate the music21 dependency:

```python
load_m21_score(filepath)               # → music21.stream.Score
get_pitch_classes(stream)              # → List[int]
get_notes_in_range(stream, start, end) # → List[note.Note]
get_key_profile(stream)                # → (tonic_pc, mode, confidence)
pitch_class_to_name(pc)                # 0 → "C", 7 → "G"
name_to_pitch_class(name)              # "G" → 7
interval_between(pc1, pc2)             # semitone distance mod 12
load_bach_chorale(bwv=None)            # loads from music21 corpus
```

**Known fix:** `corpus.search(f'bwv{bwv}')` — do NOT pass `'bach'` as a second argument (it's treated as a field name, not a composer filter).

### `src/temporal/meter_analyzer.py`

```python
from src.temporal.meter_analyzer import MeterAnalyzer
from src.core.data_structures import TimeSignature
from fractions import Fraction

grid = MeterAnalyzer().build_metric_grid(TimeSignature(4, 4))
# 16 positions (sixteenth-note grid with subdivisions=4)
grid.weight_at(Fraction(0))    # 1.0  downbeat
grid.weight_at(Fraction(2))    # 0.8  secondary strong (beat 3 in 4/4)
grid.weight_at(Fraction(1))    # 0.6  main beat
grid.weight_at(Fraction(1,2))  # 0.4  half-beat
grid.weight_at(Fraction(1,4))  # 0.2  sub-subdivision
grid.weight_at(Fraction(99))   # 0.1  fallback
```

Weight levels: 1.0 (downbeat) → 0.8 (secondary strong) → 0.6 (main beat) → 0.4 (half-beat) → 0.2 (sub-subdivision) → 0.1 (fallback).

### `src/temporal/quantizer.py`

```python
from src.temporal.quantizer import Quantizer, QuantizationConfig
from fractions import Fraction

config = QuantizationConfig()                              # grid=1/4, tolerance=1/8
# With default tolerance=grid/2, every onset snaps to nearest grid point
q_score, stats = Quantizer().quantize_score(score, config)
print(stats.num_quantized, stats.mean_onset_deviation)
```

With `tolerance = grid / 2` (default), every onset snaps. Use a tighter `tolerance` to leave uncertain events untouched. Durations always round to nearest grid multiple (minimum = `grid_resolution`).

### `src/temporal/windower.py`

```python
from src.temporal.windower import Windower, WindowType
from fractions import Fraction

windows = Windower().by_measure(score)
windows = Windower().by_beat(score)
windows = Windower().sliding(score, size=Fraction(4), hop=Fraction(2))
windows = Windower().create_windows(score, WindowType.SLIDING,
                                    size=Fraction(4), hop=Fraction(2))
```

`AnalysisWindow` fields: `start`, `end`, `events`, `measure`, `beat_position`.

For `SLIDING`, omitting `hop` defaults to `size / 2`. For `FIXED_DURATION` or `SLIDING`, `size` is required (raises `ValueError` if missing).

---

## Test infrastructure

```
tests/conftest.py          — fixtures + factories
tests/core/                — pure unit tests, no file I/O
tests/temporal/            — pure unit tests with synthetic Score objects
```

**Fixtures (scope="module"):**
- `prelude_score` — BWV 1007 Prelude, `separate_voices=False`
- `adagio_score` — BWV 1001 Adagio, `separate_voices=True`
- Both skip automatically (`pytest.skip`) if MIDI files are missing

**Factories (importable from conftest):**
```python
make_note(pitch, onset, duration, voice=0, measure=1, beat_strength=1.0)
make_rest(onset, duration, voice=0, measure=1)
make_score(events, numerator=4, denominator=4)
```

Run tests:
```bash
uv run pytest
uv run pytest tests/core/
uv run pytest tests/temporal/
```

---

## Known design decisions to keep in mind

1. **`Fraction` for all time** — never use floats for onset/duration/beat positions.
2. **`separate_voices=False` for solo MIDI** — MuseScore exports even monophonic pieces as 3+ MIDI tracks.
3. **`MeterAnalyzer` is stateless** — build the grid once per `TimeSignature` and reuse it.
4. **`Quantizer` does not mutate** — `quantize_event` returns a new `MusicalEvent`.
5. **`RomanNumeral.label` has a known gap** — non-dominant 7th qualities (`dim7`, `half-dim7`, `major7`) don't render the quality suffix yet. Fix this when implementing the chord classifier.

---

## What comes next (not yet built)

- `src/harmonic/key_estimator.py` — produce `KeyEstimate` objects from `AnalysisWindow` pitch content
- `src/harmonic/chord_classifier.py` — detect chords, produce `Chord` objects
- `src/harmonic/roman_numeral_analyzer.py` — assign `RomanNumeral` labels to chords within a key
- `src/schenkerian/` — prolongation analysis, structural reduction
- `src/generative/` — generation model trained on the analysed corpus

Data flow: `Score → [Quantizer] → [Windower] → [KeyEstimator] → [ChordClassifier] → [RNAnalyzer] → [Schenkerian] → [Generator]`
