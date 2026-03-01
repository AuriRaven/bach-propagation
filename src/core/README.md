# `src/core` — Core Data Layer

This package defines the fundamental data structures used throughout the pipeline and the loader that converts MIDI/MusicXML files into those structures.

---

## Modules

| Module | Purpose |
|--------|---------|
| `data_structures.py` | All musical domain types |
| `score_loader.py` | music21 → `Score` conversion |

---

## `data_structures.py`

All temporal values use `fractions.Fraction` for exact arithmetic — no floating-point drift.

### Enums

#### `Accidental`
Standard accidentals: `DOUBLE_FLAT (-2)`, `FLAT (-1)`, `NATURAL (0)`, `SHARP (1)`, `DOUBLE_SHARP (2)`.

#### `ProlongationLevel`
Used for Schenkerian-style structural analysis.

| Value | Meaning |
|-------|---------|
| `FOREGROUND` | Surface-level events |
| `MIDDLEGROUND` | Intermediate structural level |
| `BACKGROUND` | Deep harmonic skeleton |

#### `HarmonicFunction`
Broad functional label for a chord or segment.

| Value | Meaning |
|-------|---------|
| `TONIC` | Tonic function (I, vi, III) |
| `PREDOMINANT` | Predominant function (IV, ii) |
| `DOMINANT` | Dominant function (V, vii°) |
| `AMBIGUOUS` | Context-dependent |

---

### Frozen Dataclasses

#### `TimeSignature(numerator, denominator)`
Immutable. `denominator` must be a power of 2 (1, 2, 4, 8, 16, 32).

Key computed properties:

| Property | Returns |
|----------|---------|
| `is_compound` | `True` if numerator > 3 and divisible by 3 (e.g. 6/8, 12/8) |
| `beats_per_measure` | Number of beats (divides by 3 for compound) |
| `beat_duration` | `Fraction` — duration of one beat in quarter notes |

```python
TimeSignature(4, 4).beat_duration   # → Fraction(1)
TimeSignature(6, 8).beat_duration   # → Fraction(3, 2)  (dotted quarter)
```

#### `RomanNumeral(degree, quality, inversion, local_key_tonic, local_key_mode, ...)`
Represents a Roman numeral analysis within a key.

| Field | Type | Description |
|-------|------|-------------|
| `degree` | `int` | Scale degree 1–7 |
| `quality` | `str` | `'major'`, `'minor'`, `'diminished'`, `'augmented'`, `'dominant7'`, `'dim7'`, `'half-dim7'`, `'major7'`, `'minor7'` |
| `inversion` | `int` | 0 = root position |
| `local_key_tonic` | `int` | Pitch class 0–11 |
| `local_key_mode` | `str` | `'major'` or `'minor'` |
| `is_secondary` | `bool` | True for secondary dominants like V/V |
| `secondary_target` | `Optional[int]` | Scale degree of secondary target |
| `function` | `Optional[HarmonicFunction]` | Functional label |

The `label` property renders standard notation:

```python
RomanNumeral(5, 'dominant7', 0, 7, 'major').label         # "V7"
RomanNumeral(5, 'dominant7', 1, 7, 'major').label         # "V65"
RomanNumeral(7, 'diminished', 1, 0, 'major').label        # "viio6"
RomanNumeral(5, 'dominant7', 0, 7, 'major',
             is_secondary=True, secondary_target=5).label  # "V7/V"
```

**Known limitation:** For non-dominant seventh chord types (`dim7`, `half-dim7`, `major7`, `minor7`), the quality indicator (`o7`, `ø7`, `M7`) is not currently rendered — only the inversion figures appear.

#### `KeyEstimate(tonic, mode, confidence, onset)`
Result of a key-finding algorithm.

| Field | Type | Description |
|-------|------|-------------|
| `tonic` | `int` | Pitch class 0–11 |
| `mode` | `str` | `'major'` or `'minor'` |
| `confidence` | `float` | 0.0–1.0 |
| `onset` | `Fraction` | Position where this key applies |

```python
KeyEstimate(7, 'major', 0.92, Fraction(0)).label   # "G major"
KeyEstimate(2, 'minor', 0.85, Fraction(8)).label   # "D minor"
```

---

### Mutable Dataclasses

#### `MusicalEvent`
Atomic unit of the pipeline — a single note or rest.

| Field | Type | Description |
|-------|------|-------------|
| `pitch` | `Optional[int]` | MIDI pitch 0–127, or `None` for rests |
| `onset` | `Fraction` | Start time in quarter notes from score start |
| `duration` | `Fraction` | Length in quarter notes (must be > 0) |
| `velocity` | `int` | MIDI velocity 0–127 |
| `measure` | `int` | Measure number (1-indexed) |
| `beat` | `Fraction` | Beat position within the measure |
| `beat_strength` | `float` | Metric weight 0.0–1.0 |
| `voice` | `int` | Voice number (0-indexed) |
| `is_rest` | `bool` | True if this is a rest |

Computed properties: `offset` (onset + duration), `pitch_class` (pitch % 12).

Method: `transpose(semitones)` — returns a new event, pitch clamped to 0–127.

#### `Chord`
A detected chord with harmonic analysis.

Fields: `onset`, `duration`, `root` (pitch class), `quality`, `pitches` (frozenset of pitch classes), `bass`, `inversion`, `roman_numeral` (optional string label).

#### `HarmonicSegment`
A tonal section with consistent key context.

Fields: `start`, `end`, `key_tonic`, `key_mode`, `chords`, `roman_numerals`, `prolongation_level`.

#### `Score`
Top-level container. Events are sorted by `(onset, voice)` on construction.

| Property/Method | Description |
|-----------------|-------------|
| `duration` | `Fraction` — max `event.offset` across all events |
| `num_voices` | Count of distinct voice numbers |
| `pitch_range` | `(min_pitch, max_pitch)` tuple |
| `get_time_signature_at(onset)` | Returns the active `TimeSignature` |
| `get_events_in_range(start, end)` | Events where `onset < end AND offset > start` |
| `get_events_by_voice(voice)` | Filter by voice number |

---

## `score_loader.py`

### `ScoreLoader`

```python
loader = ScoreLoader(separate_voices=False)   # monophonic / flat
score  = loader.load("path/to/file.mid")
```

#### Constructor parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `separate_voices` | `True` | When `False`, flattens all parts to voice 0. When `True`, each MIDI track becomes a separate voice. |
| `quantize_tolerance` | `None` | Optional onset snapping during load (prefer using `Quantizer` separately). |

#### `separate_voices` behaviour

**`False` (monophonic mode):** Use for cello suites, violin sonatas, and any solo piece where you want a single melodic line. Calls `m21_score.flatten()` before event extraction.

**`True` (multi-voice mode):** Use for chorales and pieces where voice-leading matters. Each MIDI track / notated voice becomes a distinct voice number (0, 1, 2, …).

> **Note on MuseScore exports:** Even solo instrument MIDI files from MuseScore typically contain 3–4 MIDI tracks because MuseScore encodes each notated voice as a separate track (most will be nearly empty). With `separate_voices=True` on a cello suite you will get `num_voices == num_parts` (often 3), not 1.

#### Internal caching

`_get_time_signature_at_offset` caches the result of `_extract_time_signatures` keyed on `id(m21_score)`. This prevents O(n·m) traversals when loading scores with many events.

### Convenience function

```python
from src.core.score_loader import load_score

score = load_score("bach_prelude.mid", separate_voices=False)
```

---

## Design decisions

- **`Fraction` everywhere for time:** Avoids floating-point accumulation errors across hundreds of events. All downstream modules (quantizer, windower, meter analyzer) rely on this.
- **Frozen dataclasses for value types:** `TimeSignature`, `RomanNumeral`, `KeyEstimate` are frozen so they can safely be used as dict keys or set members.
- **`Score.__post_init__` sorts events:** Any downstream consumer can assume events are in `(onset, voice)` order without re-sorting.
- **`get_events_in_range` overlap semantics:** An event "is in" a window if it *sounds* during that window (`onset < end AND offset > start`). This means events that start before the window but extend into it are included.
