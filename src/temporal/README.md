# `src/temporal` — Temporal Analysis Pipeline

This package sits between score loading and harmonic analysis. It prepares a raw `Score` for downstream processing by analysing metric structure, snapping events to a grid, and slicing the score into analysis windows.

**Pipeline order:**

```
Score (from ScoreLoader)
  ↓
MeterAnalyzer  — compute beat weights per measure
  ↓
Quantizer      — snap onsets/durations to regular grid
  ↓
Windower       — slice into AnalysisWindows
  ↓
(harmonic analysis, key estimation, chord classification…)
```

---

## Modules

| Module | Purpose |
|--------|---------|
| `meter_analyzer.py` | Hierarchical beat-weight grids |
| `quantizer.py` | Onset/duration grid-snapping |
| `windower.py` | Score segmentation |

---

## `meter_analyzer.py`

### Weight hierarchy

```
1.0  — downbeat (beat 1)
0.8  — secondary strong beat (beat 3 in 4/4; beat 2 in 6/8 with 2 beats)
0.6  — other main beats
0.4  — half-beat (eighth-note in 4/4)
0.2  — sub-subdivision (sixteenth-note in 4/4)
0.1  — fallback for any position not on the grid
```

### `MetricWeight`
```python
@dataclass
class MetricWeight:
    position: Fraction   # offset within the measure in quarter notes
    level: int           # 0=downbeat … 4=sub-subdivision
    weight: float        # 0.2 – 1.0
```

### `MetricGrid`
Holds all `MetricWeight` objects for one measure.

```python
grid.weight_at(Fraction(0))     # → 1.0  (downbeat)
grid.weight_at(Fraction(2))     # → 0.8  (beat 3 in 4/4)
grid.weight_at(Fraction(99))    # → 0.1  (fallback)
```

### `MeterAnalyzer`

#### `build_metric_grid(time_sig, subdivisions=4) → MetricGrid`

Creates a grid covering every position from 0 to `measure_length` (exclusive) in steps of `beat_duration / subdivisions`.

Grid sizes (with default `subdivisions=4`):

| Time sig | Grid positions |
|----------|---------------|
| 4/4 | 16 (0, 1/4, 1/2, …, 15/4) |
| 3/4 | 12 |
| 6/8 | 8 (0, 3/8, 3/4, …, 21/8) |

**Secondary-strong-beat rules:**
- 4/4: beat index 2 (beat 3) is secondary strong.
- 6/8: beat index 1 (the second dotted-quarter) is secondary strong.
- Other meters with ≥ 5 beats: the mid-point beat is secondary strong.
- 3/4: no secondary strong beat (both inner beats are 0.6).

#### `get_beat_strength(position, time_sig) → float`
Convenience wrapper — builds the grid on the fly and queries it. For repeated queries within the same time signature, build the grid once and use `grid.weight_at()` directly.

```python
ma = MeterAnalyzer()
ts = TimeSignature(4, 4)

# One-off query
ma.get_beat_strength(Fraction(0), ts)   # 1.0

# Repeated queries — build once
grid = ma.build_metric_grid(ts)
for pos in positions:
    w = grid.weight_at(pos)
```

---

## `quantizer.py`

### `QuantizationConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `grid_resolution` | `Fraction(1, 4)` | Grid cell size in quarter notes (default = sixteenth note) |
| `tolerance` | `grid_resolution / 2` | Max distance from a grid point for snapping |

With the default `tolerance = grid / 2`, every onset snaps to the nearest grid point (no event can be more than half a grid step away). Use a smaller tolerance to leave imprecise events untouched:

```python
from fractions import Fraction
from src.temporal.quantizer import Quantizer, QuantizationConfig

default = QuantizationConfig()                                           # snaps everything
tight   = QuantizationConfig(grid_resolution=Fraction(1,4),
                              tolerance=Fraction(1,32))                  # only snaps if within 1/32
```

### `QuantizationStats`

Returned alongside the quantized score.

| Field | Description |
|-------|-------------|
| `num_events` | Total events processed |
| `num_quantized` | Events whose onset was moved |
| `max_onset_deviation` | Largest single correction |
| `total_onset_deviation` | Sum of all corrections |
| `mean_onset_deviation` | `total / num_quantized` (0 if nothing was moved) |

### `Quantizer`

#### `quantize_event(event, config) → MusicalEvent`
Returns a *new* event with snapped onset and duration. Non-temporal fields (`pitch`, `velocity`, `voice`, `beat_strength`, `is_rest`) are preserved unchanged.

**Snapping rules:**
- **Onset**: nearest grid point within `tolerance`; ties go to the lower grid point.
- **Duration**: always rounded to the nearest grid multiple; minimum is `grid_resolution`.

#### `quantize_score(score, config) → (Score, QuantizationStats)`
Quantizes all events, returns a new score and statistics.

```python
from src.temporal.quantizer import Quantizer, QuantizationConfig

config = QuantizationConfig()
q_score, stats = Quantizer().quantize_score(score, config)

print(f"{stats.num_quantized} events corrected")
print(f"mean deviation: {float(stats.mean_onset_deviation):.4f} quarters")
```

---

## `windower.py`

### `WindowType`

| Value | Strategy |
|-------|---------|
| `BEAT` | One window per beat |
| `MEASURE` | One window per measure |
| `FIXED_DURATION` | Non-overlapping windows of fixed size |
| `SLIDING` | Overlapping windows with configurable hop |

### `AnalysisWindow`

| Field | Type | Description |
|-------|------|-------------|
| `start` | `Fraction` | Window start in quarter notes |
| `end` | `Fraction` | Window end in quarter notes (exclusive) |
| `events` | `List[MusicalEvent]` | Events overlapping this window |
| `measure` | `int` | Measure number at `start` |
| `beat_position` | `Fraction` | Offset of `start` within its measure |

**Event inclusion:** an event is included if it sounds during the window — `event.onset < window.end AND event.offset > window.start`. This means events that begin before the window but extend into it are included, and events that begin in the window but extend past it are also included.

### `Windower`

#### `by_measure(score) → List[AnalysisWindow]`
One window per measure. Windows are contiguous and cover the full score duration. The last window is clipped to `score.duration` if the measure is not complete.

```python
windows = Windower().by_measure(score)
print(windows[0].start, windows[0].end)   # Fraction(0) Fraction(4)
```

#### `by_beat(score) → List[AnalysisWindow]`
One window per beat. Beat duration is determined by the active time signature at each point. Beat positions restart at each measure boundary.

```python
# 2 measures of 4/4 → 8 beat windows
windows = Windower().by_beat(score)
assert windows[0].beat_position == Fraction(0)
assert windows[4].beat_position == Fraction(0)   # new measure
```

#### `sliding(score, size, hop) → List[AnalysisWindow]`
Windows starting at 0, hop, 2·hop, … until the score end. The last window is clipped to `score.duration`.

```python
from fractions import Fraction
# 50% overlap
windows = Windower().sliding(score, size=Fraction(2), hop=Fraction(1))
```

#### `create_windows(score, window_type, size=None, hop=None) → List[AnalysisWindow]`
Dispatch helper. Raises `ValueError` if `size` is not provided for `FIXED_DURATION` or `SLIDING`. For `SLIDING`, `hop` defaults to `size / 2` when omitted.

```python
from src.temporal.windower import Windower, WindowType
from fractions import Fraction

# Measure-level windows
windows = Windower().create_windows(score, WindowType.MEASURE)

# Sliding with explicit hop
windows = Windower().create_windows(
    score, WindowType.SLIDING,
    size=Fraction(4), hop=Fraction(2)
)
```

---

## Design decisions

- **All modules are stateless:** `MeterAnalyzer`, `Quantizer`, and `Windower` hold no mutable state between calls. Instantiate once and reuse, or call as one-liners.
- **`Fraction` alignment:** All window boundaries snap to `Fraction` values matching the grid. This guarantees that `windows[i].end == windows[i+1].start` exactly (no floating-point drift).
- **Quantizer leaves uncertain events unchanged:** When an onset is farther from the nearest grid point than `tolerance`, it is not moved. This is safer than forcing every event to a grid, especially for MIDI recordings with human timing.
- **`MeterAnalyzer` is independent:** It does not read from `Score` — it only needs a `TimeSignature`. This lets you query beat weights for any hypothetical meter without having a score loaded.
- **`by_beat` and `by_measure` respect time signature changes:** Both methods call `score.get_time_signature_at(pos)` at each step, so pieces with mid-score meter changes are handled correctly.
