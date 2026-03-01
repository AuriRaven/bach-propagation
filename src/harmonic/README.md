# `src/harmonic/` — Harmonic Analysis Pipeline

This package implements the harmonic analysis layer of the bach-propagation pipeline. It transforms `AnalysisWindow` segments (from `src/temporal/`) into fully-labelled Roman numerals with harmonic function annotations.

---

## Pipeline Overview

```
Score (MusicalEvent list)
        │
        ▼
  KeyEstimator          ──► KeyEstimate (tonic, mode, confidence, onset)
        │
        ▼
  ChordClassifier       ──► List[Chord] (root, quality, inversion, onset, duration)
        │
        ▼
  RomanNumeralAnalyzer  ──► List[RomanNumeral] (degree, function, is_secondary, …)
        │
        ▼
  FunctionLabeler       ──► List[RomanNumeral] (function finalised with context)
```

All modules are stateless classes — instantiate once and reuse freely. All time values use `fractions.Fraction`.

---

## Module 1: `key_estimator.py`

### Algorithm: Krumhansl-Schmuckler (KS) Key-Finding

The KS algorithm correlates a pitch-class histogram of the input against 24 reference profiles (12 major + 12 minor) to identify the most probable key.

**Step 1 — Build pitch-class histogram**

For each non-rest `MusicalEvent`, accumulate `float(duration)` into `histogram[pitch % 12]`. Rests and `None` pitches are skipped. The result is a 12-element list weighted by note length.

**Step 2 — Pearson correlation against 24 profiles**

For each of 12 roots × 2 modes, rotate the KS profile to that root (`rotated[i] = profile[(i - root) % 12]`) and compute Pearson r against the histogram. Zero-variance histograms return r = 0.0.

**Step 3 — Confidence normalization**

```
confidence = (best_r + 1) / 2     # maps [-1, 1] → [0, 1]
```

A confidence of 1.0 means the histogram perfectly matches the winning profile. Values below ~0.5 indicate ambiguous or chromatic material.

**KS profiles (built into the module):**
```python
_KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
```

### Public API

```python
class KeyEstimator:
    def estimate(self, events: List[MusicalEvent], onset: Fraction) -> KeyEstimate
    def estimate_global(self, score: Score) -> KeyEstimate        # onset = Fraction(0)
    def estimate_local(self, score: Score,
                       window_size: Fraction,
                       hop: Fraction) -> List[KeyEstimate]
    # Private
    def _build_histogram(self, events: List[MusicalEvent]) -> List[float]
    def _correlate(self, histogram: List[float], profile: List[float]) -> float
```

`estimate_local` uses `Windower.sliding()` internally and returns one `KeyEstimate` per window, each carrying the window's start time as `onset`.

### Edge Cases

| Situation | Result |
|-----------|--------|
| Empty event list | `KeyEstimate(tonic=0, mode='major', confidence=0.0, onset=onset)` |
| Rest-only events | Histogram all-zero → same as empty |
| All-chromatic events | Low confidence (< 0.7 typical) |
| Zero-variance histogram | `_correlate` returns 0.0; confidence = 0.5 |

### Usage

```python
from fractions import Fraction
from src.harmonic.key_estimator import KeyEstimator

ke = KeyEstimator()
key = ke.estimate_global(score)
print(key.label)          # e.g. "G major"
print(key.confidence)     # e.g. 0.83

# Sliding-window local keys (16-quarter windows, 8-quarter hop)
local_keys = ke.estimate_local(score, Fraction(16), Fraction(8))
```

---

## Module 2: `chord_classifier.py`

### Algorithm: Weighted Pitch-Class Template Matching

**Step 1 — Build pitch-class profile**

For each non-rest event:
```
weight = beat_strength ** exponent * float(duration)
profile[pitch % 12] += weight
```

The `beat_weight_exponent` (default 1.0) controls how strongly metrically strong notes are emphasised. Setting it to 2.0 squares the beat strength, penalising off-beat notes.

**Step 2 — Score all candidates**

For each of 12 roots × 9 qualities (or 4 triads only if `include_seventh_chords=False`):

```
raw   = Σ profile[pc] for pc in chord_tones
score = raw / sqrt(total_profile_weight × n_chord_tones)
```

This **geometric-mean** scoring balances two factors simultaneously:
- **Coverage**: fraction of the window's note weight explained by the chord
- **Completeness**: fraction of the chord's tones actually present in the window

A dominant7 chord (4 tones) beats a major triad (3 tones) only when the window actually contains all 4 characteristic tones. If the 7th is missing, the lower `completeness` factor penalises the 7th chord.

**Step 3 — Confidence gate**

```
if best_score < config.min_confidence:
    return None
```

**Step 4 — Determine inversion**

Bass = lowest MIDI pitch among non-rest events. The bass pitch class is looked up in the sorted chord tones. Its position (offset from the root) gives the inversion number (0 = root, 1 = first, 2 = second, 3 = third for 7th chords). If the bass is not a chord tone, inversion defaults to 0.

### Chord Templates

```python
_CHORD_TEMPLATES = {
    'major':      frozenset({0, 4, 7}),
    'minor':      frozenset({0, 3, 7}),
    'diminished': frozenset({0, 3, 6}),
    'augmented':  frozenset({0, 4, 8}),
    'dominant7':  frozenset({0, 4, 7, 10}),
    'dim7':       frozenset({0, 3, 6, 9}),
    'half-dim7':  frozenset({0, 3, 6, 10}),
    'major7':     frozenset({0, 4, 7, 11}),
    'minor7':     frozenset({0, 3, 7, 10}),
}
```

Quality strings match `data_structures._QUALITY_SUFFIXES` exactly for label rendering.

### Configuration

```python
@dataclass
class ChordClassificationConfig:
    min_confidence: float = 0.3           # minimum score to yield a Chord
    include_seventh_chords: bool = True   # False → triads only
    beat_weight_exponent: float = 1.0     # 1.0=linear, 2.0=quadratic
```

### Public API

```python
class ChordClassifier:
    def classify_window(self, window: AnalysisWindow,
                        config: ChordClassificationConfig) -> Optional[Chord]
    def classify_score(self, score: Score, window_type: WindowType,
                       config: ChordClassificationConfig,
                       window_size: Optional[Fraction] = None,
                       hop: Optional[Fraction] = None) -> List[Chord]
    # Private
    def _build_pitch_class_profile(self, window, config) -> List[float]
    def _score_chord(self, profile, root, intervals) -> float   # raw sum only
    def _get_inversion(self, events, root, intervals) -> Tuple[int, int]
```

`classify_score` filters out `None` results and always returns chords in chronological order.

### Edge Cases

| Situation | Result |
|-----------|--------|
| Rest-only window | `None` immediately (profile sum = 0) |
| Empty window | `None` immediately |
| Score below `min_confidence` | `None` |
| Bass not a chord tone | `inversion = 0` (root-position best guess) |

### Usage

```python
from src.harmonic.chord_classifier import ChordClassifier, ChordClassificationConfig
from src.temporal.windower import WindowType

cc = ChordClassifier()
cfg = ChordClassificationConfig(min_confidence=0.3, include_seventh_chords=True)

# Per-measure chord classification
chords = cc.classify_score(score, WindowType.MEASURE, cfg)
for c in chords:
    print(f"{c.onset}: {c.label}")   # e.g. "0: CM  (root position)"
```

---

## Module 3: `roman_numeral_analyzer.py`

### Algorithm: Interval Arithmetic + Exception Patterns

**Step 1 — Compute interval from tonic**

```python
interval = (chord.root - key.tonic) % 12
```

**Step 2 — Look up diatonic degree**

Separate lookup tables for major and minor:

| Mode | Interval → Degree |
|------|-------------------|
| Major | 0→I, 2→II, 4→III, 5→IV, 7→V, 9→VI, 11→VII |
| Minor | 0→i, 2→ii, 3→III, 5→iv, 7→v, 8→VI, 10→VII, 11→VII |

Minor degree 7 maps both interval 10 (natural 7th) and 11 (harmonic leading tone) to degree 7.

**Step 3 — Check for secondary dominant (chromatic chords)**

A chord is a secondary dominant if:
1. Its quality is `major`, `dominant7`, or `diminished` (dominant-function quality)
2. Its quality differs from the expected diatonic quality for its diatonic degree (diatonic chords are excluded)
3. The pitch a 5th below the chord root (`resolution_pc = (chord.root + 5) % 12`) is the tonic of a diatonic degree

The formula `(root + 5) % 12` finds what key the chord dominates: D major (root=2) → resolution=(2+5)%12=7=G, so D major is V/V in C major. Diminished chords use `(root + 1) % 12` (leading tone → tonic is a semitone above).

**Step 4 — Check for borrowed chords**

A chord is borrowed (modal mixture) if its root is diatonic in the parallel mode and its quality matches the parallel-mode's expected diatonic quality for that degree. Example: F minor (root=5, quality=minor) in C major — F is diatonic in C minor as iv (degree 4, quality=minor) → borrowed.

**Step 5 — Assign harmonic function**

```python
_BASE_FUNCTION = {
    1: TONIC,        # I / i
    2: PREDOMINANT,  # ii / iio
    3: AMBIGUOUS,    # III / iii
    4: PREDOMINANT,  # IV / iv
    5: DOMINANT,     # V / v
    6: TONIC,        # VI / vi  (default; may be overridden by FunctionLabeler)
    7: DOMINANT,     # VII / viio
}
```

Secondary dominants always get `DOMINANT` regardless of degree.

### Module-Level Constants

```python
_MAJOR_INTERVAL_TO_DEGREE = {0:1, 2:2, 4:3, 5:4, 7:5, 9:6, 11:7}
_MINOR_INTERVAL_TO_DEGREE = {0:1, 2:2, 3:3, 5:4, 7:5, 8:6, 10:7, 11:7}
```

### Public API

```python
class RomanNumeralAnalyzer:
    def analyze(self, chord: Chord, key: KeyEstimate) -> RomanNumeral
    def analyze_sequence(self, chords: List[Chord],
                         keys: List[KeyEstimate]) -> List[RomanNumeral]
    # Private
    def _interval_to_degree(self, interval: int, mode: str) -> Optional[int]
    def _is_secondary_dominant(self, chord: Chord,
                               key: KeyEstimate) -> Tuple[bool, Optional[int]]
    def _is_borrowed(self, chord: Chord, key: KeyEstimate) -> bool
    def _assign_function(self, degree: int) -> HarmonicFunction
```

`analyze_sequence` matches each chord to the most-recent key whose `onset ≤ chord.onset`. If no key precedes the first chord, the earliest key is used. Raises `ValueError` if `keys` is empty.

### Label Rendering Examples

| Chord | Key | Label |
|-------|-----|-------|
| C major | C major | `I` |
| D minor | C major | `ii` |
| G major | C major | `V` |
| B diminished | C major | `viio` |
| D major | C major | `V/V` |
| F# diminished | C major | `viio/V` |
| F minor | C major | `iv` (borrowed) |

### Edge Cases

| Situation | Result |
|-----------|--------|
| Empty `keys` list | `ValueError` raised |
| Chord before first key onset | Uses earliest key |
| Chromatic root (not diatonic) | Checked as secondary or borrowed; degree=1 fallback |

### Usage

```python
from src.harmonic.roman_numeral_analyzer import RomanNumeralAnalyzer

rna = RomanNumeralAnalyzer()
key = ke.estimate_global(score)
rns = rna.analyze_sequence(chords, [key])
for rn in rns:
    print(f"{rn.label} ({rn.function.name})")   # e.g. "V (DOMINANT)"
```

---

## Module 4: `function_labeler.py`

### Algorithm: Two-Pass Sequence Labeling

**Pass 1 — Base labeling**

Each `RomanNumeral` with `function=None` is assigned the base function from the degree table above. Secondary dominants (`is_secondary=True`) always receive `DOMINANT` regardless of degree. Already-labelled RNs are returned unchanged.

**Pass 2 — Context-sensitive overrides**

The labeler applies a small override table based on adjacent pairs. Currently implemented:

```python
_CONTEXT_OVERRIDES = {
    (6, 2): PREDOMINANT,   # vi → ii  : vi relabelled PREDOMINANT
    (6, 4): PREDOMINANT,   # vi → IV  : vi relabelled PREDOMINANT
}
```

This captures the common voice-leading pattern where `vi` acts as a bridge into the predominant region rather than as a tonic extension. Override rules only affect the *preceding* chord; the following chord is relabelled purely by its own base function.

### Public API

```python
class FunctionLabeler:
    def label(self, rn: RomanNumeral) -> RomanNumeral
    # Returns rn unchanged (same object) if already labelled.
    # Returns new frozen RomanNumeral otherwise (uses dataclasses.replace).

    def label_sequence(self, rns: List[RomanNumeral]) -> List[RomanNumeral]
    # Does NOT mutate input list or RomanNumeral objects.
    # Returns a new list of new frozen instances.
```

### Usage

```python
from src.harmonic.function_labeler import FunctionLabeler

fl = FunctionLabeler()
labelled = fl.label_sequence(rns)
for rn in labelled:
    print(f"{rn.label:8s}  {rn.function.name}")
# I        TONIC
# IV       PREDOMINANT
# V        DOMINANT
# I        TONIC
```

---

## Using the Full Pipeline

```python
from fractions import Fraction
from src.core.score_loader import ScoreLoader
from src.harmonic import (
    ChordClassificationConfig, ChordClassifier,
    FunctionLabeler, KeyEstimator, RomanNumeralAnalyzer,
)
from src.temporal.windower import WindowType

# Load score
score = ScoreLoader(separate_voices=False).load("path/to/piece.mid")

# 1. Estimate key (global + local)
ke = KeyEstimator()
global_key = ke.estimate_global(score)
local_keys = ke.estimate_local(score, window_size=Fraction(16), hop=Fraction(8))

# 2. Classify chords (one per measure)
cfg = ChordClassificationConfig()
chords = ChordClassifier().classify_score(score, WindowType.MEASURE, cfg)

# 3. Assign Roman numerals (using local keys for modulation tracking)
rns = RomanNumeralAnalyzer().analyze_sequence(chords, local_keys)

# 4. Finalise harmonic functions
labelled = FunctionLabeler().label_sequence(rns)

# Inspect results
for rn in labelled[:8]:
    print(f"{float(rn.onset):5.1f}  {rn.label:10s}  {rn.function.name}")
```

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Stateless classes | Enables concurrent use across multiple scores without shared state |
| Geometric-mean chord scoring | Balances coverage (window notes explained) and completeness (template tones present), correctly handling 7th chords vs triads |
| `(root + 5) % 12` for secondary dominants | V of X means X is a perfect fifth above the dominant chord root — 5 semitones up (not 7) |
| Diatonic quality guard for secondary dominants | Prevents diatonic chords (e.g. I in C major) from being mis-classified as secondary dominants of themselves |
| Two-pass function labeling | Separates local degree semantics (Pass 1) from contextual voice-leading patterns (Pass 2) |
| `dataclasses.replace()` for updates | All data types are frozen; no mutation is permitted |
| `Fraction` throughout | Exact arithmetic for onset/duration avoids floating-point drift across long scores |

## Known Limitations

- **Modulation detection**: `RomanNumeralAnalyzer` does not detect modulations autonomously — it relies on the `KeyEstimate` list supplied by the caller. Use `KeyEstimator.estimate_local()` with a narrow window to track modulations.
- **Enharmonic equivalence**: All reasoning is in pitch-class space (0–11). Enharmonic distinctions (C# vs Db) are collapsed.
- **Borrowed chord detection**: Currently checks parallel mode only (e.g. major ↔ minor). Mixture from other modes (Dorian, Lydian, etc.) is not modelled.
- **`RomanNumeral.label` quality suffixes**: Non-dominant 7th qualities (e.g. major7, minor7) may not render a quality suffix in the label — see `data_structures._QUALITY_SUFFIXES` for the full map.
- **Polyphonic scores**: `classify_score` operates on a flat event list. Polyphonic scores should be flattened by the caller (`ScoreLoader(separate_voices=False)`) before classification.
