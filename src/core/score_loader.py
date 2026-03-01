"""
Score loader module for converting MIDI/MusicXML files into our data structures.

This module provides a robust wrapper around music21 to handle various input formats
and edge cases while producing clean, validated Score objects.
"""

from pathlib import Path
from fractions import Fraction
from typing import Optional, List, Tuple, Dict
import logging

try:
    from music21 import converter, stream, note, chord as m21_chord, meter, tempo
    from music21.stream import Score as M21Score, Part, Measure, Voice
except ImportError:
    raise ImportError(
        "music21 is required for score loading. Install with: pip install music21"
    )

from .data_structures import (
    Score,
    MusicalEvent,
    TimeSignature,
)

logger = logging.getLogger(__name__)


class ScoreLoadError(Exception):
    """Raised when a score cannot be loaded."""
    pass


class ScoreLoader:
    """
    Loads musical scores from MIDI or MusicXML files into Score objects.
    
    This class handles:
    - Multiple file formats (MIDI, MusicXML, etc.)
    - Multi-voice and multi-part scores
    - Time signature detection and changes
    - Proper temporal alignment
    - Beat strength calculation
    - Metadata extraction
    """
    
    def __init__(
        self,
        quantize_tolerance: Optional[Fraction] = None,
        separate_voices: bool = True,
    ):
        """
        Initialize the score loader.

        Args:
            quantize_tolerance: Optional tolerance for quantizing note onsets.
                              If None, no quantization is performed.
            separate_voices: When True, preserve separate voices/parts (needed
                           for chorales). When False, flatten everything to
                           voice 0 (appropriate for monophonic scores like
                           cello suites).
        """
        self.quantize_tolerance = quantize_tolerance
        self.separate_voices = separate_voices
        self._cached_time_sigs: Optional[List[Tuple[Fraction, TimeSignature]]] = None
        self._cached_time_sigs_score_id: Optional[int] = None
    
    def load(self, filepath: str | Path) -> Score:
        """
        Load a score from a file.
        
        Args:
            filepath: Path to MIDI or MusicXML file
            
        Returns:
            Score object with all musical events and metadata
            
        Raises:
            ScoreLoadError: If the file cannot be loaded or parsed
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise ScoreLoadError(f"File not found: {filepath}")
        
        # Validate file extension
        valid_extensions = {'.mid', '.midi', '.xml', '.mxl', '.musicxml'}
        if filepath.suffix.lower() not in valid_extensions:
            raise ScoreLoadError(
                f"Unsupported file format: {filepath.suffix}. "
                f"Supported formats: {', '.join(valid_extensions)}"
            )
        
        logger.info(f"Loading score from: {filepath}")
        
        try:
            # Load score with music21
            m21_score = converter.parse(str(filepath))
            
            # Extract components
            events = self._extract_events(m21_score)
            time_signatures = self._extract_time_signatures(m21_score)
            metadata = self._extract_metadata(m21_score, filepath)
            
            # Create Score object
            score = Score(
                events=events,
                time_signatures=time_signatures,
                title=metadata.get('title'),
                composer=metadata.get('composer'),
                tempo=metadata.get('tempo'),
                metadata=metadata
            )
            
            logger.info(
                f"Loaded score: {len(events)} events, "
                f"{score.num_voices} voices, "
                f"duration={float(score.duration):.2f} quarters"
            )
            
            return score
            
        except Exception as e:
            raise ScoreLoadError(f"Failed to load {filepath}: {str(e)}") from e
    
    def _extract_events(self, m21_score: M21Score) -> List[MusicalEvent]:
        """
        Extract all musical events from a music21 score.
        
        Args:
            m21_score: music21 Score object
            
        Returns:
            List of MusicalEvent objects
        """
        events: List[MusicalEvent] = []

        # Get all parts (instruments/voices)
        parts = m21_score.parts if hasattr(m21_score, 'parts') else [m21_score]

        if not self.separate_voices:
            # Flatten everything into voice 0
            flat = m21_score.flatten()
            events.extend(self._extract_voice_events(flat, 0, m21_score))
            return events

        global_voice = 0
        for part_idx, part in enumerate(parts):
            voice_streams = self._get_voices(part)
            for voice_stream in voice_streams:
                voice_events = self._extract_voice_events(
                    voice_stream, global_voice, m21_score
                )
                events.extend(voice_events)
                global_voice += 1

        return events
    
    def _get_voices(self, part: Part) -> List[stream.Stream]:
        """
        Get all voices from a part, handling both single-voice and multi-voice parts.

        Iterates through measures, collects Voice objects, and builds per-voice
        streams with correct global offsets.

        Args:
            part: music21 Part object

        Returns:
            List of voice streams (one per distinct voice found)
        """
        # Discover the maximum number of voices across all measures
        voice_ids: Dict[str, int] = {}
        for m in part.getElementsByClass(Measure):
            for v in m.voices:
                vid = v.id
                if vid not in voice_ids:
                    voice_ids[vid] = len(voice_ids)

        if not voice_ids:
            # Single-voice part — return the flattened part
            return [part.flatten()]

        # Build one stream per voice, collecting notes from each measure
        voice_streams: List[stream.Stream] = [stream.Stream() for _ in voice_ids]

        for m in part.getElementsByClass(Measure):
            measure_offset = m.offset
            m_voices = m.voices
            if m_voices:
                for v in m_voices:
                    idx = voice_ids.get(v.id)
                    if idx is None:
                        continue
                    for el in v.notesAndRests:
                        imported = el
                        voice_streams[idx].insert(
                            measure_offset + el.offset, imported
                        )
            else:
                # Measure has no explicit voices — assign to voice 0
                for el in m.notesAndRests:
                    voice_streams[0].insert(measure_offset + el.offset, el)

        return voice_streams
    
    def _extract_voice_events(
        self, 
        voice_stream: stream.Stream, 
        voice_num: int,
        m21_score: M21Score
    ) -> List[MusicalEvent]:
        """
        Extract events from a single voice.
        
        Args:
            voice_stream: music21 stream for this voice
            voice_num: Global voice number
            m21_score: Full score for context
            
        Returns:
            List of MusicalEvent objects
        """
        events = []
        
        # Get all notes and rests
        for element in voice_stream.flatten().notesAndRests:
            onset = Fraction(element.offset)
            duration = Fraction(element.quarterLength)
            
            # Get measure and beat information
            measure_num, beat_pos = self._get_measure_and_beat(element, m21_score)
            
            # Get time signature at this position
            time_sig = self._get_time_signature_at_offset(m21_score, onset)
            
            # Calculate beat strength
            beat_strength = self._calculate_beat_strength(beat_pos, time_sig)
            
            # Handle notes vs rests
            if isinstance(element, note.Rest):
                event = MusicalEvent(
                    pitch=None,
                    onset=onset,
                    duration=duration,
                    velocity=0,
                    measure=measure_num,
                    beat=beat_pos,
                    beat_strength=beat_strength,
                    voice=voice_num,
                    is_rest=True
                )
                events.append(event)
            
            elif isinstance(element, note.Note):
                event = MusicalEvent(
                    pitch=element.pitch.midi,
                    onset=onset,
                    duration=duration,
                    velocity=element.volume.velocity or 64,
                    measure=measure_num,
                    beat=beat_pos,
                    beat_strength=beat_strength,
                    voice=voice_num,
                    is_rest=False
                )
                events.append(event)
            
            elif isinstance(element, m21_chord.Chord):
                # Expand chords into individual notes with same onset
                for pitch in element.pitches:
                    event = MusicalEvent(
                        pitch=pitch.midi,
                        onset=onset,
                        duration=duration,
                        velocity=element.volume.velocity or 64,
                        measure=measure_num,
                        beat=beat_pos,
                        beat_strength=beat_strength,
                        voice=voice_num,
                        is_rest=False
                    )
                    events.append(event)
        
        return events
    
    def _get_measure_and_beat(
        self, 
        element: note.GeneralNote, 
        m21_score: M21Score
    ) -> Tuple[int, Fraction]:
        """
        Get the measure number and beat position for an element.
        
        Args:
            element: music21 note or rest
            m21_score: Full score for context
            
        Returns:
            Tuple of (measure_number, beat_position)
        """
        # Try to get measure number from the element's context
        try:
            measure = element.getContextByClass(Measure)
            if measure and measure.number is not None:
                measure_num = measure.number
            else:
                measure_num = 1
        except:
            measure_num = 1
        
        # Calculate beat position within measure
        try:
            measure = element.getContextByClass(Measure)
            if measure:
                beat_pos = Fraction(element.offset - measure.offset)
            else:
                # Fallback: use offset modulo measure length
                time_sig = self._get_time_signature_at_offset(
                    m21_score, 
                    Fraction(element.offset)
                )
                measure_length = Fraction(
                    time_sig.numerator * 4,
                    time_sig.denominator
                )
                beat_pos = Fraction(element.offset) % measure_length
        except:
            beat_pos = Fraction(0)
        
        return measure_num, beat_pos
    
    def _calculate_beat_strength(
        self, 
        beat_pos: Fraction, 
        time_sig: TimeSignature
    ) -> float:
        """
        Calculate metric strength for a position in a measure.
        
        This is a preliminary implementation. Will be replaced by
        meter_analyzer.py for more sophisticated metric hierarchy.
        
        Args:
            beat_pos: Position within measure in quarter notes
            time_sig: Current time signature
            
        Returns:
            Beat strength from 0.0 (weakest) to 1.0 (downbeat)
        """
        # Downbeat is always strongest
        if beat_pos == 0:
            return 1.0
        
        # Convert position to beat units
        beat_duration = time_sig.beat_duration
        beat_number = beat_pos / beat_duration
        
        # Simple metric hierarchy based on beat position
        if time_sig.is_compound:
            # Compound meters: strong on 1, medium on other main beats
            beat_in_measure = int(beat_number)
            if beat_in_measure == 0:
                return 1.0
            elif beat_number == int(beat_number):  # On main beat
                return 0.6
            else:
                return 0.3
        else:
            # Simple meters
            beat_in_measure = int(beat_number)
            subdivision = beat_number - beat_in_measure
            
            if beat_in_measure == 0:
                return 1.0
            elif beat_in_measure == 2 and time_sig.numerator == 4:
                # Beat 3 in 4/4
                return 0.5
            elif subdivision == 0:
                # On a main beat
                return 0.6
            elif subdivision == 0.5:
                # Eighth note subdivision
                return 0.3
            else:
                # Finer subdivision
                return 0.2
    
    def _extract_time_signatures(
        self, 
        m21_score: M21Score
    ) -> List[Tuple[Fraction, TimeSignature]]:
        """
        Extract all time signatures and their positions from the score.
        
        Args:
            m21_score: music21 Score object
            
        Returns:
            List of (onset, TimeSignature) tuples
        """
        time_signatures = []
        
        # Get all time signature elements
        for ts in m21_score.flatten().getElementsByClass(meter.TimeSignature):
            onset = Fraction(ts.offset)
            time_sig = TimeSignature(
                numerator=int(ts.numerator),
                denominator=int(ts.denominator)
            )
            time_signatures.append((onset, time_sig))
        
        # If no time signatures found, default to 4/4
        if not time_signatures:
            logger.warning("No time signature found, defaulting to 4/4")
            time_signatures.append((Fraction(0), TimeSignature(4, 4)))
        
        return time_signatures
    
    def _get_time_signature_at_offset(
        self,
        m21_score: M21Score,
        onset: Fraction
    ) -> TimeSignature:
        """
        Get the time signature in effect at a given offset.

        Uses a per-score cache so that ``_extract_time_signatures`` is only
        called once per score rather than once per note.

        Args:
            m21_score: music21 Score object
            onset: Position in quarter notes

        Returns:
            TimeSignature in effect at that position
        """
        score_id = id(m21_score)
        if self._cached_time_sigs is None or self._cached_time_sigs_score_id != score_id:
            self._cached_time_sigs = self._extract_time_signatures(m21_score)
            self._cached_time_sigs_score_id = score_id

        current_ts = self._cached_time_sigs[0][1]
        for ts_onset, ts in self._cached_time_sigs:
            if ts_onset > onset:
                break
            current_ts = ts

        return current_ts
    
    def _extract_metadata(
        self, 
        m21_score: M21Score, 
        filepath: Path
    ) -> Dict[str, any]:
        """
        Extract metadata from the score.
        
        Args:
            m21_score: music21 Score object
            filepath: Path to the source file
            
        Returns:
            Dictionary of metadata
        """
        metadata = {
            'source_file': str(filepath),
            'file_format': filepath.suffix.lower()
        }
        
        # Extract title
        if hasattr(m21_score, 'metadata') and m21_score.metadata:
            if m21_score.metadata.title:
                metadata['title'] = m21_score.metadata.title
            if m21_score.metadata.composer:
                metadata['composer'] = m21_score.metadata.composer
        
        # Use filename as fallback title
        if 'title' not in metadata:
            metadata['title'] = filepath.stem
        
        # Extract tempo
        tempo_marks = m21_score.flatten().getElementsByClass(tempo.MetronomeMark)
        if tempo_marks:
            metadata['tempo'] = int(tempo_marks[0].number)
        
        # Count parts and measures
        parts = m21_score.parts if hasattr(m21_score, 'parts') else [m21_score]
        metadata['num_parts'] = len(parts)
        
        measures = m21_score.flatten().getElementsByClass(Measure)
        if measures:
            metadata['num_measures'] = len(measures)
        
        return metadata


def load_score(filepath: str | Path, **kwargs) -> Score:
    """
    Convenience function to load a score.
    
    Args:
        filepath: Path to MIDI or MusicXML file
        **kwargs: Additional arguments passed to ScoreLoader
        
    Returns:
        Score object
        
    Example:
        >>> score = load_score("bach_cello_suite_1.mid")
        >>> print(f"Loaded {len(score.events)} events")
    """
    loader = ScoreLoader(**kwargs)
    return loader.load(filepath)