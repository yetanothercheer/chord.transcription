import os
from core.demucs_source_separator import demucs_source_separate
from core.whisper_lyrics_transcriber import whisper_lyrics_transcribe
from core.nnls_chord_transcriber import nnls_chord_transcribe, is_available as nnls_available
from core.vamp_chord_transcriber import vamp_chord_transcribe, is_available as vamp_available

def separate_sources(input_audio_path, media_root):
    """
    Separates audio into vocals and accompaniment using Demucs.
    Returns a dictionary with paths to 'vocals' and 'no_vocals'.
    """
    stems = demucs_source_separate(input_audio_path, media_root)
    if not stems or not stems.get('vocals') or not stems.get('no_vocals'):
        raise Exception("Source separation failed or returned incomplete results.")
    return stems

def transcribe_lyrics(vocals_path, media_root, language="zh"):
    """
    Transcribes lyrics from the vocals track using Whisper.
    Supports languages: 'en', 'zh', 'ja', etc.
    """
    return whisper_lyrics_transcribe(vocals_path, media_root, model_name="base", language=language)

def recognize_chords(accompaniment_path, algorithm='nnls'):
    """
    Recognizes chords from the accompaniment track using the specified algorithm.
    Also extracts beats and tempo, falling back to NNLS if necessary.
    """
    chord_results = None
    
    # helper for availability check to avoid clutter
    vamp_ok = vamp_available()

    if algorithm == 'vamp' and vamp_ok:
        chords = vamp_chord_transcribe(accompaniment_path)
        beat_info = nnls_chord_transcribe(accompaniment_path, return_beats=True)
        chord_results = {
            'chords': chords,
            'beats': beat_info['beats'],
            'tempo': beat_info['tempo']
        }
    
    # Fallback or explicit NNLS
    if chord_results is None:
        chord_results = nnls_chord_transcribe(accompaniment_path, return_beats=True)
        
    return chord_results
