import os
import librosa
import numpy as np

def is_available():
    try:
        import vamp
        return True
    except ImportError:
        return False

def vamp_chord_transcribe(audio_path):
    """
    Uses Vamp plugin system with Chordino (qm-vamp-plugins:qm-chordtranscriber).
    """
    import vamp
    
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=44100)
        
        # Chordino plugin identifier
        # Note: 'qm-vamp-plugins:qm-chordtranscriber' is the standard identifier
        # It requires the plugin to be installed in the system vamp path.
        
        data = vamp.collect(y, sr, "qm-vamp-plugins:qm-chordtranscriber")
        
        # data['list'] contains the chord segments
        chords = []
        if 'list' in data:
            for item in data['list']:
                # The plugin usually gives 'label' and 'timestamp'
                # Duration is sometimes inferred or given in the next item's timestamp
                chords.append({
                    'start': float(item['timestamp']),
                    'chord': item['label']
                })
            
            # Fill in 'end' times
            duration = librosa.get_duration(y=y, sr=sr)
            for i in range(len(chords) - 1):
                chords[i]['end'] = chords[i+1]['start']
            if chords:
                chords[-1]['end'] = float(duration)
                
            return chords
        return None
    except Exception as e:
        print(f"Vamp/Chordino Error: {e}")
        # Return None to signal failure, so we can fallback if needed
        return None
