#!/usr/bin/env python3
"""
Standalone Chord Transcription Script
=====================================
Uses NNLS Chroma with Bass/Treble Separation for 16th-note chord resolution.

Prerequisites:
--------------
- Python 3.11+
- FFmpeg (for audio loading)
- Python Packages:
    pip install numpy librosa scipy numba

Usage:
------
python transcribe_chords.py input_audio.wav [output_chords.json]
"""

import os
import sys
import json
import warnings
import numpy as np
import librosa
from scipy.optimize import nnls
from scipy.ndimage import uniform_filter1d

warnings.filterwarnings('ignore')

def is_available():
    try:
        import librosa
        import scipy
        return True
    except ImportError:
        return False

# --- CHORD DICTIONARY (Embedded) ---
CHORD_DICT_RAW = """
### Comma-Separated Chord Dictionaries
# field 1 is chord type name, 2-13 bass pitch (A-Ab), 14-25 treble pitch (A-Ab)

### Advanced Learners Chord Dictionary
=1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0
=0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0
m=1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0
m=0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0
dim7=0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,1,0,0,0,1,0
dim7=1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,1,0
6=1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,1,0,0
7=1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,1,0
maj7=1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,1
m7=1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0
m6=1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,1,0,0
=0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0
=0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0
dim=1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0
aug=1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0
=0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0
=0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0,1,0,0,0,0 
7=0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,1,0
"""

class ChordTranscriber:
    def __init__(self, sample_rate=44100, hop_size=2048):
        self.sr = sample_rate
        self.hop_size = hop_size
        self.chord_templates, self.chord_names = self._load_chord_dict()
        
    def _load_chord_dict(self):
        base_templates = []
        base_names = []
        for line in CHORD_DICT_RAW.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            chord_name, values_str = line.split('=', 1)
            values = [int(x) for x in values_str.split(',')]
            if len(values) == 24:
                base_templates.append(np.array(values, dtype=float))
                base_names.append(chord_name)

        all_templates = []
        all_names = []
        note_names = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
        
        for template_idx, base_template in enumerate(base_templates):
            chord_type = base_names[template_idx]
            for semitone in range(12):
                transposed = np.zeros(24)
                for k in range(12):
                    transposed[k] = base_template[(k - semitone + 12) % 12]
                    transposed[k + 12] = base_template[((k - semitone + 12) % 12) + 12]
                
                root_note = note_names[semitone]
                root_idx = semitone % 12
                bass_idx = -1
                for k in range(12):
                    if transposed[k] > 0.99:
                        bass_idx = k
                        break
                
                chord_name = root_note + chord_type
                if bass_idx != -1 and bass_idx != root_idx:
                    chord_name += "/" + note_names[bass_idx]
                
                all_templates.append(transposed)
                all_names.append(chord_name)
        
        # Add N (no chord)
        n_template = np.zeros(24)
        for k in range(12):
            n_template[k] = 0.5
            n_template[k + 12] = 1.0
        all_templates.append(n_template)
        all_names.append('N')
        
        normalized_templates = []
        for i, template in enumerate(all_templates):
            stand = np.power(np.sum(np.power(np.abs(template), 2.0)) / 24.0, 0.5)
            if all_names[i] == 'N': stand /= 1.1
            normalized_templates.append(template / stand if stand > 0 else template)
        
        return normalized_templates, all_names

    def extract_chroma(self, audio):
        bins_per_octave = 36
        cqt_treble = np.abs(librosa.cqt(y=audio, sr=self.sr, hop_length=self.hop_size,
                                      fmin=librosa.note_to_hz('C2'), n_bins=bins_per_octave * 4,
                                      bins_per_octave=bins_per_octave))
        cqt_bass = np.abs(librosa.cqt(y=audio, sr=self.sr, hop_length=self.hop_size,
                                    fmin=librosa.note_to_hz('C1'), n_bins=bins_per_octave * 2,
                                    bins_per_octave=bins_per_octave))

        def whiten(cqt_data):
            window = 37
            mu = uniform_filter1d(cqt_data, size=window, axis=0, mode='nearest')
            diff = np.maximum(cqt_data - mu, 0)
            sq_mu = uniform_filter1d(cqt_data**2, size=window, axis=0, mode='nearest')
            sigma = np.sqrt(np.maximum(sq_mu - mu**2, 1e-10))
            return diff / (sigma + 1e-10)

        cqt_treble, cqt_bass = whiten(cqt_treble), whiten(cqt_bass)

        def collapse(whitened):
            chroma = np.zeros((12, whitened.shape[1]))
            for b in range(whitened.shape[0]):
                chroma[(b // (bins_per_octave // 12)) % 12, :] += whitened[b, :]
            return chroma

        ct, cb = collapse(cqt_treble), collapse(cqt_bass)
        final = np.zeros((24, ct.shape[1]))
        for i in range(12):
            lib_idx = (i + 9) % 12
            tp, bp = np.max(ct, axis=0) + 1e-10, np.max(cb, axis=0) + 1e-10
            final[i, :], final[i + 12, :] = cb[lib_idx, :] / bp, ct[lib_idx, :] / tp
        return final.T

    def transcribe(self, audio, self_trans_prob=0.85):
        chroma_frames = self.extract_chroma(audio)
        tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=self.sr, hop_length=self.hop_size)
        
        sub_beat_frames = []
        for i in range(len(beat_frames) - 1):
            sub_beat_frames.extend(np.linspace(beat_frames[i], beat_frames[i+1], 5)[:-1].astype(int))
        
        chroma_subs = librosa.util.sync(chroma_frames.T, sub_beat_frames, aggregate=np.median).T
        n_subs, n_chords = chroma_subs.shape[0], len(self.chord_names)
        
        obs_matrix = np.zeros((n_subs, n_chords))
        for i in range(n_subs):
            sims = np.zeros(n_chords)
            for j in range(n_chords):
                sim = np.dot(chroma_subs[i], self.chord_templates[j])
                if self.chord_names[j] == 'N': sim *= 0.6
                sims[j] = np.power(1.6, max(0.0, min(200.0, sim)))
            obs_matrix[i] = sims / (np.sum(sims) + 1e-10)
        
        delta, psi = np.zeros((n_subs, n_chords)), np.zeros((n_subs, n_chords), dtype=int)
        delta[0, self.chord_names.index('N')] = 1.0
        delta[0] = delta[0] * obs_matrix[0]
        delta[0] /= (np.sum(delta[0]) + 1e-10)
        
        for t in range(1, n_subs):
            stay_p = self_trans_prob
            if (t % 16) == 0 or (t % 16) == 8: stay_p *= 0.8
            elif (t % 4) == 0: stay_p *= 0.9
            
            sw_p = (1.0 - stay_p) / (n_chords - 1)
            for j in range(n_chords):
                sc = delta[t-1] * sw_p
                sc[j] = delta[t-1, j] * stay_p
                psi[t, j] = np.argmax(sc)
                delta[t, j] = sc[psi[t, j]] * obs_matrix[t, j]
            delta[t] /= (np.sum(delta[t]) + 1e-10)
            
        path = np.zeros(n_subs, dtype=int)
        path[-1] = np.argmax(delta[-1])
        for t in range(n_subs - 2, -1, -1): path[t] = psi[t+1, path[t+1]]
            
        sub_times = librosa.frames_to_time(sub_beat_frames, sr=self.sr, hop_length=self.hop_size)
        estimates, start_time, curr_idx = [], 0.0, path[0]
        for i in range(1, len(path)):
            if path[i] != curr_idx:
                estimates.append({'label': self.chord_names[curr_idx], 'start': float(start_time), 'end': float(sub_times[i])})
                start_time, curr_idx = sub_times[i], path[i]
        estimates.append({'label': self.chord_names[curr_idx], 'start': float(start_time), 'end': float(librosa.get_duration(y=audio, sr=self.sr))})
        return estimates
def nnls_chord_transcribe(audio_path, return_beats=False):
    """
    High-level function to transcribe chords from an audio file.
    """
    audio, sr = librosa.load(audio_path, sr=44100)
    transcriber = ChordTranscriber(sample_rate=sr)
    
    # Get beats
    tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr, hop_length=transcriber.hop_size)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=transcriber.hop_size)
    
    estimates = transcriber.transcribe(audio)
    chords = [{'start': e['start'], 'end': e['end'], 'chord': e['label']} for e in estimates]
    
    if return_beats:
        return {
            'chords': chords,
            'beats': beat_times.tolist(),
            'tempo': float(tempo)
        }
    return chords


def main():
    if len(sys.argv) < 2:
        print("Usage: python transcribe_chords.py <audio_file> [output_json]")
        sys.exit(1)
        
    audio_path = sys.argv[1]
    print(f"Loading {audio_path}...")
    audio, sr = librosa.load(audio_path, sr=44100)
    
    print("Transcribing (16th-note resolution)...")
    transcriber = ChordTranscriber(sample_rate=sr)
    chords = transcriber.transcribe(audio)
    
    if len(sys.argv) > 2:
        with open(sys.argv[2], 'w') as f:
            json.dump(chords, f, indent=2)
        print(f"Saved to {sys.argv[2]}")
    else:
        for c in chords:
            print(f"{c['start']:6.2f}s - {c['end']:6.2f}s : {c['label']}")

if __name__ == "__main__":
    main()
