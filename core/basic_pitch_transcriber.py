import os
import numpy as np
from scipy.io import wavfile

def is_available():
    try:
        from basic_pitch.inference import predict
        return True
    except ImportError:
        return False


def midi_to_freq(midi_pitch):
    return 440.0 * (2.0 ** ((midi_pitch - 69.0) / 12.0))

def generate_sine_wave(frequency, duration, sample_rate=44100, amplitude=0.3):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    fade_len = int(sample_rate * 0.005)
    if len(wave) > fade_len * 2:
        envelope = np.ones_like(wave)
        envelope[:fade_len] = np.linspace(0, 1, fade_len)
        envelope[-fade_len:] = np.linspace(1, 0, fade_len)
        wave *= envelope
    return wave

def basic_pitch_transcribe(input_path, output_path):
    """
    Transcribes audio to midi and then synthesizes a sine wave version.
    Returns the absolute path to the generated WAV file.
    """
    from basic_pitch.inference import predict
    _, _, note_events = predict(input_path)
    
    if not note_events:
        return None

    max_time = max(note[1] for note in note_events)
    sample_rate = 44100
    total_samples = int((max_time + 0.5) * sample_rate)
    song_buffer = np.zeros(total_samples, dtype=np.float32)

    for note in note_events:
        start_s, end_s, pitch, amp, bend = note
        duration = end_s - start_s
        if duration <= 0:
            continue
            
        freq = midi_to_freq(pitch)
        wave = generate_sine_wave(freq, duration, sample_rate, amplitude=amp * 0.2)
        
        start_sample = int(start_s * sample_rate)
        end_sample = start_sample + len(wave)
        
        if end_sample <= len(song_buffer):
            song_buffer[start_sample:end_sample] += wave

    max_val = np.max(np.abs(song_buffer))
    if max_val > 0:
        song_buffer = song_buffer / max_val * 0.8

    wav_data = (song_buffer * 32767).astype(np.int16)
    wavfile.write(output_path, sample_rate, wav_data)
    
    return os.path.abspath(output_path)
