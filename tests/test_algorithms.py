import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import os
import json
from pathlib import Path

# Import the algorithms
from core.basic_pitch_transcriber import midi_to_freq, generate_sine_wave, basic_pitch_transcribe
from core.demucs_source_separator import demucs_source_separate
from core.nnls_chord_transcriber import ChordTranscriber, nnls_chord_transcribe
from core.whisper_lyrics_transcriber import whisper_lyrics_transcribe
from core.pipeline import run_pipeline

class TestAlgorithms(unittest.TestCase):

    # --- Basic Pitch Transcriber Tests ---
    def test_midi_to_freq(self):
        self.assertAlmostEqual(midi_to_freq(69), 440.0)
        self.assertAlmostEqual(midi_to_freq(60), 261.6255653)

    def test_generate_sine_wave(self):
        duration = 0.1
        sr = 44100
        wave = generate_sine_wave(440, duration, sr)
        self.assertEqual(len(wave), int(duration * sr))
        self.assertTrue(np.max(np.abs(wave)) <= 0.3)

    @patch('basic_pitch.inference.predict')
    @patch('scipy.io.wavfile.write')
    def test_basic_pitch_transcribe(self, mock_wav_write, mock_predict):
        # Mocking predict to return dummy info
        # _, _, note_events = predict(input_path)
        mock_predict.return_value = (None, None, [[0.0, 1.0, 60, 0.5, 0]])
        
        input_path = "dummy.wav"
        output_path = "output.wav"
        
        result = basic_pitch_transcribe(input_path, output_path)
        self.assertTrue(result.endswith("output.wav"))
        mock_wav_write.assert_called()

    # --- Demucs Source Separator Tests ---
    @patch('subprocess.run')
    @patch('pathlib.Path.exists')
    def test_demucs_source_separate(self, mock_exists, mock_run):
        mock_exists.return_value = True
        mock_run.return_value = MagicMock(returncode=0)
        
        input_path = "song.mp3"
        media_root = "/tmp/media"
        
        result = demucs_source_separate(input_path, media_root)
        self.assertIsNotNone(result)
        self.assertIn("vocals", result)
        self.assertIn("no_vocals", result)

    # --- NNLS Chord Transcriber Tests ---
    def test_chord_transcriber_init(self):
        ct = ChordTranscriber()
        self.assertTrue(len(ct.chord_names) > 0)
        self.assertTrue(len(ct.chord_templates) > 0)

    @patch('librosa.cqt')
    @patch('librosa.beat.beat_track')
    @patch('librosa.util.sync')
    def test_chord_transcription_logic(self, mock_sync, mock_beat, mock_cqt):
        # Mock audio processing
        audio = np.zeros(44100)
        ct = ChordTranscriber()
        
        # Mocking necessary returns
        mock_cqt.return_value = np.zeros((144, 100))
        mock_beat.return_value = (120, np.array([0, 10, 20]))
        # sync returns (n_channels, n_subs), so (24, 8) here
        mock_sync.return_value = np.zeros((24, 8)) 
        
        # This will test the HMM/Viterbi part
        results = ct.transcribe(audio)
        self.assertIsInstance(results, list)

    @patch('librosa.load')
    @patch('core.nnls_chord_transcriber.ChordTranscriber.transcribe')
    @patch('librosa.beat.beat_track')
    def test_nnls_chord_transcribe_wrapper(self, mock_beat, mock_transcribe, mock_load):
        mock_load.return_value = (np.zeros(44100), 44100)
        mock_transcribe.return_value = [{'label': 'C', 'start': 0.0, 'end': 1.0}]
        mock_beat.return_value = (120, np.array([0, 22050, 44100]))
        
        result = nnls_chord_transcribe("dummy.wav")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['chord'], 'C')

    # --- Whisper Lyrics Transcriber Tests ---
    @patch('core.whisper_lyrics_transcriber.is_available')
    @patch('os.path.exists')
    @patch('subprocess.run')
    def test_whisper_lyrics_transcribe(self, mock_run, mock_exists, mock_available):
        mock_available.return_value = (True, "/tmp/models")
        mock_exists.return_value = True
        mock_run.return_value = MagicMock(stdout=json.dumps({"transcription": [{"text": "Hello", "start": 0.0, "end": 1.0}]}), check=True)
        
        result = whisper_lyrics_transcribe("vocals.wav", "/tmp/media")
        self.assertIsNotNone(result)
        self.assertEqual(result[0]['text'], "Hello")

    # --- Services Tests ---
    
    @patch('core.services.demucs_source_separate')
    def test_service_separate_sources(self, mock_demucs):
        mock_demucs.return_value = {"vocals": "v.wav", "no_vocals": "nv.wav"}
        from core.services import separate_sources
        result = separate_sources("input.mp3", "/tmp/media")
        self.assertEqual(result['vocals'], "v.wav")

    @patch('core.services.whisper_lyrics_transcribe')
    def test_service_transcribe_lyrics(self, mock_whisper):
        mock_whisper.return_value = [{"text": "Hello"}]
        from core.services import transcribe_lyrics
        result = transcribe_lyrics("v.wav", "/tmp/media", language="en")
        mock_whisper.assert_called_with("v.wav", "/tmp/media", model_name="base", language="en")
        self.assertEqual(result[0]['text'], "Hello")



if __name__ == '__main__':
    unittest.main()
