# MEMO: Chord Transcription Workflow

## Notes & Process

- **Vocal Extraction** (Demucs)
  - Purpose: Get clean accompaniment for better chord detection.
  - CMD: `demucs --two-stems vocals "song.mp3" -o data/separated/`

- **Lyrics Transcription** (Whisper-CLI)
  - Purpose: High-quality timed transcripts (Chinese `zh` optimized).
  - CMD: `whisper-cli -m "path/to/model" -f "data/separated/.../vocals.wav" -l zh --vad -vm "path/to/vad" -vt 0.1 -oj -ml 3`

- **Chord Recognition** (NNLS-Chroma)
  - Logic: Standalone script based on Mauch & Dixon (2010).
  - Prerequisites: `pip install numpy librosa scipy numba` + `ffmpeg`.
  - CMD: `python transcribe_chords.py "data/.../no_vocals.wav" [output.json]`
