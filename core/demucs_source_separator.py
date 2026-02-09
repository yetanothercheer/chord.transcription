import shutil
import subprocess
from pathlib import Path

def is_available():
    return shutil.which("demucs") is not None

def demucs_source_separate(input_path, media_root, model_name="htdemucs"):
    """
    Uses Demucs to separate vocals from the track.
    CMD: demucs -n <model> --two-stems vocals "song.mp3" -o data/separated/
    """
    media_root = Path(media_root)
    output_dir = media_root / "separated"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "demucs",
        "-n", model_name,
        "--two-stems", "vocals",
        str(input_path),
        "-o", str(output_dir)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        filename_stem = Path(input_path).stem
        # Demucs creates a folder based on model name
        vocals_path = output_dir / model_name / filename_stem / "vocals.wav"
        no_vocals_path = output_dir / model_name / filename_stem / "no_vocals.wav"
        
        return {
            "vocals": str(vocals_path) if vocals_path.exists() else None,
            "no_vocals": str(no_vocals_path) if no_vocals_path.exists() else None
        }
    except Exception as e:
        print(f"Demucs Error: {e}")
        return None
