import shutil
import subprocess
import json
import os
from pathlib import Path

def is_available():
    # 1. Check for binary
    whisper_cli_path = shutil.which("whisper-cli")
    if whisper_cli_path is None:
        return False, "whisper-cli not found in PATH"
    
    # 2. Resolve symlink to find actual installation directory
    try:
        # Follow symlink to actual binary location
        real_path = Path(whisper_cli_path).resolve()
        
        # Navigate up to find whisper.cpp directory
        # Typical structure: whisper.cpp/whisper-cli (or whisper.cpp/build/bin/whisper-cli)
        current = real_path.parent
        whisper_cpp_dir = None
        
        # Search up the directory tree for whisper.cpp
        for _ in range(5):  # Limit search depth
            if current.name == "whisper.cpp" or (current / "models").exists():
                whisper_cpp_dir = current
                break
            current = current.parent
        
        if whisper_cpp_dir is None:
            return False, "Could not locate whisper.cpp directory from whisper-cli binary"
        
        # Look for models directory
        model_dir = whisper_cpp_dir / "models"
        if not model_dir.exists():
            # Try alternative: model (singular)
            model_dir = whisper_cpp_dir / "model"
            if not model_dir.exists():
                return False, f"Models directory not found in {whisper_cpp_dir}"
        
        return True, str(model_dir)
        
    except Exception as e:
        return False, f"Error resolving whisper-cli path: {e}"

def whisper_lyrics_transcribe(audio_path, media_root, model_name="base", language="zh"):
    """
    Uses Whisper-CLI for timed transcripts.
    CMD: whisper-cli -m "model" -f "vocals.wav" -l zh --vad -vt 0.1 -oj
    """
    status, model_info = is_available()
    if not status:
        return None
    
    # Use the found model directory
    model_path = os.path.join(model_info, f"ggml-{model_name}.bin")
    if not os.path.exists(model_path):
        # Fallback to model_name if it looks like a full path
        if os.path.exists(model_name):
            model_path = model_name
        else:
            print(f"Whisper Model Error: {model_path} not found")
            return None

    cmd = [
        "whisper-cli",
        "-m", model_path,
        "-f", str(audio_path),
        "-l", language,
        "--vad",
        "-vt", "0.1",
        "-oj"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        # whisper-cli -oj usually returns result in 'transcription' or directly
        # depending on version. Let's assume it has 'transcription' or 'segments'
        if isinstance(data, list):
            return data
        return data.get('transcription', data.get('segments', data))
    except Exception as e:
        print(f"Whisper Error: {e}")
        return None
