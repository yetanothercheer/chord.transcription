from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
import uuid
import json
from pathlib import Path
from core.basic_pitch_transcriber import basic_pitch_transcribe, is_available as notes_available
from core.demucs_source_separator import demucs_source_separate, is_available as demucs_available
from core.whisper_lyrics_transcriber import whisper_lyrics_transcribe, is_available as whisper_available
from core.nnls_chord_transcriber import nnls_chord_transcribe, is_available as nnls_available
from core.vamp_chord_transcriber import vamp_chord_transcribe, is_available as vamp_available

from .models import TranscriptionTask
from .tasks import process_audio_pipeline

def index(request):
    # Get available songs from data/songs directory
    songs_dir = Path(settings.BASE_DIR).parent / 'data' / 'songs'
    available_songs = []
    if songs_dir.exists():
        for file in songs_dir.glob('*'):
            if file.suffix.lower() in ['.mp3', '.wav', '.m4a', '.flac', '.ogg']:
                available_songs.append({
                    'name': file.stem,
                    'path': str(file),
                    'size': file.stat().st_size
                })
    return render(request, 'transcriber/index.html', {'available_songs': available_songs})

def list_songs(request):
    """API endpoint to list available songs"""
    songs_dir = Path(settings.BASE_DIR).parent / 'data' / 'songs'
    available_songs = []
    if songs_dir.exists():
        for file in songs_dir.glob('*'):
            if file.suffix.lower() in ['.mp3', '.wav', '.m4a', '.flac', '.ogg']:
                available_songs.append({
                    'name': file.stem,
                    'path': str(file)
                })
    return JsonResponse({'songs': available_songs})

def check_status(request):
    return JsonResponse({
        'stems': {'available': demucs_available(), 'msg': 'Demucs binary missing' if not demucs_available() else 'Ready'},
        'lyrics': {'available': whisper_available()[0], 'msg': whisper_available()[1]},
        'chords': {
            'available': nnls_available() or vamp_available(), 
            'msg': 'Ready',
            'nnls': nnls_available(),
            'vamp': vamp_available()
        },
        'notes': {'available': notes_available(), 'msg': 'Basic-pitch not installed' if not notes_available() else 'Ready'}
    })

def check_status_fragments(request):
    status_map = {
        'stems': {'available': demucs_available(), 'msg': 'Demucs binary missing' if not demucs_available() else 'Ready'},
        'lyrics': {'available': whisper_available()[0], 'msg': whisper_available()[1]},
        'chords': {'available': chords_available(), 'msg': 'Dependencies missing' if not chords_available() else 'Ready'},
        'notes': {'available': notes_available(), 'msg': 'Basic-pitch not installed' if not notes_available() else 'Ready'}
    }
    return render(request, 'transcriber/partials/_status_alerts.html', {'status_map': status_map})

def upload_audio(request):
    if request.method == 'POST' and request.FILES.get('audio'):
        audio_file = request.FILES['audio']
        fs = FileSystemStorage()
        ext = os.path.splitext(audio_file.name)[1]
        filename = f"{uuid.uuid4()}{ext}"
        filepath = fs.save(filename, audio_file)
        
        file_path = fs.path(filepath)
        file_url = fs.url(filepath)
        
        if request.headers.get('HX-Request'):
            return JsonResponse({
                'status': 'success',
                'file_url': file_url,
                'file_path': file_path,
                'file_name': audio_file.name
            })
            
        return JsonResponse({
            'status': 'success',
            'file_url': file_url,
            'file_path': file_path
        })
    return JsonResponse({'status': 'error', 'message': 'Invalid request'}, status=400)

def extract_stems(request):
    if not demucs_available():
        return JsonResponse({'status': 'error', 'message': 'Demucs is not installed or not in PATH.'}, status=412)
        
    file_path = request.POST.get('file_path')
    model_name = request.POST.get('model_name', 'htdemucs')
    if not file_path:
        return JsonResponse({'status': 'error', 'message': 'No file path provided'}, status=400)
    
    stems = demucs_source_separate(file_path, settings.MEDIA_ROOT, model_name=model_name)
    if stems:
        filename_stem = Path(file_path).stem
        context = {
            'vocals_url': f"{settings.MEDIA_URL}separated/{model_name}/{filename_stem}/vocals.wav",
            'no_vocals_url': f"{settings.MEDIA_URL}separated/{model_name}/{filename_stem}/no_vocals.wav",
            'vocals_path': stems['vocals'],
            'no_vocals_path': stems['no_vocals']
        }
        if request.headers.get('HX-Request'):
            return render(request, 'transcriber/partials/_stems_result.html', context)
        return JsonResponse({'status': 'success', **context})

    return JsonResponse({'status': 'error', 'message': 'Separation failed'}, status=500)

def transcribe_lyrics(request):
    available, msg = whisper_available()
    if not available:
        return JsonResponse({'status': 'error', 'message': f'Whisper Error: {msg}'}, status=412)
        
    file_path = request.POST.get('file_path')
    model_name = request.POST.get('model_name', 'base')
    language = request.POST.get('language', 'zh')
    if not file_path:
        return JsonResponse({'status': 'error', 'message': 'No file path provided'}, status=400)
    
    lyrics = whisper_lyrics_transcribe(file_path, settings.MEDIA_ROOT, model_name=model_name, language=language)
    if lyrics:
        if request.headers.get('HX-Request'):
            return render(request, 'transcriber/partials/_lyrics_result.html', {'lyrics': lyrics})
        return JsonResponse({'status': 'success', 'lyrics': lyrics})
    return JsonResponse({'status': 'error', 'message': 'Transcription failed'}, status=500)

def recognize_chords(request):
    file_path = request.POST.get('file_path')
    algorithm = request.POST.get('algorithm', 'nnls') # 'nnls', 'madmom', 'vamp'
    
    if not file_path:
        return JsonResponse({'status': 'error', 'message': 'No file path provided'}, status=400)
    
    chords = None
    if algorithm == 'vamp':
        if not vamp_available():
            return JsonResponse({'status': 'error', 'message': 'Vamp/Chordino is not installed.'}, status=412)
        chords = vamp_chord_transcribe(file_path)
    else: # Default/NNLS
        if not nnls_available():
            return JsonResponse({'status': 'error', 'message': 'NNLS dependencies missing.'}, status=412)
        chords = nnls_chord_transcribe(file_path)
    
    if chords is not None:
        if request.headers.get('HX-Request'):
            return render(request, 'transcriber/partials/_chords_result.html', {'chords': chords, 'algorithm': algorithm})
        return JsonResponse({'status': 'success', 'chords': chords, 'algorithm': algorithm})
    
    return JsonResponse({'status': 'error', 'message': f'Chord recognition ({algorithm}) failed'}, status=500)

def transcribe_notes(request):
    if not notes_available():
        return JsonResponse({'status': 'error', 'message': 'Note Transcribe package is not installed.'}, status=412)
        
    file_path = request.POST.get('file_path')
    if not file_path:
        return JsonResponse({'status': 'error', 'message': 'No file path provided'}, status=400)
    
    output_filename = f"notes_{uuid.uuid4()}.wav"
    output_path = os.path.join(settings.MEDIA_ROOT, output_filename)
    result_path = basic_pitch_transcribe(file_path, output_path)
    if result_path:
        context = {'notes_url': f"{settings.MEDIA_URL}{output_filename}"}
        if request.headers.get('HX-Request'):
            return render(request, 'transcriber/partials/_notes_result.html', context)
        return JsonResponse({'status': 'success', **context})
    return JsonResponse({'status': 'error', 'message': 'Note Transcription failed'}, status=500)

def start_pipeline(request):
    file_path = request.POST.get('file_path')
    file_name = request.POST.get('file_name', 'Unknown')
    if not file_path:
        return JsonResponse({'status': 'error', 'message': 'No file path provided'}, status=400)
    
    chord_algorithm = request.POST.get('chord_algorithm', 'madmom')
    language = request.POST.get('language', 'zh')
    
    task = TranscriptionTask.objects.create(
        original_filename=file_name,
        audio_file_path=file_path
    )
    process_audio_pipeline.delay(str(task.id), chord_algorithm=chord_algorithm, language=language)
    
    return JsonResponse({
        'status': 'success',
        'task_id': str(task.id)
    })

def pipeline_status(request, task_id):
    try:
        task = TranscriptionTask.objects.get(id=task_id)
        return JsonResponse({
            'status': task.status,
            'progress': task.progress,
            'current_step': task.current_step,
            'error_message': task.error_message
        })
    except TranscriptionTask.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Task not found'}, status=404)

def pipeline_result(request, task_id):
    try:
        task = TranscriptionTask.objects.get(id=task_id)
        if task.status != 'SUCCESS':
            return JsonResponse({'status': 'error', 'message': 'Task not finished'}, status=400)
        
        return render(request, 'transcriber/partials/_pipeline_result.html', {'task': task, 'result': task.result_json})
    except TranscriptionTask.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Task not found'}, status=404)






