from celery import shared_task
from .models import TranscriptionTask
from core.services import separate_sources, transcribe_lyrics, recognize_chords
from django.conf import settings
import os

@shared_task(bind=True)
def process_audio_pipeline(self, task_id, chord_algorithm='nnls', language='zh'):
    try:
        task = TranscriptionTask.objects.get(id=task_id)
        task.status = 'PROCESSING'
        task.save()

        def update_progress(percent, step):
            task.progress = percent
            task.current_step = step
            task.save()

        update_progress(10, "Separating audio sources...")
        
        # 1. Source Separation
        stems = separate_sources(task.audio_file_path, settings.MEDIA_ROOT)
        vocals_path = stems['vocals']
        accompaniment_path = stems['no_vocals']
        
        update_progress(40, f"Transcribing lyrics from vocals ({language})...")
        
        # 2. Lyrics Transcription
        lyrics_data = transcribe_lyrics(vocals_path, settings.MEDIA_ROOT, language=language)
        
        update_progress(70, "Recognizing chords from accompaniment...")
        
        # 3. Chord Recognition
        chord_results = recognize_chords(accompaniment_path, algorithm=chord_algorithm)
        
        update_progress(90, "Aligning results...")

        results = {
            "audio_url": None,
            "vocals_url": None,
            "chords": chord_results['chords'],
            "beats": chord_results['beats'],
            "tempo": chord_results['tempo'],
            "lyrics": lyrics_data,
            "vocals_path": vocals_path,
            "accompaniment_path": accompaniment_path,
        }
        
        task.result_json = results
        task.status = 'SUCCESS'
        task.progress = 100
        task.save()
        
    except Exception as e:
        try:
            task = TranscriptionTask.objects.get(id=task_id)
            task.status = 'FAILURE'
            task.error_message = str(e)
            task.save()
        except:
            pass
        print(f"Task Error: {e}")
