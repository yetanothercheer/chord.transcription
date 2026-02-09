from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('status/', views.check_status, name='check_status'),
    path('status/fragments/', views.check_status_fragments, name='check_status_fragments'),
    path('upload/', views.upload_audio, name='upload_audio'),
    path('stems/', views.extract_stems, name='extract_stems'),
    path('lyrics/', views.transcribe_lyrics, name='transcribe_lyrics'),
    path('chords/', views.recognize_chords, name='recognize_chords'),
    path('notes/', views.transcribe_notes, name='transcribe_notes'),
    path('pipeline/start/', views.start_pipeline, name='start_pipeline'),
    path('pipeline/status/<uuid:task_id>/', views.pipeline_status, name='pipeline_status'),
    path('pipeline/result/<uuid:task_id>/', views.pipeline_result, name='pipeline_result'),
]

