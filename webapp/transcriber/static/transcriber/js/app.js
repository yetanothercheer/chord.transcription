// --- Core UI State ---
let separatedPaths = { vocals: null, no_vocals: null };

function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

function selectSong(filePath, fileName) {
    // Update all file_path inputs
    document.querySelectorAll('input[name="file_path"]').forEach(i => i.value = filePath);

    // Update file name display
    const fileNameEl = document.getElementById('file-name');
    if (fileNameEl) fileNameEl.textContent = fileName;

    // Hide welcome message
    const welcomeEl = document.getElementById('welcome-message');
    if (welcomeEl) welcomeEl.classList.add('hidden');

    // Update status
    const statusEl = document.getElementById('current-task-status');
    if (statusEl) statusEl.textContent = 'Ready: ' + fileName;
}

function startPipeline() {
    const filePath = document.getElementById('stems-file-path').value;
    const fileName = document.getElementById('file-name').textContent;

    if (!filePath || filePath === "") return alert("Please upload or select a file first.");

    document.getElementById('pipeline-cta').classList.add('hidden');
    document.getElementById('pipeline-progress-box').classList.remove('hidden');

    const formData = new FormData();
    formData.append('file_path', filePath);
    formData.append('file_name', fileName);
    formData.append('chord_algorithm', document.getElementById('pipeline-chord-algo').value);
    formData.append('language', document.getElementById('pipeline-lang').value);

    fetch('/pipeline/start/', {
        method: 'POST',
        headers: {
            'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
        },
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                document.getElementById('pipeline-progress-box').classList.remove('hidden');
                pollTaskStatus(data.task_id);
            } else {
                alert('Error starting pipeline: ' + data.message);
                document.getElementById('pipeline-cta').classList.remove('hidden');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while starting the pipeline.');
            document.getElementById('pipeline-cta').classList.remove('hidden');
        });
}

function pollTaskStatus(taskId) {
    const interval = setInterval(() => {
        fetch(`/pipeline/status/${taskId}/`)
            .then(r => r.json())
            .then(data => {
                document.getElementById('pipeline-percent').textContent = data.progress + '%';
                document.getElementById('pipeline-progress-bar').value = data.progress;
                document.getElementById('pipeline-step-msg').textContent = data.current_step || 'Processing...';

                if (data.status === 'SUCCESS') {
                    clearInterval(interval);
                    loadPipelineResult(taskId);
                } else if (data.status === 'FAILURE') {
                    clearInterval(interval);
                    alert("Pipeline Failed: " + data.error_message);
                    document.getElementById('pipeline-cta').classList.remove('hidden');
                    document.getElementById('pipeline-progress-box').classList.add('hidden');
                }
            });
    }, 2000);
}

function loadPipelineResult(taskId) {
    htmx.ajax('GET', `/pipeline/result/${taskId}/`, { target: '#res-pipeline' })
        .then(() => {
            document.getElementById('res-pipeline').classList.remove('hidden');
            document.getElementById('pipeline-progress-box').classList.add('hidden');
        });
}
