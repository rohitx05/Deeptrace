// ── DeepTrace Frontend Logic ──────────────────────────────────────────────

let currentFile = null;

// ── Media tab selection ────────────────────────────────────────────────────
function selectMedia(type) {
  if (type !== 'photo') return;
  document.querySelectorAll('.media-card').forEach(c => {
    c.classList.remove('active');
    c.querySelector('.active-dot')?.remove();
  });
  const card = document.getElementById('tab-photo');
  card.classList.add('active');
  if (!card.querySelector('.active-dot')) {
    const dot = document.createElement('div');
    dot.className = 'active-dot';
    card.appendChild(dot);
  }
}

function showComingSoon(type) {
  const icons = { Video: '🎬', Audio: '🎙️' };
  document.getElementById('modal-icon').textContent = icons[type] || '🔒';
  document.getElementById('modal-title').textContent = `${type} Analysis`;
  document.getElementById('modal-overlay').style.display = 'flex';
}

function closeModal() {
  document.getElementById('modal-overlay').style.display = 'none';
}

// ── File handling ──────────────────────────────────────────────────────────
function handleDrop(e) {
  e.preventDefault();
  document.getElementById('upload-zone').classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
}

function handleFile(file) {
  if (!file) return;
  const allowed = ['image/jpeg', 'image/png', 'image/bmp', 'image/webp'];
  if (!allowed.includes(file.type)) {
    alert('Please upload a JPG, PNG, BMP, or WEBP image.');
    return;
  }
  currentFile = file;

  const zone = document.getElementById('upload-zone');
  zone.classList.add('has-file');

  // Show preview in upload zone
  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById('upload-label').innerHTML =
      `<img src="${e.target.result}" style="max-height:160px;border-radius:8px;margin-bottom:0.5rem;"><br>` +
      `<strong>${file.name}</strong> · ${(file.size / 1024).toFixed(1)} KB`;
  };
  reader.readAsDataURL(file);

  document.getElementById('analyze-btn').disabled = false;
}

// ── Analysis ───────────────────────────────────────────────────────────────
async function analyze() {
  if (!currentFile) return;

  // Show loading
  document.getElementById('analyzing-overlay').style.display = 'flex';
  document.getElementById('progress-fill').style.animation = 'none';
  void document.getElementById('progress-fill').offsetWidth; // reflow
  document.getElementById('progress-fill').style.animation = 'progress-anim 3s ease-in-out forwards';

  const formData = new FormData();
  formData.append('file', currentFile);

  // 120-second timeout — GPU inference can take a while
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 120000);

  try {
    const res = await fetch('/analyze', {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    const data = await res.json();

    document.getElementById('analyzing-overlay').style.display = 'none';

    if (data.error) {
      alert('Error: ' + data.error);
      return;
    }

    showResults(data);
  } catch (err) {
    clearTimeout(timeoutId);
    document.getElementById('analyzing-overlay').style.display = 'none';
    if (err.name === 'AbortError') {
      alert('Analysis timed out (>120s). The model may still be loading — try again in a moment.');
    } else {
      alert('Connection error: ' + err.message + '\n\nMake sure the server is running at http://localhost:5000');
    }
  }
}

// ── Show results ───────────────────────────────────────────────────────────
function showResults(data) {
  document.getElementById('upload-section').style.display = 'none';
  const rs = document.getElementById('results-section');
  rs.style.display = 'block';
  rs.scrollIntoView({ behavior: 'smooth' });

  const isFake = data.prediction === 'FAKE';
  const fakeProb = (data.fake_probability * 100).toFixed(1);
  const conf = (data.confidence * 100).toFixed(1);

  // Images
  if (data.original_b64) {
    document.getElementById('orig-img').src = 'data:image/png;base64,' + data.original_b64;
  }
  if (data.heatmap_b64) {
    document.getElementById('heatmap-img').src = 'data:image/png;base64,' + data.heatmap_b64;
  } else {
    document.getElementById('heatmap-img').parentElement.style.display = 'none';
  }

  // Verdict
  const vcard = document.getElementById('verdict-card');
  const vbadge = document.getElementById('verdict-badge');
  vcard.classList.toggle('fake', isFake);
  vcard.classList.toggle('real', !isFake);
  vbadge.textContent = isFake ? '⚠ DEEPFAKE DETECTED' : '✓ AUTHENTIC';
  vbadge.className = 'verdict-badge ' + (isFake ? 'fake' : 'real');
  document.getElementById('verdict-prob').textContent = fakeProb + '%';

  // Gauge
  const arc = 188.5;
  const fill = arc * (1 - data.confidence);
  document.getElementById('gauge-arc').style.strokeDashoffset = fill;
  document.getElementById('gauge-value').textContent = conf + '%';

  // Stats
  document.getElementById('stat-fake-prob').textContent = fakeProb + '%';
  document.getElementById('stat-confidence').textContent = conf + '%';
  document.getElementById('stat-manip').textContent = data.manipulation_type || 'N/A';
  document.getElementById('stat-threshold').textContent = (data.threshold ?? 0.1341).toFixed(4);

  // Forensic
  document.getElementById('forensic-text').textContent = data.forensic_explanation || '—';

  // Meta
  document.getElementById('meta-filename').textContent = data.filename || '—';
  document.getElementById('meta-threshold').textContent = (data.threshold ?? 0.1341).toFixed(4);
}

// ── Reset ──────────────────────────────────────────────────────────────────
function resetApp() {
  currentFile = null;
  document.getElementById('results-section').style.display = 'none';
  document.getElementById('upload-section').style.display = 'block';

  // Reset upload zone
  const zone = document.getElementById('upload-zone');
  zone.classList.remove('has-file');
  document.getElementById('upload-label').innerHTML = 'Drag &amp; drop image here';
  document.getElementById('file-input').value = '';
  document.getElementById('analyze-btn').disabled = true;

  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ── Copy report ────────────────────────────────────────────────────────────
function copyReport() {
  const lines = [
    '=== DeepTrace Forensic Report ===',
    `Verdict: ${document.getElementById('verdict-badge').textContent}`,
    `Fake Probability: ${document.getElementById('stat-fake-prob').textContent}`,
    `Model Confidence: ${document.getElementById('stat-confidence').textContent}`,
    `Manipulation Type: ${document.getElementById('stat-manip').textContent}`,
    `Decision Threshold: ${document.getElementById('stat-threshold').textContent}`,
    '',
    'Forensic Justification:',
    document.getElementById('forensic-text').textContent,
  ];
  navigator.clipboard.writeText(lines.join('\n'))
    .then(() => alert('Report copied to clipboard!'));
}

// ── Keyboard shortcuts ─────────────────────────────────────────────────────
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeModal();
});
