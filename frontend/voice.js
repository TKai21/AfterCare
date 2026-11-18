// Playback of voice clips sent before clips were removed (authenticated, on demand),
// and dictation: the microphone button that turns speech into text in a text box.
import {api, getToken, clearSession, escapeHtml} from '/static/api.js';

export const MAX_SECONDS = 60;
const MAX_BYTES = 5 * 1024 * 1024;
const components = new Set();
const esc = escapeHtml;

const sheet = document.createElement('link');
sheet.rel = 'stylesheet';
sheet.href = '/static/voice.css';
document.head.append(sheet);

function track(component) {
  components.add(component);
  return () => { components.delete(component); component.dispose(); };
}

function cleanHidden() {
  for (const component of [...components]) {
    if (!component.element.isConnected || component.element.closest('[hidden]')) {
      components.delete(component);
      component.dispose();
    }
  }
}

export function clearVoice(root) {
  for (const component of [...components]) {
    if (root === component.element || root.contains(component.element)) {
      components.delete(component);
      component.dispose();
    }
  }
}

new MutationObserver(cleanHidden).observe(document.documentElement, {childList: true, subtree: true, attributes: true, attributeFilter: ['hidden']});
window.addEventListener('pagehide', () => clearVoice(document.documentElement));
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    for (const component of components) component.cancelRecording?.();
  }
});

export function voicePlayerHtml(clip, label = 'Voice message') {
  if (!clip || !/^\/api\/voice\/clips\/[1-9]\d*$/.test(clip.url || '')) return '';
  const seconds = Math.max(1, Math.round((Number(clip.duration_ms) || 0) / 1000));
  return `<div class="voice-player" data-voice-url="${esc(clip.url)}">
    <button type="button" class="btn btn-secondary btn-small voice-load">Load ${esc(label)} (${seconds}s)</button>
    <audio controls preload="none" hidden aria-label="${esc(label)}"></audio>
    <span class="voice-status" role="status"></span>
  </div>`;
}

export function mountVoicePlayers(root) {
  root.querySelectorAll('[data-voice-url]').forEach(element => {
    if (element.dataset.voiceBound) return;
    element.dataset.voiceBound = 'true';
    let url = null, disposed = false;
    const controller = new AbortController();
    const button = element.querySelector('.voice-load');
    const audio = element.querySelector('audio');
    const status = element.querySelector('.voice-status');
    track({element, dispose() {
      if (disposed) return;
      disposed = true;
      controller.abort();
      audio.pause();
      audio.removeAttribute('src');
      audio.load();
      if (url) URL.revokeObjectURL(url);
      delete element.dataset.voiceBound;
    }});
    button.addEventListener('click', async () => {
      if (disposed) return;
      button.disabled = true;
      status.textContent = 'Loading recording...';
      try {
        const response = await fetch(element.dataset.voiceUrl, {
          headers: {Authorization: 'Bearer ' + getToken()}, cache: 'no-store', signal: controller.signal,
        });
        if (response.status === 401) { clearSession(); window.location.href = '/'; return; }
        if (!response.ok) throw new Error('The recording could not be loaded. Try again.');
        const blob = await response.blob();
        if (disposed) return;
        url = URL.createObjectURL(blob);
        audio.src = url;
        audio.hidden = false;
        button.hidden = true;
        status.textContent = '';
      } catch (error) {
        if (disposed || error.name === 'AbortError') return;
        status.textContent = error.message;
        button.disabled = false;
      }
    });
  });
}

const RECORDER_TYPES = ['audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/mp4', 'audio/webm', 'audio/ogg'];
const recorderSupported = () => !!(window.isSecureContext && navigator.mediaDevices?.getUserMedia && window.MediaRecorder);

function openRecorder(stream) {
  const mimeType = RECORDER_TYPES.find(type => MediaRecorder.isTypeSupported(type));
  return mimeType ? new MediaRecorder(stream, {mimeType}) : new MediaRecorder(stream);
}

function microphoneError(failure) {
  return failure.name === 'NotAllowedError'
    ? 'Microphone permission was denied. Allow it in your browser and try again.'
    : 'The microphone could not be opened. Check your device and try again.';
}

export function recordingFilename(type) {
  const mime = type.split(';', 1)[0].toLowerCase();
  return 'recording.' + ({'audio/webm': 'webm', 'audio/ogg': 'ogg', 'audio/mp4': 'm4a', 'audio/wav': 'wav'}[mime] || 'audio');
}

const MIC_ICON = '<svg viewBox="0 0 20 20" width="18" height="18" aria-hidden="true"><rect x="7" y="2" width="6" height="10" rx="3" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M4.5 9.5a5.5 5.5 0 0011 0M10 15v3" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>';
const STOP_ICON = '<svg viewBox="0 0 20 20" width="18" height="18" aria-hidden="true"><rect x="5" y="5" width="10" height="10" rx="2" fill="currentColor"/></svg>';
const clock = seconds => Math.floor(seconds / 60) + ':' + String(seconds % 60).padStart(2, '0');

// Dictation: record, send the audio to /api/voice/transcribe, append the text to
// the textarea. Nothing is sent to anyone: the person reads it and presses Send.
// `button` is an empty <button type="button">; this owns its content and state.
// onBusy(true) runs from the first tap until the text is in (or it failed).
export function attachDictation(button, textarea, {onBusy = () => {}} = {}) {
  if (!recorderSupported()) { button.hidden = true; return () => {}; }
  let mode = 'idle', disposed = false, generation = 0;
  let stream = null, recorder = null, parts = [], ticker = null, deadline = null, started = 0;
  let wasDisabled = false, errorEl = null;

  function setError(message) {
    if (!message) { errorEl?.remove(); errorEl = null; return; }
    if (!errorEl) {
      errorEl = document.createElement('p');
      errorEl.className = 'form-error dictate-error';
      errorEl.setAttribute('role', 'alert');
      textarea.insertAdjacentElement('afterend', errorEl);
    }
    errorEl.textContent = message;
  }

  function release() {
    clearInterval(ticker);
    clearTimeout(deadline);
    ticker = deadline = null;
    if (stream) stream.getTracks().forEach(track => track.stop());
    stream = null;
  }

  function render() {
    if (disposed) return;
    const labels = {idle: 'Dictate', requesting: 'Waiting for microphone', recording: 'Stop dictation', transcribing: 'Turning speech into text'};
    button.dataset.dictation = mode;
    button.classList.toggle('is-recording', mode === 'recording');
    button.setAttribute('aria-label', labels[mode]);
    button.title = labels[mode];
    if (mode === 'transcribing' || mode === 'requesting') button.disabled = true;
    else button.disabled = false;
    const icon = mode === 'recording' ? STOP_ICON : mode === 'idle' ? MIC_ICON : '<span class="spinner" aria-hidden="true"></span>';
    const text = mode === 'recording' ? `<span class="dictate-text" aria-hidden="true">Stop</span><span class="dictate-time" aria-hidden="true">${clock(0)}</span>`
      : mode === 'idle' ? '<span class="dictate-text">Dictate</span>' : '';
    button.innerHTML = icon + text;
  }

  function finish(message = '') {
    generation++;
    if (recorder && recorder.state !== 'inactive') recorder.stop();
    release();
    recorder = null;
    parts = [];
    const was = mode;
    mode = 'idle';
    if (was === 'transcribing') textarea.disabled = wasDisabled;
    if (!disposed) { setError(message); render(); }
    if (was !== 'idle') onBusy(false);
  }

  function append(text) {
    const base = textarea.value.replace(/\s+$/, '');
    let next = base ? base + ' ' + text : text;
    if (textarea.maxLength > 0) next = next.slice(0, textarea.maxLength);
    textarea.value = next;
    textarea.dispatchEvent(new Event('input', {bubbles: true}));
    textarea.focus();
    textarea.setSelectionRange(next.length, next.length);
  }

  async function transcribe(blob, current) {
    mode = 'transcribing';
    wasDisabled = textarea.disabled;
    textarea.disabled = true;
    render();
    try {
      const form = new FormData();
      form.append('file', blob, recordingFilename(blob.type));
      const out = await api('/api/voice/transcribe', {method: 'POST', form});
      if (disposed || current !== generation) return;
      finish();
      append(out.text);
    } catch (failure) {
      if (disposed || current !== generation) return;
      finish(failure.message || 'Dictation failed. Please try again.');
    }
  }

  async function start() {
    if (disposed || mode !== 'idle') return;
    mode = 'requesting'; setError(''); onBusy(true); render();
    const current = ++generation;
    try {
      const acquired = await navigator.mediaDevices.getUserMedia({audio: true});
      if (disposed || current !== generation) { acquired.getTracks().forEach(track => track.stop()); return; }
      stream = acquired;
      const recording = recorder = openRecorder(stream);
      parts = [];
      recording.ondataavailable = event => {
        if (current !== generation || disposed || !event.data.size) return;
        parts.push(event.data);
        if (parts.reduce((sum, p) => sum + p.size, 0) > MAX_BYTES) finish('The recording is too large. Try a shorter one.');
      };
      recording.onerror = () => { if (!disposed && current === generation) finish('Recording failed. Please try again.'); };
      recording.onstop = () => {
        if (disposed || current !== generation) return;
        release();
        const blob = new Blob(parts, {type: recording.mimeType || parts[0]?.type || 'audio/webm'});
        if (!blob.size) { finish('Nothing was recorded. Please try again.'); return; }
        transcribe(blob, current);
      };
      recording.start(250);
      started = performance.now();
      mode = 'recording'; render();
      ticker = setInterval(() => {
        const label = clock(Math.min(MAX_SECONDS, Math.floor((performance.now() - started) / 1000)));
        const time = button.querySelector('.dictate-time');
        if (time && time.textContent !== label) time.textContent = label;
      }, 250);
      deadline = setTimeout(stopRecording, MAX_SECONDS * 1000);
    } catch (failure) {
      if (disposed || current !== generation) return;
      finish(microphoneError(failure));
    }
  }

  function stopRecording() {
    if (recorder && recorder.state === 'recording') recorder.stop();
    release();
  }

  function onClick() {
    if (mode === 'idle') start();
    else if (mode === 'recording') stopRecording();
  }
  button.addEventListener('click', onClick);
  // Typing again clears an old error.
  const clearError = () => { if (mode === 'idle') setError(''); };
  textarea.addEventListener('input', clearError);
  render();

  return track({element: button, cancelRecording() {
    if (mode === 'recording' || mode === 'requesting') finish('Dictation stopped because the page was hidden.');
  }, dispose() {
    if (disposed) return;
    const busy = mode !== 'idle';
    if (mode === 'transcribing') textarea.disabled = wasDisabled;
    disposed = true;
    generation++;
    if (recorder && recorder.state !== 'inactive') recorder.stop();
    release();
    button.removeEventListener('click', onClick);
    textarea.removeEventListener('input', clearError);
    setError('');
    if (busy) onBusy(false);
  }});
}
