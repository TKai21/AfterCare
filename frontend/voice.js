// Voice clips only: recording, preview and authenticated playback. No dictation.
import {getToken, clearSession, escapeHtml} from '/static/api.js';

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

export function recordingFilename(type) {
  const mime = type.split(';', 1)[0].toLowerCase();
  return 'recording.' + ({'audio/webm': 'webm', 'audio/ogg': 'ogg', 'audio/mp4': 'm4a', 'audio/wav': 'wav'}[mime] || 'audio');
}

export function attachVoiceComposer(element, {label = 'Record a voice message', onSend, onBusy = () => {}, context = () => ''}) {
  let mode = 'idle', error = '', disposed = false, generation = 0;
  let stream = null, recorder = null, parts = [], blob = null, url = null;
  let ticker = null, deadline = null, started = 0, target = null;
  const supported = !!(window.isSecureContext && navigator.mediaDevices?.getUserMedia && window.MediaRecorder);

  function stopPreview() {
    const preview = element.querySelector('audio');
    if (preview) { preview.pause(); preview.removeAttribute('src'); preview.load(); }
  }

  function release() {
    clearInterval(ticker);
    clearTimeout(deadline);
    ticker = deadline = null;
    if (stream) stream.getTracks().forEach(track => track.stop());
    stream = null;
  }

  function reset(message = '') {
    generation++;
    stopPreview();
    if (recorder && recorder.state !== 'inactive') recorder.stop();
    release();
    recorder = null;
    parts = [];
    blob = null;
    if (url) URL.revokeObjectURL(url);
    url = null;
    mode = 'idle';
    error = message;
    onBusy(false);
    if (!disposed) render();
  }

  function stop() {
    if (recorder && recorder.state === 'recording') recorder.stop();
    release();
  }

  async function start() {
    if (disposed || mode !== 'idle') return;
    mode = 'requesting'; error = ''; onBusy(true); render();
    const current = ++generation;
    target = context();
    try {
      const acquired = await navigator.mediaDevices.getUserMedia({audio: true});
      if (disposed || current !== generation) {
        acquired.getTracks().forEach(track => track.stop());
        return;
      }
      stream = acquired;
      const types = ['audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/mp4', 'audio/webm', 'audio/ogg'];
      const mimeType = types.find(type => MediaRecorder.isTypeSupported(type));
      recorder = mimeType ? new MediaRecorder(stream, {mimeType}) : new MediaRecorder(stream);
      const recording = recorder;
      parts = [];
      recording.ondataavailable = event => {
        if (current !== generation || disposed || !event.data.size) return;
        parts.push(event.data);
        if (parts.reduce((sum, p) => sum + p.size, 0) > MAX_BYTES) reset('The recording is too large. Try a shorter clip.');
      };
      recording.onerror = () => {
        if (!disposed && current === generation) reset('Recording failed. Please try again.');
      };
      recording.onstop = () => {
        if (disposed || current !== generation) return;
        release();
        blob = new Blob(parts, {type: recording.mimeType || parts[0]?.type || 'audio/webm'});
        if (!blob.size) { reset('Nothing was recorded. Please try again.'); return; }
        url = URL.createObjectURL(blob);
        mode = 'ready';
        render();
      };
      recorder.start(250);
      started = performance.now();
      mode = 'recording'; render();
      ticker = setInterval(() => {
        const clock = element.querySelector('[data-voice-timer]');
        const label = Math.min(MAX_SECONDS, Math.floor((performance.now() - started) / 1000)) + ' / 60 seconds';
        if (clock && clock.textContent !== label) clock.textContent = label;
      }, 250);
      deadline = setTimeout(stop, MAX_SECONDS * 1000);
    } catch (failure) {
      if (disposed || current !== generation) return;
      reset(failure.name === 'NotAllowedError' ? 'Microphone permission was denied. Allow it in your browser and try again.' : 'The microphone could not be opened. Check your device and try again.');
    }
  }

  async function send() {
    if (disposed || mode !== 'ready' || !blob) return;
    if (target !== context()) { reset('The selected recipient changed. Record the message again.'); return; }
    mode = 'uploading'; error = ''; render();
    try {
      await onSend(blob, recordingFilename(blob.type), target);
      if (!disposed) reset();
    } catch (failure) {
      if (disposed) return;
      mode = 'ready'; error = failure.message || 'Could not send recording. Please try again.'; render();
    }
  }

  function render() {
    if (disposed) return;
    stopPreview();
    if (!supported) {
      element.innerHTML = '<p class="field-hint">Voice recording is unavailable in this browser or connection. Try another supported browser.</p>';
      return;
    }
    element.innerHTML = `<div class="voice-widget">
      <p class="voice-label">${esc(label)}</p>
      ${mode === 'recording' ? '<p class="voice-recording" role="status">Recording <span data-voice-timer>0 / 60 seconds</span></p>' : ''}
      ${mode === 'requesting' ? '<p role="status">Waiting for microphone permission...</p>' : ''}
      ${url ? `<audio controls preload="metadata" src="${esc(url)}" aria-label="Preview your recording"></audio>` : ''}
      <div class="voice-actions">
        ${mode === 'idle' ? '<button type="button" class="btn btn-secondary btn-small" data-voice-action="record">Record voice clip</button>' : ''}
        ${mode === 'recording' ? '<button type="button" class="btn btn-secondary btn-small" data-voice-action="stop">Stop recording</button>' : ''}
        ${mode === 'ready' || mode === 'uploading' ? `<button type="button" class="btn btn-primary btn-small" data-voice-action="send" ${mode === 'uploading' ? 'disabled' : ''}>${mode === 'uploading' ? 'Sending...' : 'Send voice clip'}</button>` : ''}
        ${mode !== 'idle' ? `<button type="button" class="btn btn-secondary btn-small" data-voice-action="cancel" ${mode === 'uploading' ? 'disabled' : ''}>Discard</button>` : ''}
      </div>
      ${error ? `<p class="form-error" role="alert">${esc(error)}</p>` : ''}
      <p class="field-hint">Up to 60 seconds. Listen before sending. This recording is not transcribed or checked by AI. This inbox is not for urgent help.</p>
    </div>`;
    element.querySelector('[data-voice-action="record"]')?.addEventListener('click', start);
    element.querySelector('[data-voice-action="stop"]')?.addEventListener('click', stop);
    element.querySelector('[data-voice-action="cancel"]')?.addEventListener('click', () => reset());
    element.querySelector('[data-voice-action="send"]')?.addEventListener('click', send);
  }

  const untrack = track({element, cancelRecording() {
    if (mode === 'recording' || mode === 'requesting') reset('Recording stopped because the page was hidden.');
  }, dispose() {
    if (disposed) return;
    disposed = true;
    reset();
    element.replaceChildren();
    delete element.dataset.voiceComposerBound;
  }});
  element.dataset.voiceComposerBound = 'true';
  render();
  return untrack;
}
