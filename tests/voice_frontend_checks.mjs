// Exercise the actual voice module (playback and dictation) with browser-device/DOM test doubles.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';

class Element {
  constructor() { this.dataset = {}; this.isConnected = true; this.hidden = false; this.html = ''; this.nodes = new Map(); this.events = {}; }
  set innerHTML(value) { this.html = value; this.nodes.clear(); }
  get innerHTML() { return this.html; }
  querySelector(selector) {
    if (!this.html.includes(selector.startsWith('[') ? selector.slice(1, -1) : selector === 'audio' ? '<audio' : selector.slice(1))) return null;
    if (!this.nodes.has(selector)) this.nodes.set(selector, new Element());
    return this.nodes.get(selector);
  }
  querySelectorAll(selector) { return this.children || []; }
  addEventListener(name, fn) { this.events[name] = fn; }
  click() { return this.events.click?.(); }
  append() {}
  replaceChildren() { this.innerHTML = ''; }
  closest() { return this.hidden ? this : null; }
  contains(other) { return this === other || (this.children || []).includes(other); }
  pause() { this.paused = true; }
  load() {}
  removeAttribute(key) { delete this[key]; }
  removeEventListener(name, fn) { if (this.events[name] === fn) delete this.events[name]; }
  setAttribute(key, value) { this.attrs = {...this.attrs, [key]: value}; }
  get classList() { return {toggle: () => {}}; }
  remove() { this.removed = true; }
}
class Textarea extends Element {
  constructor(value = '') { super(); this.value = value; this.maxLength = -1; this.disabled = false; this.inserted = []; this.inputs = 0; }
  dispatchEvent() { this.inputs++; }
  focus() { this.focused = true; }
  setSelectionRange() {}
  insertAdjacentElement(where, el) { this.inserted.push(el); }
}

const windowEvents = {}, documentEvents = {};
const documentRoot = new Element();
globalThis.document = {
  head: new Element(), documentElement: documentRoot, hidden: false,
  createElement: () => new Element(),
  addEventListener: (name, fn) => { documentEvents[name] = fn; },
};
globalThis.window = {isSecureContext: true, location: {}, addEventListener: (name, fn) => { windowEvents[name] = fn; }};
let sweep;
globalThis.MutationObserver = class { constructor(fn) { sweep = fn; } observe() {} };
let lastStream;
const makeStream = () => {
  const track = {stops: 0, stop() { this.stops++; }};
  return {track, getTracks: () => [track]};
};
Object.defineProperty(globalThis, 'navigator', {configurable: true, value: {
  mediaDevices: {getUserMedia: async () => (lastStream = makeStream())},
}});
class Recorder {
  static isTypeSupported() { return true; }
  constructor(stream, options = {}) { this.stream = stream; this.mimeType = options.mimeType || 'audio/webm'; this.state = 'inactive'; }
  start() { this.state = 'recording'; }
  stop() {
    this.state = 'inactive';
    queueMicrotask(() => {
      this.ondataavailable?.({data: new Blob(['example recording'], {type: this.mimeType})});
      this.onstop?.();
    });
  }
}
globalThis.MediaRecorder = window.MediaRecorder = Recorder;
let timer = 0;
globalThis.setInterval = () => ++timer;
globalThis.setTimeout = () => ++timer;
globalThis.clearInterval = () => {};
globalThis.clearTimeout = () => {};
let created = 0, revoked = 0; // object URLs, used by playback
URL.createObjectURL = () => 'blob:test-' + (++created);
URL.revokeObjectURL = () => { revoked++; };
const path = new URL('../frontend/voice.js', import.meta.url);
const source = readFileSync(path, 'utf8').replace(
  "import {api, getToken, clearSession, escapeHtml} from '/static/api.js';",
  "const api = (...args) => globalThis.__api(...args); const getToken = () => 'test-token'; const clearSession = () => {}; const escapeHtml = x => String(x);",
);
const voice = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));
const flush = () => new Promise(resolve => setImmediate(resolve));
let passed = 0;
async function test(name, fn) {
  await fn(); passed++; console.log('PASS:', name);
}

const settle = async () => { for (let i = 0; i < 6; i++) await flush(); };
const dictate = async (button, text) => { await button.click(); await settle(); await button.click(); await settle(); };

await test('dictation appends the transcript to typed text, never submits, and releases the microphone', async () => {
  const button = new Element(), box = new Textarea('I have a runny nose.  ');
  let busy = [], upload;
  globalThis.__api = async (path, options) => { upload = {path, options}; return {text: 'No fever.'}; };
  const dispose = voice.attachDictation(button, box, {onBusy: b => busy.push(b)});
  await dictate(button, 'x');
  assert.equal(upload.path, '/api/voice/transcribe');
  assert.ok(upload.options.form.get('file'));
  assert.equal(box.value, 'I have a runny nose. No fever.');
  assert.ok(box.inputs > 0 && box.focused);
  assert.equal(box.disabled, false);
  assert.deepEqual(busy, [true, false]);
  assert.ok(lastStream.track.stops > 0);
  dispose();
});

await test('dictation respects the box length limit', async () => {
  const button = new Element(), box = new Textarea('abc'); box.maxLength = 8;
  globalThis.__api = async () => ({text: 'defghijk'});
  const dispose = voice.attachDictation(button, box);
  await dictate(button);
  assert.equal(box.value, 'abc defg');
  dispose();
});

await test('a failed transcription keeps the typed text and shows the error', async () => {
  const button = new Element(), box = new Textarea('keep me');
  globalThis.__api = async () => { throw new Error('I could not hear anything. Please try again.'); };
  const dispose = voice.attachDictation(button, box);
  await dictate(button);
  assert.equal(box.value, 'keep me');
  assert.equal(box.disabled, false);
  assert.equal(box.inserted.at(-1).textContent, 'I could not hear anything. Please try again.');
  dispose();
});

await test('microphone denial shows a message and leaves the box usable', async () => {
  const previous = navigator.mediaDevices.getUserMedia;
  navigator.mediaDevices.getUserMedia = async () => { const e = new Error('denied'); e.name = 'NotAllowedError'; throw e; };
  const button = new Element(), box = new Textarea('typed');
  const dispose = voice.attachDictation(button, box);
  await button.click(); await settle();
  assert.ok(box.inserted.at(-1).textContent.includes('permission was denied'));
  assert.equal(box.value, 'typed');
  navigator.mediaDevices.getUserMedia = previous; dispose();
});

await test('removing the box stops dictation and releases its microphone', async () => {
  const button = new Element(), box = new Textarea('');
  voice.attachDictation(button, box);
  await button.click(); await settle(); const stream = lastStream;
  button.isConnected = false; sweep(); await settle();
  assert.ok(stream.track.stops > 0);
});

await test('hiding the page stops dictation', async () => {
  const button = new Element(), box = new Textarea('');
  const dispose = voice.attachDictation(button, box);
  await button.click(); await settle(); const stream = lastStream;
  document.hidden = true; documentEvents.visibilitychange(); await settle(); document.hidden = false;
  assert.ok(stream.track.stops > 0);
  dispose();
});

await test('playback fetch uses the bearer header, not a token in the URL', async () => {
  const player = new Element();
  player.dataset.voiceUrl = '/api/voice/clips/5';
  player.innerHTML = '<button class="voice-load"></button><audio></audio><span class="voice-status"></span>';
  const root = new Element(); root.children = [player];
  let captured;
  globalThis.fetch = async (url, options) => { captured = {url, options}; return {ok: true, status: 200, blob: async () => new Blob(['audio'])}; };
  voice.mountVoicePlayers(root);
  await player.querySelector('.voice-load').click();
  assert.equal(captured.url, '/api/voice/clips/5');
  assert.equal(captured.options.headers.Authorization, 'Bearer test-token');
  assert.equal(captured.options.cache, 'no-store');
  voice.clearVoice(root);
});

await test('audio players reject untrusted playback URLs', async () => {
  assert.equal(voice.voicePlayerHtml({url: 'https://example.com/audio', duration_ms: 1}), '');
});

console.log(`${passed} frontend voice checks passed.`);
