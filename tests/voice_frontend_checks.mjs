// Exercise the actual voice module with browser-device/DOM test doubles.
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
let created = 0, revoked = 0;
URL.createObjectURL = () => 'blob:test-' + (++created);
URL.revokeObjectURL = () => { revoked++; };
const path = new URL('../frontend/voice.js', import.meta.url);
const source = readFileSync(path, 'utf8').replace(
  "import {api, getToken, clearSession, escapeHtml} from '/static/api.js';",
  "const api = async () => ({text: ''}); const getToken = () => 'test-token'; const clearSession = () => {}; const escapeHtml = x => String(x);",
);
const voice = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));
const flush = () => new Promise(resolve => setImmediate(resolve));
const action = (element, name) => element.querySelector(`[data-voice-action="${name}"]`);
let passed = 0;
async function test(name, fn) {
  await fn(); passed++; console.log('PASS:', name);
}

await test('recording previews before sending, and stops microphone tracks', async () => {
  const element = new Element(); let sent = 0;
  const dispose = voice.attachVoiceComposer(element, {onSend: async blob => { assert.ok(blob.size); sent++; }});
  await action(element, 'record').click();
  await action(element, 'stop').click(); await flush();
  assert.ok(element.innerHTML.includes('Preview your recording'));
  assert.equal(sent, 0);
  assert.ok(lastStream.track.stops > 0);
  await action(element, 'send').click();
  assert.equal(sent, 1);
  dispose();
});

await test('cancelling a pending permission request closes a late-arriving stream', async () => {
  const previous = navigator.mediaDevices.getUserMedia;
  let accept;
  navigator.mediaDevices.getUserMedia = () => new Promise(resolve => { accept = resolve; });
  const element = new Element();
  const dispose = voice.attachVoiceComposer(element, {onSend: async () => {}});
  const opening = action(element, 'record').click();
  action(element, 'cancel').click();
  const stream = makeStream(); accept(stream); await opening;
  assert.equal(stream.track.stops, 1);
  assert.ok(action(element, 'record'));
  navigator.mediaDevices.getUserMedia = previous; dispose();
});

await test('microphone denial restores usable controls', async () => {
  const previous = navigator.mediaDevices.getUserMedia;
  navigator.mediaDevices.getUserMedia = async () => { const e = new Error('denied'); e.name = 'NotAllowedError'; throw e; };
  const element = new Element();
  const dispose = voice.attachVoiceComposer(element, {onSend: async () => {}});
  await action(element, 'record').click();
  assert.ok(element.innerHTML.includes('permission was denied'));
  assert.ok(action(element, 'record'));
  navigator.mediaDevices.getUserMedia = previous; dispose();
});

await test('discarding a preview revokes its temporary audio URL', async () => {
  const element = new Element(); const before = revoked;
  const dispose = voice.attachVoiceComposer(element, {onSend: async () => {}});
  await action(element, 'record').click(); action(element, 'stop').click(); await flush();
  action(element, 'cancel').click();
  assert.equal(revoked, before + 1); dispose();
});

await test('removing a composer stops recording and releases its microphone', async () => {
  const element = new Element();
  voice.attachVoiceComposer(element, {onSend: async () => {}});
  await action(element, 'record').click(); const stream = lastStream;
  element.isConnected = false; sweep(); await flush();
  assert.ok(stream.track.stops > 0);
});

await test('hiding the page stops recording without permanently removing controls', async () => {
  const element = new Element();
  const dispose = voice.attachVoiceComposer(element, {onSend: async () => {}});
  await action(element, 'record').click(); const stream = lastStream;
  document.hidden = true; documentEvents.visibilitychange(); await flush(); document.hidden = false;
  assert.ok(stream.track.stops > 0);
  assert.ok(action(element, 'record')); dispose();
});

await test('changing the selected recipient cannot send a stale recording', async () => {
  const element = new Element(); let selected = 1, sent = 0;
  const dispose = voice.attachVoiceComposer(element, {context: () => selected, onSend: async () => { sent++; }});
  await action(element, 'record').click(); action(element, 'stop').click(); await flush();
  selected = 2; await action(element, 'send').click();
  assert.equal(sent, 0);
  assert.ok(element.innerHTML.includes('recipient changed')); dispose();
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
