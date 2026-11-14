// Care team tab: questions to bring to the care team, open first, resolved collapsed.
import {api, escapeHtml, registerTab} from '/static/api.js';
import {attachVoiceComposer, clearVoice, mountVoicePlayers, voicePlayerHtml} from '/static/voice.js';
import {
  CHECK_ICON, MAX_QUESTION, askDoctorDirectly, emergencyHtml, loadDoctors, pickDoctor,
  warningSignsOf,
} from '/static/doctors.js';

const esc = escapeHtml;

const SOURCE_LABELS = {
  agent: 'From the assistant',
  validation_fallback: 'From the assistant',
  rejected_item: 'From your instruction review',
  patient: 'Asked by you',
  patient_voice: 'Voice question from you',
};

const view = {
  questions: null,
  error: null,
  busy: null,          // id being resolved
  itemError: null,     // {id, text}
  resolvedOpen: false,
  doctors: [],
  // chosen: the patient picked a doctor in the select during this visit to the tab.
  form: {doctorId: null, chosen: false, text: '', sending: false, error: '', sent: null, emergency: false},
  signs: [],           // active warning signs, for the emergency banner
  voiceActive: false,
};

// Each visit starts on the doctor last chosen in Ask, unless the patient picks
// another one in the form during the visit.
function pickFormDoctor() {
  const form = view.form;
  if (form.chosen && view.doctors.some(d => d.id === form.doctorId)) return;
  const picked = pickDoctor(view.doctors);
  form.doctorId = picked ? picked.id : null;
}

function show(panel, ctx) {
  view.signs = warningSignsOf(ctx && ctx.document);
  view.form.chosen = false;
  pickFormDoctor();
  markAnswersSeen();
  render(panel);
  load(panel);
  loadDoctors().then(doctors => {
    view.doctors = doctors;
    pickFormDoctor();
    render(panel);
  }).catch(() => {
    // The form stays hidden; the list still works.
  });
}

registerTab('careteam', {show});

// ---------- new-answers badge support ----------

const SEEN_KEY = 'aftercare.seenAnswers';
const listeners = new Set();

function readSeen() {
  try {
    const raw = JSON.parse(localStorage.getItem(SEEN_KEY) || '[]');
    return new Set(Array.isArray(raw) ? raw : []);
  } catch (error) {
    return new Set();
  }
}

function writeSeen(ids) {
  try {
    localStorage.setItem(SEEN_KEY, JSON.stringify([...ids]));
  } catch (error) {
    // Storage unavailable: the badge simply reappears next visit.
  }
}

const answeredIds = () =>
  (view.questions || []).filter(q => q.status === 'answered').map(q => q.id);

// Count of answered questions this browser has not shown yet.
export function unseenAnswerCount() {
  const seen = readSeen();
  return answeredIds().filter(id => !seen.has(id)).length;
}

// Marks everything answered so far as seen (only call while the tab is shown).
function markAnswersSeen() {
  const ids = answeredIds();
  if (ids.length) writeSeen(new Set([...readSeen(), ...ids]));
  notify();
}

// onCareTeamChange(fn): fn runs whenever the question list or seen-set changes.
export function onCareTeamChange(fn) {
  listeners.add(fn);
}

function notify() {
  listeners.forEach(fn => fn());
}

// Fetches the list without rendering; used at boot and after Ask answers.
export async function refreshCareTeam() {
  try {
    const data = await api('/api/care-team');
    view.questions = data.questions;
    view.error = null;
  } catch (error) {
    if (!view.questions) view.error = error.message;
  }
  notify();
}

async function load(panel) {
  await refreshCareTeam();
  if (!panel.hidden) markAnswersSeen();
  render(panel);
}

async function resolve(panel, id) {
  view.busy = id;
  view.itemError = null;
  render(panel);
  try {
    const updated = await api(`/api/care-team/${id}/resolve`, {method: 'POST'});
    view.questions = view.questions.map(q => (q.id === id ? updated : q));
  } catch (error) {
    view.itemError = {id, text: error.message};
  }
  view.busy = null;
  render(panel);
  const next = panel.querySelector('.resolve-btn') || panel.querySelector('.resolved summary');
  if (next) next.focus();
}

// Server timestamps are UTC without an offset.
function formatDate(iso) {
  if (!iso) return '';
  const stamp = /[zZ]|[+-]\d\d:?\d\d$/.test(iso) ? iso : iso + 'Z';
  const d = new Date(stamp);
  if (Number.isNaN(d.getTime())) return '';
  return d.toLocaleDateString('en-US', {month: 'short', day: 'numeric', year: 'numeric'});
}

function metaHtml(q) {
  const parts = [formatDate(q.created_at), q.doctor_name ? 'For ' + q.doctor_name : '', SOURCE_LABELS[q.source]].filter(Boolean);
  return `<p class="question-meta">${parts.map(esc).join('<span aria-hidden="true"> &middot; </span>')}</p>`;
}

function openHtml(q) {
  const busy = view.busy === q.id;
  const error = view.itemError && view.itemError.id === q.id
    ? `<p class="form-error" role="alert">${esc(view.itemError.text)}</p>` : '';
  return `
    <li class="item question" data-id="${q.id}">
      ${metaHtml(q)}
      <p class="question-text">${esc(q.question)}</p>
      ${voicePlayerHtml(q.voice?.question, 'your question')}
      ${q.reason && q.source !== 'patient' ? `<p class="question-reason"><span class="question-reason-label">Why:</span> ${esc(q.reason)}</p>` : ''}
      ${error}
      <div class="question-actions">
        <button type="button" class="btn btn-secondary btn-small resolve-btn" data-id="${q.id}" ${view.busy ? 'disabled' : ''}>${busy ? 'Saving...' : 'Mark resolved'}</button>
      </div>
    </li>`;
}

function shortDate(iso) {
  return formatDate(iso).replace(/,\s*\d{4}$/, '');
}

function answeredHtml(q) {
  const busy = view.busy === q.id;
  const error = view.itemError && view.itemError.id === q.id
    ? `<p class="form-error" role="alert">${esc(view.itemError.text)}</p>` : '';
  const who = q.answered_by_name ? `${q.answered_by_name} replied` : 'Your doctor replied';
  const date = shortDate(q.answered_at);
  return `
    <li class="item question" data-id="${q.id}">
      ${metaHtml(q)}
      <p class="question-text">${esc(q.question)}</p>
      ${voicePlayerHtml(q.voice?.question, 'your question')}
      <section class="doctor-answer" aria-labelledby="answer-head-${q.id}">
        <h2 class="doctor-answer-head" id="answer-head-${q.id}">${esc(who)}${date ? ` &middot; ${esc(date)}` : ''}</h2>
        <p class="doctor-answer-text">${esc(q.answer)}</p>
        ${voicePlayerHtml(q.voice?.answer, 'doctor reply')}
      </section>
      ${error}
      <div class="question-actions">
        <button type="button" class="btn btn-secondary btn-small resolve-btn" data-id="${q.id}" ${view.busy ? 'disabled' : ''}>${busy ? 'Saving...' : 'Mark resolved'}</button>
      </div>
    </li>`;
}

function resolvedHtml(q) {
  return `
    <li class="item question is-resolved">
      ${metaHtml(q)}
      <p class="question-text">${esc(q.question)}</p>
      ${voicePlayerHtml(q.voice?.question, 'your question')}
      ${q.reason && q.source !== 'patient' ? `<p class="question-reason"><span class="question-reason-label">Why:</span> ${esc(q.reason)}</p>` : ''}
      ${q.answer ? `<p class="question-reason"><span class="question-reason-label">${esc(q.answered_by_name || 'Your doctor')} replied:</span> ${esc(q.answer)}</p>` : ''}
      ${voicePlayerHtml(q.voice?.answer, 'doctor reply')}
    </li>`;
}

function formHtml() {
  if (!view.doctors.length) return '';
  const form = view.form;
  const disabled = form.sending || !form.text.trim();
  return `
    <form class="direct-box" id="directBox" novalidate>
      <h2 class="section-title">Ask a doctor directly</h2>
      <div class="field">
        <label for="directDoctor">Doctor</label>
        <select id="directDoctor">
          ${view.doctors.map(d => `<option value="${esc(d.id)}" ${d.id === form.doctorId ? 'selected' : ''}>${esc(d.name)}</option>`).join('')}
        </select>
      </div>
      <div class="field">
        <label for="directQuestion">Your question</label>
        <textarea id="directQuestion" maxlength="${MAX_QUESTION}" ${form.sending ? 'disabled' : ''}>${esc(form.text)}</textarea>
      </div>
      ${form.error ? `<p class="form-error" role="alert">${esc(form.error)}</p>` : ''}
      <div class="form-actions">
        <button type="submit" class="btn btn-primary btn-small" id="directSubmit" ${disabled ? 'disabled' : ''}>${form.sending ? 'Sending...' : 'Send'}</button>
      </div>
      <div id="patientVoiceComposer"></div>
      ${form.sent ? `
        <div class="direct-done">
          ${form.emergency ? emergencyHtml(view.signs) : ''}
          <p class="direct-sent" role="status">${CHECK_ICON}<span>Sent to ${esc(form.sent)}. You'll see the reply below.</span></p>
        </div>` : ''}
    </form>`;
}

async function sendDirect(panel) {
  const form = view.form;
  const question = form.text.trim();
  if (!question || view.voiceActive || form.sending || form.doctorId === null) return;
  // Sending disables the form, which drops focus; it returns to the question
  // box afterwards unless the patient has moved on.
  const hadFocus = panel.contains(document.activeElement);
  form.sending = true;
  form.error = '';
  render(panel);
  try {
    const data = await askDoctorDirectly(form.doctorId, question);
    const created = data.question;
    if (created) {
      view.questions = [created, ...(view.questions || []).filter(q => q.id !== created.id)];
    }
    form.text = '';
    form.sent = (created && created.doctor_name) || 'your doctor';
    form.emergency = !!(data.emergency && data.emergency.banner);
    notify();
  } catch (error) {
    form.error = error.message;
  }
  form.sending = false;
  render(panel);
  const lost = !document.activeElement || document.activeElement === document.body;
  if (hadFocus && lost) {
    const input = panel.querySelector('#directQuestion');
    if (input) input.focus();
  }
}

function render(panel) {
  if (panel.hidden) return;
  // Everything is rebuilt: the focused field gets its focus, caret, selection
  // and scroll back.
  const active = document.activeElement;
  const refocus = active && panel.contains(active) && active.id ? active.id : null;
  const caret = refocus && typeof active.selectionStart === 'number'
    ? {start: active.selectionStart, end: active.selectionEnd,
       direction: active.selectionDirection, scroll: active.scrollTop}
    : null;
  let body;
  if (view.error && !view.questions) {
    body = `<p class="form-error" role="alert">${esc(view.error)}</p>`;
  } else if (!view.questions) {
    body = '<p class="busy-note" role="status"><span class="spinner" aria-hidden="true"></span>Loading your questions...</p>';
  } else {
    const open = view.questions.filter(q => q.status === 'open');
    const answered = view.questions.filter(q => q.status === 'answered');
    const resolved = view.questions.filter(q => q.status === 'resolved');
    body = `
      ${open.length
        ? `<ul class="item-list">${open.map(openHtml).join('')}</ul>`
        : answered.length ? '' : `<div class="card empty-inline">
             <p class="section-title">${resolved.length ? 'No open questions' : 'No questions yet'}</p>
             <p class="section-text">Questions the assistant cannot answer from your sources, and instructions you mark as not matching your document, are added here.</p>
           </div>`}
      ${answered.length ? `<ul class="item-list answered-list">${answered.map(answeredHtml).join('')}</ul>` : ''}
      ${resolved.length ? `
        <details class="resolved" ${view.resolvedOpen ? 'open' : ''}>
          <summary class="resolved-summary">Resolved <span class="count">${resolved.length}</span></summary>
          <ul class="item-list">${resolved.map(resolvedHtml).join('')}</ul>
        </details>` : ''}`;
  }

  clearVoice(panel);
  panel.innerHTML = `
    <div class="careteam split">
      <div class="split-aside">
        <div class="page-head">
          <p class="eyebrow">Care team</p>
          <h1 class="page-title">Questions for your <em>care team</em></h1>
          <p class="lead">Bring these to your next visit or call. Mark one as resolved once you have an answer.</p>
        </div>
      </div>
      <div class="split-main">
        ${formHtml()}
        ${body}
      </div>
    </div>`;

  const box = panel.querySelector('#directBox');
  if (box) {
    box.querySelector('#directDoctor').addEventListener('change', event => {
      const picked = view.doctors.find(d => String(d.id) === event.target.value);
      if (picked) {
        view.form.doctorId = picked.id;
        view.form.chosen = true;
      }
    });
    box.querySelector('#directQuestion').addEventListener('input', event => {
      view.form.text = event.target.value;
      view.form.sent = null;
      box.querySelector('#directSubmit').disabled = view.voiceActive || view.form.sending || !view.form.text.trim();
    });
    box.addEventListener('submit', event => {
      event.preventDefault();
      sendDirect(panel);
    });
  }
  if (refocus) {
    const again = panel.querySelector('#' + CSS.escape(refocus));
    if (again && !again.disabled) {
      again.focus();
      if (caret && typeof again.setSelectionRange === 'function') {
        again.setSelectionRange(caret.start, caret.end, caret.direction);
        again.scrollTop = caret.scroll;
      }
    }
  }

  panel.querySelectorAll('.resolve-btn').forEach(button => {
    button.addEventListener('click', () => resolve(panel, Number(button.dataset.id)));
  });
  const details = panel.querySelector('.resolved');
  if (details) details.addEventListener('toggle', () => { view.resolvedOpen = details.open; });
  mountVoicePlayers(panel);
  const voiceBox = panel.querySelector('#patientVoiceComposer');
  if (voiceBox && !view.form.sending) {
    attachVoiceComposer(voiceBox, {
      label: 'Or send your doctor a voice question',
      context: () => view.form.doctorId,
      onBusy: busy => {
        view.voiceActive = busy;
        const doctor = panel.querySelector('#directDoctor');
        const send = panel.querySelector('#directSubmit');
        if (doctor) doctor.disabled = busy;
        if (send) send.disabled = busy || view.form.sending || !view.form.text.trim();
      },
      onSend: async (blob, filename, doctorId) => {
        const form = new FormData();
        form.append('doctor_id', String(doctorId));
        form.append('file', blob, filename);
        const out = await api('/api/voice/questions', {method: 'POST', form});
        view.questions = [out.question, ...(view.questions || [])];
        view.form.sent = out.question.doctor_name || 'your doctor';
        view.form.emergency = false;
        notify();
        render(panel);
      },
    });
  }
}
