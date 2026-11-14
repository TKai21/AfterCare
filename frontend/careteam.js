// Care team tab: questions to bring to the care team, open first, resolved collapsed.
import {api, escapeHtml, registerTab} from '/static/api.js';
import {
  MAX_QUESTION, askDoctorDirectly, emergencyHtml, loadDoctors, pickDoctor, warningSignsOf,
} from '/static/doctors.js';

const esc = escapeHtml;

const SOURCE_LABELS = {
  agent: 'From the assistant',
  validation_fallback: 'From the assistant',
  rejected_item: 'From your instruction review',
  patient: 'Asked by you',
};

const view = {
  questions: null,
  error: null,
  busy: null,          // id being resolved
  itemError: null,     // {id, text}
  resolvedOpen: false,
  doctors: [],
  form: {doctorId: null, text: '', sending: false, error: '', sent: null, emergency: false},
  signs: [],           // active warning signs, for the emergency banner
};

function show(panel, ctx) {
  view.signs = warningSignsOf(ctx && ctx.document);
  markAnswersSeen();
  render(panel);
  load(panel);
  loadDoctors().then(doctors => {
    view.doctors = doctors;
    const current = doctors.find(d => d.id === view.form.doctorId);
    if (!current) {
      const picked = pickDoctor(doctors);
      view.form.doctorId = picked ? picked.id : null;
    }
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
      <section class="doctor-answer" aria-labelledby="answer-head-${q.id}">
        <h2 class="doctor-answer-head" id="answer-head-${q.id}">${esc(who)}${date ? ` &middot; ${esc(date)}` : ''}</h2>
        <p class="doctor-answer-text">${esc(q.answer)}</p>
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
      ${q.reason && q.source !== 'patient' ? `<p class="question-reason"><span class="question-reason-label">Why:</span> ${esc(q.reason)}</p>` : ''}
      ${q.answer ? `<p class="question-reason"><span class="question-reason-label">${esc(q.answered_by_name || 'Your doctor')} replied:</span> ${esc(q.answer)}</p>` : ''}
    </li>`;
}

function formHtml() {
  if (!view.doctors.length) return '';
  const form = view.form;
  const disabled = form.sending || !form.text.trim();
  return `
    <form class="direct-box" id="directBox" novalidate>
      <h2 class="section-title">Ask a doctor directly</h2>
      ${form.sent ? `
        ${form.emergency ? emergencyHtml(view.signs) : ''}
        <p class="direct-sent" role="status">Sent to ${esc(form.sent)}. You'll see the reply below.</p>` : ''}
      <div class="field">
        <label for="directDoctor">Doctor</label>
        <select id="directDoctor">
          ${view.doctors.map(d => `<option value="${esc(d.id)}" ${d.id === form.doctorId ? 'selected' : ''}>${esc(d.name)}</option>`).join('')}
        </select>
      </div>
      <div class="field">
        <label for="directQuestion">Your question</label>
        <textarea id="directQuestion" maxlength="${MAX_QUESTION}">${esc(form.text)}</textarea>
      </div>
      ${form.error ? `<p class="form-error" role="alert">${esc(form.error)}</p>` : ''}
      <div class="form-actions">
        <button type="submit" class="btn btn-primary btn-small" id="directSubmit" ${disabled ? 'disabled' : ''}>${form.sending ? 'Sending...' : 'Send'}</button>
      </div>
    </form>`;
}

async function sendDirect(panel) {
  const form = view.form;
  const question = form.text.trim();
  if (!question || form.sending || form.doctorId === null) return;
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
}

function render(panel) {
  if (panel.hidden) return;
  const active = document.activeElement;
  const refocus = active && panel.contains(active) && active.id ? active.id : null;
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

  panel.innerHTML = `
    <div class="careteam">
      <div class="page-head">
        <h1 class="page-title">Questions for your care team</h1>
        <p class="lead">Bring these to your next visit or call. Mark one as resolved once you have an answer.</p>
      </div>
      ${formHtml()}
      ${body}
    </div>`;

  const box = panel.querySelector('#directBox');
  if (box) {
    box.querySelector('#directDoctor').addEventListener('change', event => {
      const picked = view.doctors.find(d => String(d.id) === event.target.value);
      if (picked) view.form.doctorId = picked.id;
    });
    box.querySelector('#directQuestion').addEventListener('input', event => {
      view.form.text = event.target.value;
      view.form.sent = null;
      box.querySelector('#directSubmit').disabled = view.form.sending || !view.form.text.trim();
    });
    box.addEventListener('submit', event => {
      event.preventDefault();
      sendDirect(panel);
    });
  }
  if (refocus) {
    const again = panel.querySelector('#' + refocus);
    if (again) again.focus();
  }

  panel.querySelectorAll('.resolve-btn').forEach(button => {
    button.addEventListener('click', () => resolve(panel, Number(button.dataset.id)));
  });
  const details = panel.querySelector('.resolved');
  if (details) details.addEventListener('toggle', () => { view.resolvedOpen = details.open; });
}
