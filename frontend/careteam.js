// Care team tab: questions to bring to the care team, open first, resolved collapsed.
import {api, escapeHtml, registerTab} from '/static/api.js';

const esc = escapeHtml;

const SOURCE_LABELS = {
  agent: 'From the assistant',
  validation_fallback: 'From the assistant',
  rejected_item: 'From your instruction review',
};

const view = {
  questions: null,
  error: null,
  busy: null,          // id being resolved
  itemError: null,     // {id, text}
  resolvedOpen: false,
};

function show(panel) {
  markAnswersSeen();
  render(panel);
  load(panel);
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
  const parts = [formatDate(q.created_at), SOURCE_LABELS[q.source]].filter(Boolean);
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
      ${q.reason ? `<p class="question-reason"><span class="question-reason-label">Why:</span> ${esc(q.reason)}</p>` : ''}
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
      ${q.reason ? `<p class="question-reason"><span class="question-reason-label">Why:</span> ${esc(q.reason)}</p>` : ''}
      ${q.answer ? `<p class="question-reason"><span class="question-reason-label">${esc(q.answered_by_name || 'Your doctor')} replied:</span> ${esc(q.answer)}</p>` : ''}
    </li>`;
}

function render(panel) {
  if (panel.hidden) return;
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
      ${body}
    </div>`;

  panel.querySelectorAll('.resolve-btn').forEach(button => {
    button.addEventListener('click', () => resolve(panel, Number(button.dataset.id)));
  });
  const details = panel.querySelector('.resolved');
  if (details) details.addEventListener('toggle', () => { view.resolvedOpen = details.open; });
}
