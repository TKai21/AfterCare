// Ask tab: the cited Q&A conversation about the confirmed discharge document.
import {api, escapeHtml, registerTab} from '/static/api.js';
import {refreshCareTeam} from '/static/careteam.js';
import {
  EMERGENCY_COPY, MAX_QUESTION, askDoctorDirectly, emergencyHtml as sharedEmergencyHtml,
  loadDoctors, pickDoctor, rememberDoctor, warningSignsOf,
} from '/static/doctors.js';

const esc = escapeHtml;

const UNCONFIGURED = 'The assistant is not configured on this server yet.';
const TEMPORARY = 'The assistant could not answer right now. Please try again.';
const SUGGESTIONS = [
  'What warning signs should I watch for?',
  'What does my document say about what I can eat?',
  'What should I do if I miss a dose?',
];
const COARSE = window.matchMedia('(pointer: coarse)');

// UI state that survives re-renders and tab switches.
const view = {
  docId: null,
  status: 'idle',    // 'idle' | 'loading' | 'ready' | 'failed'
  messages: [],      // [{role, content, payload}]
  busy: false,
  notice: null,      // {kind: 'unconfigured' | 'error', text}
  draft: '',
  doctors: [],       // [{id, name}]
  doctorId: null,    // the doctor whose clone is shown
  drafts: {},        // unsent text of the other doctors' tabs
  direct: {},        // per doctor: {open, text, touched, sending, sent, emergency, error}
  open: new Set(),   // 'messageIndex:sourceNumber' of expanded sources
};

let panelRef = null;
let ctxRef = null;

function show(panel, ctx) {
  panelRef = panel;
  ctxRef = ctx;
  const doc = ctx.document;
  if (doc.id !== view.docId) {
    view.docId = doc.id;
    view.status = 'idle';
    view.messages = [];
    view.notice = null;
    view.drafts = {};
    view.direct = {};
    view.open.clear();
  }
  renderShell(panel);
  if (view.status === 'idle' || view.status === 'failed') loadConversation(doc.id);
}

registerTab('ask', {show});

const isVisible = () => panelRef && !panelRef.hidden && panelRef.querySelector('.ask');

// ---------- requests ----------

function noticeFor(error) {
  if (error.status === 503) return {kind: 'unconfigured', text: UNCONFIGURED};
  if (error.status === 502) return {kind: 'error', text: TEMPORARY};
  return {kind: 'error', text: error.message};
}

const currentDoctor = () => view.doctors.find(d => d.id === view.doctorId) || null;
const doctorName = () => (currentDoctor() ? currentDoctor().name : 'your doctor');

async function loadConversation(docId) {
  view.status = 'loading';
  view.notice = null;
  update();
  try {
    if (!view.doctors.length) {
      view.doctors = await loadDoctors();
      if (view.docId !== docId) return;
      const picked = pickDoctor(view.doctors);
      if (!picked) throw new Error('No doctors are available yet.');
      view.doctorId = picked.id;
      view.draft = view.drafts[picked.id] || '';
    }
    const doctorId = view.doctorId;
    const data = await api('/api/conversation?doctor_id=' + encodeURIComponent(doctorId));
    if (view.docId !== docId || view.doctorId !== doctorId) return;
    view.messages = (data && data.messages) || [];
    view.status = 'ready';
  } catch (error) {
    if (view.docId !== docId) return;
    view.notice = noticeFor(error);
    view.status = 'failed';
  }
  update();
  if (view.messages.length) scrollToLatest('auto');
}

function switchDoctor(id) {
  if (id === view.doctorId || view.busy) return;
  view.drafts[view.doctorId] = view.draft;
  view.doctorId = id;
  view.draft = view.drafts[id] || '';
  view.messages = [];
  view.notice = null;
  view.open.clear();
  rememberDoctor(id);
  renderShell(panelRef);
  loadConversation(view.docId);
}

async function ask(question) {
  const docId = view.docId;
  const doctorId = view.doctorId;
  const pending = {role: 'user', content: question};
  view.messages.push(pending);
  view.busy = true;
  view.notice = null;
  view.draft = '';
  update({clearInput: true});
  scrollToLatest('smooth');
  try {
    const payload = await api('/api/ask', {method: 'POST', json: {question, doctor_id: doctorId}});
    if (view.docId !== docId || view.doctorId !== doctorId) return;
    view.messages.push({role: 'assistant', content: payload.answer, payload});
    view.busy = false;
    update();
    refreshCareTeam();
    announce('Answer received.');
    scrollToLatest('smooth');
  } catch (error) {
    if (view.docId !== docId || view.doctorId !== doctorId) return;
    view.messages = view.messages.filter(m => m !== pending);
    view.busy = false;
    view.notice = noticeFor(error);
    if (!view.draft) view.draft = question;  // keep the question so it can be sent again
    update({restoreInput: true});
    if (error.status === 409 && ctxRef) ctxRef.reloadDocument();
  }
}

// ---------- answer text ----------

// Escapes first, then turns [n] / [n, m] markers into superscript buttons for
// sources that exist on this answer. Unknown numbers stay as plain text.
function citeHtml(escapedText, numbers) {
  return escapedText.replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (whole, list) => {
    const nums = list.split(',').map(n => Number(n.trim()));
    if (!nums.every(n => numbers.has(n))) return whole;
    return nums.map(n =>
      `<sup class="cite-sup"><button type="button" class="cite" data-n="${n}" aria-label="Source ${n}">${n}</button></sup>`
    ).join('');
  });
}

function answerTextHtml(text, numbers) {
  const blocks = String(text || '').trim().split(/\n\s*\n/);
  return blocks.map(block => {
    const lines = block.split('\n').map(l => l.trim()).filter(Boolean);
    if (!lines.length) return '';
    const bullet = /^[-*]\s+/;
    if (lines.every(l => bullet.test(l))) {
      return '<ul class="answer-list">' +
        lines.map(l => `<li>${citeHtml(esc(l.replace(bullet, '')), numbers)}</li>`).join('') + '</ul>';
    }
    return `<p>${lines.map(l => citeHtml(esc(l), numbers)).join('<br>')}</p>`;
  }).join('');
}

function sourceLabel(source) {
  if (source.kind === 'document') return 'Your discharge document';
  if (source.kind === 'library') return 'Education library: ' + (source.title || 'Untitled');
  if (source.kind === 'food') return 'Food data: ' + (source.title || 'Untitled');
  if (source.kind === 'doctor') return doctorLabel(source.title);
  return source.title || 'Source';
}

// "Dr. Ting Kai's advice, Nov 12, 2026" -> "Dr. Ting Kai's advice (Nov 12, 2026)".
function doctorLabel(title) {
  const text = String(title || '');
  const match = /^(.+?), ([A-Z][a-z]{2} \d{1,2}, \d{4})$/.exec(text);
  return match ? `${match[1]} (${match[2]})` : (text || 'Doctor advice');
}

const EXTERNAL_ICON = '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M9 3h4v4M13 3L7.5 8.5M11.5 9.5V12a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V5.5a1 1 0 0 1 1-1h2.5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>';
const CHECK_ICON = '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M3.5 8.5l3 3 6-7" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
const INFO_ICON = '<svg viewBox="0 0 20 20" width="20" height="20" aria-hidden="true"><circle cx="10" cy="10" r="8" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M10 9v5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><circle cx="10" cy="6.3" r="1.1" fill="currentColor"/></svg>';
const SEND_ICON = '<svg viewBox="0 0 20 20" width="18" height="18" aria-hidden="true"><path d="M10 16V4m0 0L5 9m5-5l5 5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';

function safeUrl(url) {
  return typeof url === 'string' && /^https:\/\//i.test(url) ? url : '';
}

function sourcesHtml(index, sources) {
  if (!sources.length) return '';
  return `
    <section class="sources" aria-label="Sources">
      <h3 class="sources-title">Sources</h3>
      <ol class="source-list">
        ${sources.map(source => {
          const key = index + ':' + source.number;
          const url = safeUrl(source.url);
          return `
            <li>
              <details class="source-item" id="src-${index}-${esc(source.number)}" data-key="${esc(key)}" ${view.open.has(key) ? 'open' : ''}>
                <summary>
                  <span class="source-num">${esc(source.number)}</span>
                  <span class="source-label">${esc(sourceLabel(source))}</span>
                </summary>
                <div class="source-body">
                  <blockquote class="quote source-quote">${esc(source.content)}</blockquote>
                  ${url ? `<a class="source-link" href="${esc(url)}" target="_blank" rel="noopener noreferrer">Open source${EXTERNAL_ICON}<span class="visually-hidden"> (opens in a new tab)</span></a>` : ''}
                </div>
              </details>
            </li>`;
        }).join('')}
      </ol>
    </section>`;
}

function emergencyHtml() {
  return sharedEmergencyHtml(warningSignsOf(ctxRef && ctxRef.document));
}

function assistantHtml(message, index) {
  const payload = message.payload || {kind: 'answer', answer: message.content, sources: []};
  const sources = Array.isArray(payload.sources) ? payload.sources : [];
  const numbers = new Set(sources.map(s => Number(s.number)));
  const flagged = payload.kind === 'not_covered' || payload.flagged === true;
  const emergency = payload.kind === 'emergency' || !!(payload.emergency && payload.emergency.banner);
  // The server's fixed emergency answer is the banner copy itself; the banner alone says it.
  if (emergency && String(payload.answer || '').trim() === EMERGENCY_COPY && !sources.length) {
    return `<li class="msg msg-assistant" data-index="${index}">${emergencyHtml()}</li>`;
  }
  return `
    <li class="msg msg-assistant" data-index="${index}">
      ${emergency ? emergencyHtml() : ''}
      <article class="answer${payload.kind === 'out_of_scope' ? ' is-muted' : ''}" aria-label="AI clone answer">
        <p class="clone-label">AI clone of ${esc(payload.doctor_name || doctorName())}</p>
        <div class="answer-text">${answerTextHtml(payload.answer, numbers)}</div>
        ${flagged ? `<a class="chip" href="#careteam">${CHECK_ICON}${payload.doctor_name ? 'Sent to ' + esc(payload.doctor_name) : 'Added to your care-team questions'}</a>` : ''}
        ${sourcesHtml(index, sources)}
      </article>
    </li>`;
}

function userHtml(message, index) {
  return `
    <li class="msg msg-user" data-index="${index}">
      <p class="bubble"><span class="visually-hidden">You asked: </span>${esc(message.content)}</p>
    </li>`;
}

// ---------- rendering ----------

function renderShell(panel) {
  panel.innerHTML = `
    <div class="ask">
      <div class="page-head">
        <h1 class="page-title">Ask about your recovery</h1>
        <p class="lead" id="askLead"></p>
      </div>
      <div id="askTabs"></div>
      <div id="askBody"></div>
      <div id="askNotice"></div>
      <form class="composer" id="askForm" novalidate>
        <label class="visually-hidden" for="askInput">Your question</label>
        <div class="composer-box">
          <textarea id="askInput" class="composer-input" rows="1" maxlength="${MAX_QUESTION}" placeholder="Ask a question about your recovery"></textarea>
          <button type="submit" class="btn btn-primary composer-send" id="askSend" aria-label="Send">${SEND_ICON}<span class="send-text">Send</span></button>
        </div>
        <p class="composer-hint">Press Enter to send, Shift + Enter for a new line.</p>
      </form>
      <div id="askDirect"></div>
      <p class="visually-hidden" role="status" id="askStatus"></p>
    </div>`;

  const input = panel.querySelector('#askInput');
  input.value = view.draft;
  input.addEventListener('input', () => {
    view.draft = input.value;
    autosize(input);
    syncSend();
  });
  input.addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing && !COARSE.matches) {
      event.preventDefault();
      submit();
    }
  });
  panel.querySelector('#askForm').addEventListener('submit', event => {
    event.preventDefault();
    submit();
  });
  panel.querySelector('#askDirect').addEventListener('click', onDirectClick);
  panel.querySelector('#askDirect').addEventListener('input', onDirectInput);
  panel.querySelector('#askDirect').addEventListener('submit', onDirectSubmit);
  update();
}

function submit() {
  const question = view.draft.trim();
  if (!question || view.busy || inputLocked()) return;
  ask(question);
}

const inputLocked = () => view.status !== 'ready';

function autosize(input) {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight + 2, 168) + 'px';
}

function syncSend() {
  const send = panelRef.querySelector('#askSend');
  if (send) send.disabled = view.busy || inputLocked() || !view.draft.trim();
}

function bodyHtml() {
  if (view.status === 'loading' || view.status === 'idle') {
    return '<p class="busy-note ask-loading" role="status"><span class="spinner" aria-hidden="true"></span>Loading your conversation...</p>';
  }
  if (view.status === 'failed') return '';
  if (!view.messages.length && !view.busy) {
    return `
      <div class="card ask-empty">
        <h2 class="section-title">Ask anything about your discharge instructions</h2>
        <p class="section-text">For example:</p>
        <div class="suggestions">
          ${SUGGESTIONS.map(s => `<button type="button" class="suggestion">${esc(s)}</button>`).join('')}
        </div>
      </div>`;
  }
  return `
    <ol class="chat" aria-label="Conversation">
      ${view.messages.map((m, i) => (m.role === 'user' ? userHtml(m, i) : assistantHtml(m, i))).join('')}
      ${view.busy ? `
        <li class="msg msg-assistant">
          <div class="answer typing">
            <span class="dots" aria-hidden="true"><i></i><i></i><i></i></span>
            <span class="typing-text">Checking your document and sources...</span>
          </div>
        </li>` : ''}
    </ol>`;
}

function noticeHtml() {
  const notice = view.notice;
  if (!notice) return '';
  if (notice.kind === 'error') {
    return `
      <div class="form-error ask-error" role="alert">
        <span>${esc(notice.text)}</span>
        ${view.status === 'failed' ? '<button type="button" class="btn-link" id="retryLoad">Try again</button>' : ''}
      </div>`;
  }
  return `<p class="notice" role="status">${INFO_ICON}<span>${esc(notice.text)}</span></p>`;
}

function renderTabs() {
  const box = panelRef.querySelector('#askTabs');
  if (!view.doctors.length) {
    box.innerHTML = '';
    return;
  }
  const sig = view.doctors.map(d => d.id + ':' + d.name).join('|') + '#' + view.doctorId + '#' + view.busy;
  if (box.dataset.sig === sig) return;
  box.dataset.sig = sig;
  box.innerHTML = `
    <div class="doctor-picker" role="group" aria-label="Choose a doctor">
      ${view.doctors.map(d => `
        <button type="button" class="doctor-tab" data-id="${esc(d.id)}" aria-pressed="${d.id === view.doctorId}" ${view.busy ? 'disabled' : ''}>${esc(d.name)}</button>`).join('')}
    </div>`;
  box.querySelectorAll('.doctor-tab').forEach(button => {
    button.addEventListener('click', () => {
      const id = view.doctors.find(d => String(d.id) === button.dataset.id);
      if (id) switchDoctor(id.id);
    });
  });
}

// ---------- ask the real doctor ----------

const directState = () => {
  if (!view.direct[view.doctorId]) {
    view.direct[view.doctorId] = {open: false, text: '', touched: false, sending: false, sent: null, emergency: false, error: ''};
  }
  return view.direct[view.doctorId];
};

function lastQuestion() {
  const users = view.messages.filter(m => m.role === 'user');
  return users.length ? String(users[users.length - 1].content || '') : '';
}

function directHtml() {
  if (view.status !== 'ready' || !currentDoctor()) return '';
  const name = doctorName();
  const state = directState();
  if (state.sent) {
    return `
      <div class="direct">
        ${state.emergency ? emergencyHtml() : ''}
        <p class="direct-sent" role="status">${CHECK_ICON}<span>Sent to ${esc(state.sent)}. You'll see the reply in <a href="#careteam">Care team</a>.</span></p>
      </div>`;
  }
  if (!state.open) {
    return `
      <div class="direct">
        <p class="direct-line">Want the real ${esc(name)}? <button type="button" class="btn btn-secondary btn-small" data-direct="open">Ask ${esc(name)} directly</button></p>
      </div>`;
  }
  return `
    <form class="direct direct-form" novalidate>
      <label class="direct-label" for="directInput">Your question for ${esc(name)}</label>
      <textarea id="directInput" class="direct-input" rows="3" maxlength="${MAX_QUESTION}">${esc(state.text)}</textarea>
      ${state.error ? `<p class="form-error" role="alert">${esc(state.error)}</p>` : ''}
      <div class="form-actions">
        <button type="button" class="btn btn-secondary btn-small" data-direct="cancel" ${state.sending ? 'disabled' : ''}>Cancel</button>
        <button type="submit" class="btn btn-primary btn-small" id="directSend" ${state.sending || !state.text.trim() ? 'disabled' : ''}>${state.sending ? 'Sending...' : 'Send to ' + esc(name)}</button>
      </div>
    </form>`;
}

function renderDirect() {
  const box = panelRef.querySelector('#askDirect');
  const focused = document.activeElement && document.activeElement.id === 'directInput';
  box.innerHTML = directHtml();
  const input = box.querySelector('#directInput');
  if (input && focused) input.focus();
}

function onDirectClick(event) {
  const button = event.target.closest('[data-direct]');
  if (!button) return;
  const state = directState();
  if (button.dataset.direct === 'open') {
    state.open = true;
    if (!state.touched) state.text = lastQuestion();
    renderDirect();
    const input = panelRef.querySelector('#directInput');
    if (input) input.focus();
  } else if (button.dataset.direct === 'cancel' && !state.sending) {
    state.open = false;
    renderDirect();
  }
}

function onDirectInput(event) {
  if (event.target.id !== 'directInput') return;
  const state = directState();
  state.text = event.target.value;
  state.touched = true;
  const send = panelRef.querySelector('#directSend');
  if (send) send.disabled = state.sending || !state.text.trim();
}

async function onDirectSubmit(event) {
  event.preventDefault();
  const doctorId = view.doctorId;
  const state = directState();
  const question = state.text.trim();
  if (!question || state.sending) return;
  state.sending = true;
  state.error = '';
  renderDirect();
  try {
    const data = await askDoctorDirectly(doctorId, question);
    state.sending = false;
    state.sent = (data && data.question && data.question.doctor_name) || (view.doctors.find(d => d.id === doctorId) || {}).name || 'your doctor';
    state.emergency = !!(data && data.emergency && data.emergency.banner);
    refreshCareTeam();
  } catch (error) {
    state.sending = false;
    state.error = error.message;
  }
  if (isVisible() && view.doctorId === doctorId) renderDirect();
}

function update({clearInput = false, restoreInput = false} = {}) {
  if (!isVisible()) return;
  const panel = panelRef;
  const name = doctorName();
  panel.querySelector('#askLead').textContent = currentDoctor()
    ? `Answers come from your discharge document and ${name}'s earlier advice, given by ${name}'s AI clone.`
    : 'Answers come from your discharge document and your doctor\'s earlier advice, given by an AI clone.';
  panel.querySelector('#askInput').placeholder = currentDoctor() ? `Ask ${name}'s AI clone...` : 'Ask a question about your recovery';
  renderTabs();
  renderDirect();
  panel.querySelector('#askBody').innerHTML = bodyHtml();
  panel.querySelector('#askNotice').innerHTML = noticeHtml();
  const input = panel.querySelector('#askInput');
  const locked = inputLocked();
  input.disabled = locked;
  panel.querySelector('#askForm').classList.toggle('is-locked', !!locked);
  if (clearInput) input.value = '';
  if (restoreInput && !input.value) input.value = view.draft;
  autosize(input);
  syncSend();

  panel.querySelectorAll('.suggestion').forEach(button => {
    button.addEventListener('click', () => {
      if (view.busy || inputLocked()) return;
      ask(button.textContent);
    });
  });
  panel.querySelectorAll('.source-item').forEach(details => {
    details.addEventListener('toggle', () => {
      if (details.open) view.open.add(details.dataset.key);
      else view.open.delete(details.dataset.key);
    });
  });
  panel.querySelectorAll('.cite').forEach(button => {
    button.addEventListener('click', () => {
      const index = button.closest('.msg').dataset.index;
      const details = panel.querySelector(`#src-${index}-${button.dataset.n}`);
      if (!details) return;
      details.open = true;
      details.classList.remove('is-flash');
      void details.offsetWidth;  // restart the highlight animation
      details.classList.add('is-flash');
      details.scrollIntoView({behavior: 'smooth', block: 'nearest'});
      details.querySelector('summary').focus({preventScroll: true});
    });
  });
  const retry = panel.querySelector('#retryLoad');
  if (retry) retry.addEventListener('click', () => loadConversation(view.docId));
}

function announce(text) {
  const status = panelRef && panelRef.querySelector('#askStatus');
  if (status) status.textContent = text;
}

// Bring the start of the latest question into view, so its answer reads from the top.
function scrollToLatest(behavior) {
  if (!isVisible()) return;
  const users = panelRef.querySelectorAll('.msg-user');
  const last = users[users.length - 1];
  if (last) last.scrollIntoView({behavior, block: 'start'});
}
