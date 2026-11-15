// Doctor page (/doctor): the question queue, "My guidance" and the drug label
// review (rendered by labels.js).
import {api, clearSession, confirmDialog, escapeHtml} from '/static/api.js';
import {countPendingLabels, hideLabels, showLabels} from '/static/labels.js';
import {enter, initials, syncTabs} from '/static/shell.js';
import {attachVoiceComposer, clearVoice, mountVoicePlayers, voicePlayerHtml} from '/static/voice.js';

const esc = escapeHtml;
const SUGGEST_DELAY_MS = 800;
const SUGGEST_FAILED = 'Could not suggest tags. Add them yourself.';

const state = {
  open: [], answered: [], guidance: null,
  labelCount: 0,        // drug labels waiting for review
  labelCountFromTab: false,
  tab: 'questions',
  notice: '',           // short success line on the Questions tab
  queueError: '',
  guidanceError: '',
  filters: {scope: 'all', status: 'active'},
  drafts: new Map(),    // open question id -> answer draft
  draftStats: null,     // /api/doctor/draft-stats
  answeredOpen: false,
  openPanels: new Set(), // expanded context panels, keyed id:kind
  editing: new Map(),   // guidance id -> edit draft
  rowErrors: new Map(), // guidance id -> message
  busyRows: new Set(),
  rulesOpen: new Set(), // guidance ids whose rules disclosure is open
};

const main = document.getElementById('main');

// ---------- helpers ----------

// Server timestamps are UTC without an offset.
function formatDate(iso) {
  if (!iso) return '';
  const stamp = /[zZ]|[+-]\d\d:?\d\d$/.test(iso) ? iso : iso + 'Z';
  const d = new Date(stamp);
  if (Number.isNaN(d.getTime())) return '';
  return d.toLocaleDateString('en-US', {month: 'short', day: 'numeric'});
}

const uniqueTags = tags => {
  const seen = new Set();
  return tags.map(t => String(t).trim()).filter(t => {
    const key = t.toLowerCase();
    if (!t || seen.has(key)) return false;
    seen.add(key);
    return true;
  });
};

const TABS = ['questions', 'guidance', 'labels'];

function tabFromHash() {
  const tab = window.location.hash.slice(1);
  return TABS.includes(tab) ? tab : 'questions';
}

function chipItemsHtml(kind, tags) {
  return tags.map((t, i) => `<li class="doctor-chip">${esc(t)}<button type="button" class="doctor-chip-x" data-action="remove-tag" data-kind="${kind}" data-index="${i}" aria-label="Remove ${esc(t)}">&times;</button></li>`).join('');
}

function chipsHtml(kind, tags, label, pending = '') {
  return `
    <div class="doctor-tag-row" data-kind="${kind}">
      <span class="doctor-tag-label">${label}</span>
      <ul class="doctor-chips" aria-label="${label}">${chipItemsHtml(kind, tags)}</ul>
      <span class="doctor-tag-add">
        <input type="text" class="doctor-tag-input" data-kind="${kind}" maxlength="80" placeholder="Add" value="${esc(pending)}" aria-label="Add to ${label.toLowerCase()}">
        <button type="button" class="btn btn-secondary btn-small" data-action="add-tag" data-kind="${kind}">Add</button>
      </span>
    </div>`;
}

// ---------- Questions tab ----------

function draftFor(q) {
  if (!state.drafts.has(q.id)) {
    state.drafts.set(q.id, {
      answer: '', general: false, savedQuestion: q.question,
      drugs: [], conditions: [], tagError: '', suggesting: false,
      pending: {drugs: '', conditions: ''}, sending: false, error: '', lastSuggested: null, timer: null, seq: 0,
      clone: q.draft || null, drafting: false, draftError: '',
    });
  }
  return state.drafts.get(q.id);
}

const isOpen = (id, kind) => state.openPanels.has(id + ':' + kind) ? 'open' : '';

function contextHtml(q) {
  const c = q.context || {};
  const paragraphs = c.document_paragraphs || [];
  const items = c.confirmed_items || [];
  const convo = c.conversation || [];
  const panels = [];
  if (paragraphs.length) {
    panels.push(`<details class="doctor-context" data-key="${q.id}:doc" ${isOpen(q.id, 'doc')}><summary>Discharge document</summary>
      <div class="doctor-context-body">${paragraphs.map(p => `<p>${esc(p)}</p>`).join('')}</div></details>`);
  }
  if (items.length) {
    panels.push(`<details class="doctor-context" data-key="${q.id}:items" ${isOpen(q.id, 'items')}><summary>Confirmed instructions</summary>
      <ul class="doctor-context-body doctor-context-list">${items.map(i =>
        `<li><span class="doctor-context-cat">${esc(String(i.category || '').replace(/_/g, ' '))}</span> ${esc(i.text)}</li>`).join('')}</ul></details>`);
  }
  if (convo.length) {
    panels.push(`<details class="doctor-context" data-key="${q.id}:convo" ${isOpen(q.id, 'convo')}><summary>Recent conversation (last ${convo.length} messages)</summary>
      <ul class="doctor-context-body doctor-context-list">${convo.map(m =>
        `<li><span class="doctor-context-cat">${m.role === 'user' ? 'Patient' : 'Your AI clone'}</span> ${esc(m.content)}</li>`).join('')}</ul></details>`);
  }
  return panels.join('');
}

function suggestStatusHtml(d) {
  if (d.suggesting) return '<p class="busy-note"><span class="spinner" aria-hidden="true"></span>Suggesting tags...</p>';
  return d.tagError ? `<p class="doctor-hint">${esc(d.tagError)}</p>` : '';
}

function generalHtml(q, d) {
  if (!d.general) return '';
  return `
    <div class="doctor-general-fields">
      <div class="field">
        <label for="saved-${q.id}">Question as saved</label>
        <input type="text" id="saved-${q.id}" class="doctor-saved-question" maxlength="4000" value="${esc(d.savedQuestion)}">
        <span class="field-hint">Rewrite it so nothing specific to this patient is shared.</span>
      </div>
      ${chipsHtml('drugs', d.drugs, 'Drugs', d.pending.drugs)}
      ${chipsHtml('conditions', d.conditions, 'Conditions', d.pending.conditions)}
      <div class="doctor-suggest-status" role="status">${suggestStatusHtml(d)}</div>
    </div>`;
}

// Open and answered cards. Answering assigns a question, so an answered card
// never shows "Unassigned".
function metaLabelsHtml(q) {
  const labels = [];
  if (q.source === 'patient' || q.source === 'patient_voice') labels.push('Asked by the patient');
  if (q.doctor_id == null) labels.push('Unassigned');
  return labels.map(l => ` &middot; <span class="doctor-meta-label">${esc(l)}</span>`).join('');
}

const SPARK_ICON = '<svg class="btn-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3.75l1.6 4.65L18.25 10l-4.65 1.6L12 16.25l-1.6-4.65L5.75 10l4.65-1.6z"/><path d="M18.5 15.5l.7 1.8 1.8.7-1.8.7-.7 1.8-.7-1.8-1.8-.7 1.8-.7z"/></svg>';

const PROBLEM_TEXT = {
  no_source: 'Not in your sources',
  number: 'A number is not in the cited source',
  unsupported: 'The cited source does not say this',
  unchecked: 'Source not checked',
};

function cloneDraftHtml(q, d) {
  const c = d.clone;
  const button = `<button type="button" class="btn btn-secondary btn-small clone-draft-btn" data-action="draft" ${d.drafting ? 'disabled' : ''}>
      ${d.drafting ? '<span class="spinner" aria-hidden="true"></span>Drafting...' : `${SPARK_ICON}${c ? 'Redraft' : 'Draft with my clone'}`}</button>`;
  const error = d.draftError ? `<p class="form-error" role="alert">${esc(d.draftError)}</p>` : '';
  if (!c) return `<div class="clone-draft-bar">${button}</div>${error}`;
  const flagged = c.sentences.filter(x => !x.supported).length;
  const sentences = c.sentences.map(x => x.supported
    ? `<span class="draft-sentence">${esc(x.text)}${x.sources.map(n => `<sup class="draft-ref">${n}</sup>`).join('')}</span>`
    : `<mark class="draft-sentence is-flagged">${esc(x.text)}<span class="draft-flag">${esc(PROBLEM_TEXT[x.problem] || 'Check this')}</span></mark>`).join(' ');
  const sources = c.evidence.length ? `
      <ol class="draft-sources">${c.evidence.map(e => `
        <li><span class="draft-source-num">${e.number}</span><span class="draft-source-body"><span class="draft-source-title">${esc(e.title)}</span>
          <span class="draft-source-quote">${esc(e.quote)}</span></span></li>`).join('')}</ol>` : '';
  return `
    <section class="clone-draft" aria-label="Your AI clone's draft">
      <div class="clone-draft-head">
        <p class="clone-draft-title">${SPARK_ICON}Your clone's draft</p>
        <p class="clone-draft-sum">${flagged ? `<strong>${flagged}</strong> to check` : 'Every sentence has a source'}</p>
      </div>
      <p class="clone-draft-text">${sentences}</p>
      ${sources}
      <div class="form-actions clone-draft-actions">
        ${button}
        <button type="button" class="btn btn-secondary btn-small" data-action="use-draft" ${d.drafting ? 'disabled' : ''}>Use draft</button>
      </div>
    </section>${error}`;
}

function openCardHtml(q) {
  const d = draftFor(q);
  return `
    <li class="card doctor-card" data-id="${q.id}">
      <p class="doctor-meta"><strong>${esc(q.patient.username)}</strong> &middot; asked ${esc(formatDate(q.created_at))}${metaLabelsHtml(q)}</p>
      <p class="doctor-question">${esc(q.question)}</p>
      ${voicePlayerHtml(q.voice?.question, 'patient question')}
      ${q.reason && q.source !== 'patient' ? `<p class="doctor-reason"><span class="doctor-reason-label">Why it came here:</span> ${esc(q.reason)}</p>` : ''}
      <div class="doctor-contexts">${contextHtml(q)}</div>
      ${q.source === 'patient_voice' ? '' : `<div class="clone-draft-slot">${cloneDraftHtml(q, d)}</div>
      <div class="field">
        <label for="answer-${q.id}">Your answer</label>
        <textarea id="answer-${q.id}" class="doctor-answer-box" maxlength="4000" placeholder="Your answer...">${esc(d.answer)}</textarea>
        <span class="field-hint">Your AI clone follows the discharge document over this answer if they conflict.</span>
      </div>
      <label class="doctor-check">
        <input type="checkbox" class="doctor-general-check" ${d.general ? 'checked' : ''}>
        <span>General advice: your AI clone may use this for other patients</span>
      </label>
      <div class="doctor-general">${generalHtml(q, d)}</div>
      <div class="doctor-error">${d.error ? `<p class="form-error" role="alert">${esc(d.error)}</p>` : ''}</div>
      <div class="form-actions">
        <button type="button" class="btn btn-primary doctor-send" data-action="send">${d.sending ? 'Sending...' : 'Send answer'}</button>
      </div>`}
      <div data-doctor-voice="${q.id}"></div>
    </li>`;
}

function answeredCardHtml(q) {
  return `
    <li class="card doctor-card doctor-card-answered">
      <p class="doctor-meta"><strong>${esc(q.patient ? q.patient.username : '')}</strong> &middot; asked ${esc(formatDate(q.created_at))}${metaLabelsHtml(q)}</p>
      <p class="doctor-question">${esc(q.question)}</p>
      ${voicePlayerHtml(q.voice?.question, 'patient question')}
      <p class="doctor-reason"><span class="doctor-reason-label">${esc(q.answered_by_name || 'Doctor')} answered ${esc(formatDate(q.answered_at))}:</span> ${esc(q.answer)}</p>
      ${voicePlayerHtml(q.voice?.answer, 'your reply')}
    </li>`;
}

function draftStatsHtml() {
  const st = state.draftStats;
  if (!st || !st.sent) return '';
  const pct = Math.round(st.mean_kept * 100);
  return `
    <section class="draft-stats" aria-label="Your clone's drafts">
      <p class="draft-stats-title">Your clone's drafts</p>
      <dl class="draft-stats-grid">
        <div><dt>Sent</dt><dd>${st.sent}</dd></div>
        <div><dt>Unchanged</dt><dd>${st.unchanged}</dd></div>
        <div><dt>Avg. kept</dt><dd>${pct}<span class="draft-stats-unit">%</span></dd></div>
      </dl>
    </section>`;
}

function renderQuestions() {
  clearVoice(main);
  const cards = state.open.length
    ? `<ul class="item-list doctor-list">${state.open.map(openCardHtml).join('')}</ul>`
    : `<div class="card empty-inline"><p class="section-title">No open questions</p>
         <p class="section-text">Questions your AI clone cannot answer for patients show up here.</p></div>`;
  main.innerHTML = `
    <div class="split">
    <div class="split-aside">
    <div class="page-head"><p class="eyebrow">Inbox</p><h1 class="page-title">Patient <em>questions</em></h1>
      <p class="lead">Your answer goes to the patient. Mark it general and your AI clone can use it for others too.</p></div>
    ${draftStatsHtml()}
    </div>
    <div class="split-main">
    <p class="doctor-notice" role="status">${esc(state.notice)}</p>
    ${state.queueError ? `<p class="form-error" role="alert">${esc(state.queueError)}</p>` : ''}
    ${cards}
    ${state.answered.length ? `
      <details class="doctor-answered" ${state.answeredOpen ? 'open' : ''}>
        <summary class="resolved-summary">Answered <span class="count">${state.answered.length}</span></summary>
        <ul class="item-list doctor-list">${state.answered.map(answeredCardHtml).join('')}</ul>
      </details>` : ''}
    </div>
    </div>`;
  state.open.forEach(q => updateSend(q.id));
  mountDoctorVoice(main);
}

function mountDoctorVoice(root) {
  mountVoicePlayers(root);
  root.querySelectorAll('[data-doctor-voice]').forEach(element => {
    if (element.dataset.voiceComposerBound) return;
    const id = Number(element.dataset.doctorVoice);
    attachVoiceComposer(element, {
      label: 'Reply with a voice clip (not saved as AI guidance)',
      context: () => id,
      onBusy: busy => {
        const card = cardEl(id);
        if (!card) return;
        const draft = state.drafts.get(id);
        if (busy && draft) { clearTimeout(draft.timer); draft.seq++; draft.suggesting = false; }
        card.dataset.voiceBusy = busy ? 'true' : 'false';
        card.querySelectorAll('.doctor-answer-box, .doctor-general-check, .doctor-saved-question, .doctor-tag-input, [data-action="add-tag"], [data-action="remove-tag"]').forEach(input => { input.disabled = busy; });
        updateSend(id);
      },
      onSend: async (blob, filename) => {
        const form = new FormData();
        form.append('file', blob, filename);
        const out = await api(`/api/voice/questions/${id}/reply`, {method: 'POST', form});
        const old = state.open.find(q => q.id === id);
        const draft = state.drafts.get(id);
        if (draft) { clearTimeout(draft.timer); draft.seq++; }
        state.open = state.open.filter(q => q.id !== id);
        state.answered = [{...old, ...out.question}, ...state.answered];
        state.drafts.delete(id);
        state.notice = 'Voice reply sent.';
        renderTabs();
        if (state.tab === 'questions') renderQuestions();
      },
    });
  });
}

const cardEl = id => main.querySelector(`.doctor-card[data-id="${id}"]`);

function updateSend(id) {
  const card = cardEl(id);
  const d = state.drafts.get(id);
  if (!card || !d) return;
  const noTags = d.general && !d.drugs.length && !d.conditions.length;
  const button = card.querySelector('.doctor-send');
  if (button) button.disabled = card.dataset.voiceBusy === 'true' || d.sending || !d.answer.trim() || noTags;
}

function rerenderGeneral(id) {
  const q = state.open.find(x => x.id === id);
  const card = cardEl(id);
  if (!q || !card) return;
  const d = state.drafts.get(id);
  const panel = card.querySelector('.doctor-general');
  const fields = panel.querySelector('.doctor-general-fields');
  if (!d.general || !fields) {
    // Ticking or unticking: build or remove the whole panel.
    panel.innerHTML = generalHtml(q, d);
  } else {
    // Patch only the chips and the status line: inputs keep focus, caret and text.
    fields.querySelectorAll('.doctor-tag-row').forEach(row => {
      const kind = row.dataset.kind;
      row.querySelector('.doctor-chips').innerHTML = chipItemsHtml(kind, d[kind]);
    });
    fields.querySelector('.doctor-suggest-status').innerHTML = suggestStatusHtml(d);
  }
  updateSend(id);
}

function rerenderDraft(id) {
  const q = state.open.find(x => x.id === id);
  const card = cardEl(id);
  if (!q || !card) return;
  const d = state.drafts.get(id);
  card.querySelector('.clone-draft-slot').innerHTML = cloneDraftHtml(q, d);
  const box = card.querySelector('.doctor-answer-box');
  if (box.value !== d.answer) box.value = d.answer;
  updateSend(id);
}

async function requestDraft(q) {
  const d = draftFor(q);
  if (d.drafting) return;
  d.drafting = true;
  d.draftError = '';
  rerenderDraft(q.id);
  try {
    d.clone = await api(`/api/doctor/questions/${q.id}/draft`, {method: 'POST'});
    if (!d.answer.trim()) d.answer = d.clone.text;
  } catch (error) {
    d.draftError = error.message;
  }
  d.drafting = false;
  rerenderDraft(q.id);
}

async function useDraft(q) {
  const d = draftFor(q);
  if (!d.clone) return;
  if (d.answer.trim() && d.answer.trim() !== d.clone.text) {
    const ok = await confirmDialog({
      title: 'Replace your answer?',
      body: 'Your answer box will be replaced with the clone\'s draft.',
      confirmLabel: 'Replace',
    });
    if (!ok) return;
  }
  d.answer = d.clone.text;
  rerenderDraft(q.id);
  scheduleSuggest(q, d);
}

function rerenderCard(id) {
  const q = state.open.find(x => x.id === id);
  const card = cardEl(id);
  if (!q || !card) return;
  const tmp = document.createElement('div');
  tmp.innerHTML = openCardHtml(q);
  clearVoice(card);
  card.replaceWith(tmp.firstElementChild);
  updateSend(id);
  mountDoctorVoice(cardEl(id));
}

async function suggestTags(q, d) {
  if (!d.general || !d.answer.trim()) return;
  const question = d.savedQuestion.trim() || q.question;
  const key = question + '\n' + d.answer;
  if (key === d.lastSuggested) return;
  d.lastSuggested = key;
  const seq = ++d.seq;
  d.suggesting = true;
  d.tagError = '';
  rerenderGeneral(q.id);
  try {
    const out = await api('/api/doctor/suggest-tags', {
      method: 'POST', json: {question, answer: d.answer},
    });
    if (seq !== d.seq) return;
    d.drugs = uniqueTags([...d.drugs, ...(out.drug_tags || [])]);
    d.conditions = uniqueTags([...d.conditions, ...(out.condition_tags || [])]);
    d.suggesting = false;
  } catch (error) {
    if (seq !== d.seq) return;
    d.suggesting = false;
    d.lastSuggested = null;
    d.tagError = SUGGEST_FAILED;
  }
  rerenderGeneral(q.id);
}

function scheduleSuggest(q, d) {
  clearTimeout(d.timer);
  d.seq++;  // an in-flight suggestion for older text must not merge
  if (d.suggesting) d.lastSuggested = null;
  if (!d.general) return;
  d.timer = setTimeout(() => suggestTags(q, d), SUGGEST_DELAY_MS);
}

function addTag(id, kind, input) {
  const d = state.drafts.get(id);
  const value = input.value.trim();
  if (!value) return;
  const field = kind === 'drugs' ? 'drugs' : 'conditions';
  d[field] = uniqueTags([...d[field], value]);
  d.pending[field] = '';
  input.value = '';
  rerenderGeneral(id);
  const next = cardEl(id).querySelector(`.doctor-tag-input[data-kind="${kind}"]`);
  if (next) next.focus();
}

async function sendAnswer(q) {
  const d = state.drafts.get(q.id);
  if (cardEl(q.id)?.dataset.voiceBusy === 'true' || d.sending || !d.answer.trim()) return;
  clearTimeout(d.timer);
  d.seq++;  // drops an in-flight tag suggestion
  d.suggesting = false;
  d.sending = true;
  d.error = '';
  clearVoice(cardEl(q.id));
  cardEl(q.id).querySelector('.doctor-send').textContent = 'Sending...';
  updateSend(q.id);
  try {
    const out = await api(`/api/doctor/questions/${q.id}/answer`, {
      method: 'POST',
      json: {
        answer: d.answer.trim(),
        general: d.general,
        saved_question: d.general ? (d.savedQuestion.trim() || null) : null,
        drug_tags: d.general ? d.drugs : [],
        condition_tags: d.general ? d.conditions : [],
        draft_id: d.clone ? d.clone.id : null,
      },
    });
    state.open = state.open.filter(x => x.id !== q.id);
    state.answered = [{...q, ...out.question}, ...state.answered];
    state.drafts.delete(q.id);
    state.guidance = null;
    state.notice = 'Answer sent.';
    loadDraftStats();
    renderTabs();
    renderQuestions();
  } catch (error) {
    d.sending = false;
    d.error = error.message;
    if (error.status === 409) {
      state.drafts.delete(q.id);
      state.notice = '';
      await loadQueue();
      state.queueError = error.message;
      if (state.tab === 'questions') renderQuestions();
      return;
    }
    rerenderCard(q.id);
  }
}

function bindQuestions() {
  main.addEventListener('input', event => {
    const card = event.target.closest('.doctor-card');
    if (!card || !card.dataset.id) return;
    const id = Number(card.dataset.id);
    const q = state.open.find(x => x.id === id);
    const d = state.drafts.get(id);
    if (!q || !d) return;
    if (event.target.classList.contains('doctor-answer-box')) {
      d.answer = event.target.value;
      updateSend(id);
      scheduleSuggest(q, d);
    } else if (event.target.classList.contains('doctor-saved-question')) {
      d.savedQuestion = event.target.value;
    } else if (event.target.classList.contains('doctor-tag-input')) {
      d.pending[event.target.dataset.kind] = event.target.value;
    }
  });
  main.addEventListener('change', event => {
    if (!event.target.classList.contains('doctor-general-check')) return;
    const id = Number(event.target.closest('.doctor-card').dataset.id);
    const q = state.open.find(x => x.id === id);
    const d = state.drafts.get(id);
    d.general = event.target.checked;
    if (!d.general) { clearTimeout(d.timer); d.seq++; d.suggesting = false; }
    rerenderGeneral(id);
    if (d.general) suggestTags(q, d);
  });
  main.addEventListener('toggle', event => {
    if (event.target.classList.contains('doctor-answered')) state.answeredOpen = event.target.open;
    const key = event.target.dataset && event.target.dataset.key;
    if (key && event.target.classList.contains('doctor-context')) {
      if (event.target.open) state.openPanels.add(key); else state.openPanels.delete(key);
    }
  }, true);
}

// ---------- My guidance tab ----------

function statusMatches(e, status) {
  if (status === 'conflicts') return e.active && e.conflicts.length > 0;
  return status === 'active' ? e.active : !e.active;
}

function visibleEntries() {
  const {scope, status} = state.filters;
  return (state.guidance || []).filter(e =>
    (scope === 'all' || e.scope === scope) && statusMatches(e, status));
}

// Entries with an open conflict, as the Guidance tab badge counts them.
function conflictCount() {
  return (state.guidance || []).filter(e => e.active && e.conflicts.length).length;
}

const WARN_ICON = '<svg class="conflict-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4.5l8.5 15h-17z"/><path d="M12 10.25v4"/><path d="M12 17.25v.01"/></svg>';

function conflictHtml(c) {
  const o = c.other;
  const other = o.kind === 'label'
    ? `<p class="conflict-side-label">FDA drug label${o.title ? `: ${o.url ? `<a href="${esc(o.url)}" target="_blank" rel="noopener">${esc(o.title)}</a>` : esc(o.title)}` : ''}</p>
       <blockquote class="conflict-quote">${esc(o.quote)}</blockquote>`
    : `<p class="conflict-side-label">Your earlier advice, on &ldquo;${esc(o.question)}&rdquo;</p>
       <blockquote class="conflict-quote">${esc(o.quote)}</blockquote>`;
  return `
    <li class="conflict" data-cid="${c.id}">
      <p class="conflict-head">${WARN_ICON}<span>Possible conflict</span></p>
      <p class="conflict-why">${esc(c.explanation)}</p>
      <div class="conflict-sides">
        <div class="conflict-side"><p class="conflict-side-label">This advice</p>
          <blockquote class="conflict-quote">${esc(c.rule.quote)}</blockquote></div>
        <div class="conflict-side">${other}</div>
      </div>
      <div class="form-actions conflict-actions">
        <button type="button" class="btn btn-secondary btn-small" data-action="dismiss-conflict" data-cid="${c.id}">Not a conflict</button>
      </div>
    </li>`;
}

function rulesHtml(e) {
  if (!e.rules.length) return '';
  const open = state.rulesOpen.has(e.id) ? ' open' : '';
  return `
    <details class="doctor-rules"${open} data-gid="${e.id}">
      <summary class="doctor-rules-summary">How your clone reads this <span class="count">${e.rules.length}</span></summary>
      <ol class="rule-list">${e.rules.map(r => `
        <li class="rule"><p class="rule-statement">${esc(r.statement)}</p>
          ${r.quote !== r.statement ? `<p class="rule-quote"><span class="rule-quote-label">Your words</span> ${esc(r.quote)}</p>` : ''}</li>`).join('')}
      </ol>
    </details>`;
}

function tagsLine(e) {
  const chips = [...e.drug_tags.map(t => `<li class="doctor-chip">${esc(t)}</li>`),
    ...e.condition_tags.map(t => `<li class="doctor-chip doctor-chip-cond">${esc(t)}</li>`)];
  return chips.length ? `<ul class="doctor-chips">${chips.join('')}</ul>` : '';
}

function editHtml(e, ed) {
  return `
    <div class="field"><label for="gq-${e.id}">Question</label>
      <textarea id="gq-${e.id}" class="doctor-edit-question doctor-short" maxlength="4000">${esc(ed.question)}</textarea></div>
    <div class="field"><label for="ga-${e.id}">Answer</label>
      <textarea id="ga-${e.id}" class="doctor-edit-answer" maxlength="4000">${esc(ed.answer)}</textarea></div>
    <div class="field"><label for="gs-${e.id}">Scope</label>
      <select id="gs-${e.id}" class="doctor-edit-scope">
        <option value="general" ${ed.scope === 'general' ? 'selected' : ''}>General (all patients)</option>
        <option value="patient" ${ed.scope === 'patient' ? 'selected' : ''}${e.patient === null ? ' disabled' : ''}>Patient-specific</option>
      </select></div>
    ${ed.scope === 'general' ? chipsHtml('drugs', ed.drugs, 'Drugs', ed.pending.drugs) + chipsHtml('conditions', ed.conditions, 'Conditions', ed.pending.conditions) : ''}`;
}

function rowHtml(e) {
  const ed = state.editing.get(e.id);
  const busy = state.busyRows.has(e.id);
  const err = state.rowErrors.get(e.id);
  const labels = [];
  if (e.scope === 'patient' && e.stale) labels.push('for an earlier document');
  if (!e.searchable) labels.push('not searchable yet');
  else if (e.rules_status === 'failed') labels.push('read as a whole');
  const scope = e.scope === 'general' ? 'General' : `Patient: ${esc(e.patient || 'unknown')}`;
  const meta = [scope, formatDate(e.created_at), `cited ${e.cited_count} ${e.cited_count === 1 ? 'time' : 'times'}`]
    .filter(Boolean).join(' &middot; ');
  return `
    <li class="card doctor-card doctor-row" data-gid="${e.id}">
      <p class="doctor-meta">${meta}${labels.map(l => ` <span class="badge">${esc(l)}</span>`).join('')}</p>
      ${ed ? editHtml(e, ed) : `
        <p class="doctor-question">${esc(e.question)}</p>
        <p class="doctor-answer-text">${esc(e.answer)}</p>
        ${tagsLine(e)}
        ${e.active && e.conflicts.length ? `<ul class="conflict-list">${e.conflicts.map(conflictHtml).join('')}</ul>` : ''}
        ${rulesHtml(e)}`}
      ${err ? `<p class="form-error" role="alert">${esc(err)}</p>` : ''}
      <div class="form-actions doctor-row-actions">
        ${ed ? `
          <button type="button" class="btn btn-secondary btn-small" data-action="cancel-edit" ${busy ? 'disabled' : ''}>Cancel</button>
          <button type="button" class="btn btn-primary btn-small" data-action="save-edit" ${busy ? 'disabled' : ''}>${busy ? 'Saving...' : 'Save'}</button>` : `
          ${e.active && (!e.searchable || e.rules_status === 'failed') ? `<button type="button" class="btn btn-secondary btn-small" data-action="embed" ${busy ? 'disabled' : ''}>${!e.searchable ? 'Retry embedding' : 'Retry reading rules'}</button>` : ''}
          <button type="button" class="btn btn-secondary btn-small" data-action="edit" ${busy ? 'disabled' : ''}>Edit</button>
          <button type="button" class="btn ${e.active ? 'btn-secondary' : 'btn-primary'} btn-small" data-action="toggle-active" ${busy ? 'disabled' : ''}>${e.active ? 'Retire' : 'Re-activate'}</button>`}
      </div>
    </li>`;
}

function renderGuidance() {
  clearVoice(main);
  let body;
  if (state.guidanceError && !state.guidance) {
    body = `<p class="form-error" role="alert">${esc(state.guidanceError)}</p>`;
  } else if (!state.guidance) {
    body = '<p class="busy-note" role="status"><span class="spinner" aria-hidden="true"></span>Loading...</p>';
  } else {
    const rows = visibleEntries();
    body = rows.length
      ? `<ul class="item-list doctor-list">${rows.map(rowHtml).join('')}</ul>`
      : '<div class="card empty-inline"><p class="section-title">Nothing here</p><p class="section-text">No guidance matches these filters.</p></div>';
  }
  const f = state.filters;
  const opt = (v, cur, label) => `<option value="${v}" ${cur === v ? 'selected' : ''}>${label}</option>`;
  main.innerHTML = `
    <div class="split">
    <div class="split-aside">
    <div class="page-head"><p class="eyebrow">Your AI clone</p><h1 class="page-title">My <em>guidance</em></h1>
      <p class="lead">Editing changes your AI clone's future answers only. Past chat messages keep the text they showed.</p></div>
    <div class="doctor-filters">
      <div class="field"><label for="fScope">Scope</label>
        <select id="fScope">${opt('all', f.scope, 'All')}${opt('general', f.scope, 'General')}${opt('patient', f.scope, 'Patient-specific')}</select></div>
      <div class="field"><label for="fStatus">Status</label>
        <select id="fStatus">${opt('active', f.status, 'Active')}${opt('conflicts', f.status, 'With conflicts')}${opt('retired', f.status, 'Retired')}</select></div>
    </div>
    ${conflictCount() ? `<button type="button" class="conflict-summary" data-action="show-conflicts">${WARN_ICON}<span><strong>${conflictCount()}</strong> ${conflictCount() === 1 ? 'entry has' : 'entries have'} a possible conflict</span></button>` : ''}
    </div>
    <div class="split-main">
    ${state.guidanceError && state.guidance ? `<p class="form-error" role="alert">${esc(state.guidanceError)}</p>` : ''}
    ${body}
    </div>
    </div>`;
}

async function loadGuidance() {
  try {
    state.guidance = (await api('/api/doctor/guidance')).entries;
    state.guidanceError = '';
  } catch (error) {
    state.guidanceError = error.message;
  }
  renderTabs();
  if (state.tab === 'guidance') renderGuidance();
}

async function patchRow(id, json, {optimisticClose} = {}) {
  state.busyRows.add(id);
  state.rowErrors.delete(id);
  renderGuidance();
  try {
    const updated = await api(`/api/doctor/guidance/${id}`, {method: 'PATCH', json});
    state.guidance = state.guidance.map(e => e.id === id ? updated : e);
    if (optimisticClose) state.editing.delete(id);
  } catch (error) {
    state.rowErrors.set(id, error.message);
  }
  state.busyRows.delete(id);
  renderGuidance();
}

async function rowAction(action, id, cid) {
  const entry = state.guidance.find(e => e.id === id);
  const ed = state.editing.get(id);
  if (action === 'edit') {
    state.editing.set(id, {
      question: entry.question, answer: entry.answer, scope: entry.scope,
      drugs: [...entry.drug_tags], conditions: [...entry.condition_tags],
      pending: {drugs: '', conditions: ''},
    });
    state.rowErrors.delete(id);
    renderGuidance();
  } else if (action === 'cancel-edit') {
    state.editing.delete(id);
    state.rowErrors.delete(id);
    renderGuidance();
  } else if (action === 'save-edit') {
    const json = {question: ed.question.trim(), answer: ed.answer.trim(), scope: ed.scope};
    if (ed.scope === 'general') { json.drug_tags = ed.drugs; json.condition_tags = ed.conditions; }
    await patchRow(id, json, {optimisticClose: true});
  } else if (action === 'toggle-active') {
    if (entry.active) {
      const ok = await confirmDialog({
        title: 'Retire this guidance?',
        body: 'Your AI clone will stop using it. You can re-activate it later.',
        confirmLabel: 'Retire', danger: true,
      });
      if (!ok) return;
    }
    await patchRow(id, {active: !entry.active});
  } else if (action === 'dismiss-conflict') {
    await dismissConflict(id, Number(cid));
  } else if (action === 'embed') {
    state.busyRows.add(id);
    renderGuidance();
    try {
      const updated = await api(`/api/doctor/guidance/${id}/embed`, {method: 'POST'});
      state.guidance = state.guidance.map(e => e.id === id ? updated : e);
      state.rowErrors.delete(id);
    } catch (error) {
      state.rowErrors.set(id, error.message);
    }
    state.busyRows.delete(id);
    renderGuidance();
  }
}

async function dismissConflict(id, cid) {
  state.busyRows.add(id);
  renderGuidance();
  try {
    await api(`/api/doctor/conflicts/${cid}/dismiss`, {method: 'POST'});
    // A conflict shows on both of its entries: drop it everywhere.
    state.guidance = state.guidance.map(e => ({...e, conflicts: e.conflicts.filter(c => c.id !== cid)}));
    state.rowErrors.delete(id);
  } catch (error) {
    state.rowErrors.set(id, error.message);
  }
  state.busyRows.delete(id);
  renderTabs();
  renderGuidance();
}

function bindGuidance() {
  main.addEventListener('toggle', event => {
    const details = event.target;
    if (!details.classList || !details.classList.contains('doctor-rules')) return;
    const id = Number(details.dataset.gid);
    if (details.open) state.rulesOpen.add(id); else state.rulesOpen.delete(id);
  }, true);
  main.addEventListener('change', event => {
    if (event.target.id === 'fScope') { state.filters.scope = event.target.value; renderGuidance(); }
    else if (event.target.id === 'fStatus') { state.filters.status = event.target.value; renderGuidance(); }
    else if (event.target.classList.contains('doctor-edit-scope')) {
      const id = Number(event.target.closest('.doctor-row').dataset.gid);
      state.editing.get(id).scope = event.target.value;
      renderGuidance();
    }
  });
  main.addEventListener('input', event => {
    const row = event.target.closest('.doctor-row');
    if (!row) return;
    const ed = state.editing.get(Number(row.dataset.gid));
    if (!ed) return;
    if (event.target.classList.contains('doctor-edit-answer')) ed.answer = event.target.value;
    else if (event.target.classList.contains('doctor-edit-question')) ed.question = event.target.value;
    else if (event.target.classList.contains('doctor-tag-input')) ed.pending[event.target.dataset.kind] = event.target.value;
  });
}

function addEditTag(id, kind, input) {
  const ed = state.editing.get(id);
  const value = input.value.trim();
  if (!value) return;
  const field = kind === 'drugs' ? 'drugs' : 'conditions';
  ed[field] = uniqueTags([...ed[field], value]);
  ed.pending[field] = '';
  renderGuidance();
  const next = main.querySelector(`.doctor-row[data-gid="${id}"] .doctor-tag-input[data-kind="${kind}"]`);
  if (next) next.focus();
}

// ---------- shared click and key handling ----------

function bindShared() {
  main.addEventListener('click', event => {
    const button = event.target.closest('button[data-action]');
    if (!button) return;
    const action = button.dataset.action;
    const kind = button.dataset.kind;
    const qcard = button.closest('.doctor-card[data-id]');
    const row = button.closest('.doctor-row');
    if (qcard) {
      const id = Number(qcard.dataset.id);
      const d = state.drafts.get(id);
      if (action === 'send') sendAnswer(state.open.find(x => x.id === id));
      else if (action === 'draft') requestDraft(state.open.find(x => x.id === id));
      else if (action === 'use-draft') useDraft(state.open.find(x => x.id === id));
      else if (action === 'remove-tag') {
        const field = kind === 'drugs' ? 'drugs' : 'conditions';
        d[field].splice(Number(button.dataset.index), 1);
        rerenderGeneral(id);
      } else if (action === 'add-tag') {
        addTag(id, kind, qcard.querySelector(`.doctor-tag-input[data-kind="${kind}"]`));
      }
    } else if (row) {
      const id = Number(row.dataset.gid);
      const ed = state.editing.get(id);
      if (action === 'remove-tag') {
        ed[kind === 'drugs' ? 'drugs' : 'conditions'].splice(Number(button.dataset.index), 1);
        renderGuidance();
      } else if (action === 'add-tag') {
        addEditTag(id, kind, row.querySelector(`.doctor-tag-input[data-kind="${kind}"]`));
      } else {
        rowAction(action, id, button.dataset.cid);
      }
    } else if (action === 'show-conflicts') {
      state.filters.status = 'conflicts';
      renderGuidance();
    }
  });
  main.addEventListener('keydown', event => {
    if (!event.target.classList.contains('doctor-tag-input')) return;
    if (event.key !== 'Enter' && event.key !== ',') return;
    event.preventDefault();
    const kind = event.target.dataset.kind;
    const qcard = event.target.closest('.doctor-card[data-id]');
    const row = event.target.closest('.doctor-row');
    if (qcard) addTag(Number(qcard.dataset.id), kind, event.target);
    else if (row) addEditTag(Number(row.dataset.gid), kind, event.target);
  });
}

// ---------- shell ----------

function setBadge(tab, count, one, many) {
  let badge = tab.querySelector('.tab-badge');
  if (!count) {
    if (badge) badge.remove();
    return;
  }
  if (!badge) {
    badge = document.createElement('span');
    badge.className = 'tab-badge';
    badge.setAttribute('role', 'img');
    tab.append(badge);
  }
  badge.textContent = String(count);
  badge.setAttribute('aria-label', count === 1 ? `1 ${one}` : `${count} ${many}`);
}

function renderTabs() {
  setBadge(document.getElementById('tabQuestions'), state.open.length, 'open question', 'open questions');
  setBadge(document.getElementById('tabLabels'), state.labelCount, 'label to review', 'labels to review');
  setBadge(document.getElementById('tabGuidance'), conflictCount(), 'entry with a possible conflict', 'entries with a possible conflict');
  document.querySelectorAll('.tab').forEach(tab => {
    if (tab.dataset.tab === state.tab) tab.setAttribute('aria-current', 'page');
    else tab.removeAttribute('aria-current');
  });
  syncTabs(document.querySelector('.tabs'));
}

function setLabelCount(count) {
  state.labelCount = count;
  renderTabs();
}

async function loadDraftStats() {
  try {
    state.draftStats = await api('/api/doctor/draft-stats');
  } catch (error) {
    return;  // the numbers are a nicety; the queue shows its own errors
  }
  if (state.tab === 'questions') {
    const aside = main.querySelector('.split-aside');
    const old = aside && aside.querySelector('.draft-stats');
    const html = draftStatsHtml();
    if (old) old.outerHTML = html;
    else if (aside && html) aside.insertAdjacentHTML('beforeend', html);
  }
}

async function loadQueue() {
  try {
    const data = await api('/api/doctor/questions');
    state.open = data.open;
    state.answered = data.answered;
    state.queueError = '';
  } catch (error) {
    state.queueError = error.message;
  }
  renderTabs();
  if (state.tab === 'questions') renderQuestions();
}

function route() {
  const previous = state.tab;
  state.tab = tabFromHash();
  if (previous !== state.tab || !main.dataset.entered) {
    main.dataset.entered = 'true';
    enter(main);
  }
  renderTabs();
  if (state.tab !== 'labels') hideLabels(main);
  if (state.tab === 'labels') {
    showLabels(main, {onCount: count => { state.labelCountFromTab = true; setLabelCount(count); }});
  } else if (state.tab === 'guidance') {
    renderGuidance();
    loadGuidance();
  } else {
    state.notice = '';
    renderQuestions();
  }
}

async function boot() {
  document.getElementById('logoutBtn').addEventListener('click', () => {
    clearSession();
    window.location.href = '/';
  });
  let me;
  try {
    me = await api('/api/me');
  } catch (error) {
    if (error.status === 401) return;
    main.innerHTML = `<p class="form-error" role="alert">${esc(error.message)}</p>`;
    return;
  }
  if (me.role !== 'doctor') { window.location.replace('/app'); return; }
  const name = me.display_name || me.username;
  document.getElementById('brandName').textContent = name;
  document.getElementById('avatar').textContent = initials(name);
  const specialty = document.getElementById('brandSpecialty');
  specialty.textContent = me.specialty || '';
  specialty.hidden = !me.specialty;
  bindQuestions();
  bindGuidance();
  bindShared();
  window.addEventListener('hashchange', route);
  state.tab = tabFromHash();
  main.innerHTML = '<p class="busy-note" role="status"><span class="spinner" aria-hidden="true"></span>Loading...</p>';
  // The badge only (the tab shows its own errors); a count the tab reported wins.
  countPendingLabels().then(count => { if (!state.labelCountFromTab) setLabelCount(count); }, () => {});
  loadGuidance();  // for the conflict badge; the tab reloads it when opened
  loadDraftStats();
  await loadQueue();
  route();
}

boot();
