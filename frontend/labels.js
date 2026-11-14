// Doctor page "Drug labels" tab: FDA labels fetched for the medications on
// patients' confirmed discharge documents. A doctor approves a label (it joins
// the education library), rejects it, picks another candidate label, or retries
// a failed lookup. doctor.js only routes here and shows the badge.
import {api, escapeHtml} from '/static/api.js';

const esc = escapeHtml;
const POLL_MS = 4000;  // while a lookup runs in the background
const EMPTY = 'No drug labels waiting.';
const NO_LABEL = 'DailyMed has no label for this ingredient.';
const NOT_MATCHED = 'No drug in RxNorm matched this medication.';
const BUSY_TEXT = {approve: 'Approving...', reject: 'Rejecting...', pick: 'Fetching...', retry: 'Trying...'};

// DailyMed titles read "<NAME> (<INGREDIENTS>) <DOSE FORM> [<LABELER>]". The
// dose form starts at the first dosage form word after the names.
const FORM_WORDS = /\b(TABLETS?|CAPSULES?|CAPLETS?|SUSPENSION|SOLUTION|INJECTION|INJECTABLE|CREAM|OINTMENT|GEL|PATCH|KIT|POWDER|LIQUID|SPRAY|LOTION|SYRUP|ELIXIR|GRANULES?|CLOTH|LOZENGE|SUPPOSITORY|AEROSOL|FILM|INHALANT|CONCENTRATE|EMULSION|PELLET|IMPLANT|DROPS?|SHAMPOO|FOAM|PASTE|GUM|INSERT|RING|SWAB|WAFER|TROCHE|STRIP)\b/;

const state = {
  data: null,           // {labels, unresolved} from the server
  error: '',            // the list could not be loaded
  busy: new Map(),      // card key -> running action
  errors: new Map(),    // card key -> message
  picks: new Map(),     // label id -> chosen setid in its select
  openText: new Set(),  // label ids with the label text expanded
  seq: 0,               // drops a list response older than the latest change
};
const views = new WeakMap();  // main element -> {onCount, poll}
let mounted = null;           // the main element showing this tab

const labelKey = id => 'label:' + id;
const drugKey = id => 'drug:' + id;

// ---------- formatting ----------

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

// "2026-11-09" -> "Nov 9, 2026", read as a calendar date (no time zone shift).
function formatDay(iso) {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(iso || ''));
  if (!m) return '';
  return `${MONTHS[Number(m[2]) - 1]} ${Number(m[3])}, ${m[1]}`;
}

// Server timestamps are UTC without an offset.
function formatDate(iso) {
  if (!iso) return '';
  const stamp = /[zZ]|[+-]\d\d:?\d\d$/.test(iso) ? iso : iso + 'Z';
  const d = new Date(stamp);
  if (Number.isNaN(d.getTime())) return '';
  return d.toLocaleDateString('en-US', {month: 'short', day: 'numeric'});
}

const sentenceCase = text => text ? text.charAt(0).toUpperCase() + text.slice(1).toLowerCase() : '';

// The title without its trailing "[LABELER]" (the labeler has its own line).
function productName(title) {
  return String(title || '').replace(/\s*\[[^\]]*\]\s*$/, '').trim();
}

export function doseForm(title) {
  let product = productName(title);
  const paren = product.lastIndexOf(')');
  if (paren !== -1) product = product.slice(paren + 1).trim();
  const match = FORM_WORDS.exec(product.toUpperCase());
  if (match) return sentenceCase(product.slice(match.index).trim());
  return paren !== -1 ? sentenceCase(product) : '';
}

// ---------- rendering ----------

function statusBadge(text, tone) {
  return `<span class="badge label-badge label-badge-${tone}">${esc(text)}</span>`;
}

function sourcesHtml(sources) {
  if (!sources || !sources.length) return '';
  return `<ul class="label-sources">${sources.map(s =>
    `<li><span class="label-sources-from">From:</span> ${esc(s.medication)} <span class="label-sources-who">(${esc(s.username)})</span></li>`).join('')}</ul>`;
}

function factsHtml(label) {
  const facts = [
    ['Labeler', label.labeler],
    ['Dose form', doseForm(label.title)],
    ['Published', formatDay(label.published)],
  ].filter(([, value]) => value);
  if (!facts.length) return '';
  return `<dl class="label-facts">${facts.map(([name, value]) =>
    `<div><dt>${name}</dt><dd>${esc(value)}</dd></div>`).join('')}</dl>`;
}

function textHtml(label) {
  if (!label.text && !label.url) return '';
  return `
    <div class="doctor-contexts label-links">
      ${label.text ? `<details class="doctor-context label-text" data-label-text="${label.id}" ${state.openText.has(label.id) ? 'open' : ''}>
        <summary>Label text</summary>
        <div class="doctor-context-body label-text-body" tabindex="0">${esc(label.text)}</div>
      </details>` : ''}
      ${label.url ? `<a class="label-dailymed" href="${esc(label.url)}" target="_blank" rel="noopener noreferrer">Open on DailyMed<span class="visually-hidden"> (new tab)</span></a>` : ''}
    </div>`;
}

// The date comes first so a long product name never cuts it off. The labeler
// goes last: candidates often share a date and a product name.
function optionText(candidate) {
  const day = formatDay(candidate.published);
  const labeler = candidate.labeler ? ` (${candidate.labeler})` : '';
  return (day ? day + ' - ' : '') + productName(candidate.title) + labeler;
}

function pickHtml(label, busy) {
  const options = (label.candidates || []).filter(c => c.setid !== label.setid);
  if (!options.length) return '';
  const chosen = state.picks.get(label.id);
  const selected = options.some(c => c.setid === chosen) ? chosen : options[0].setid;
  return `
    <div class="label-pick">
      <label for="pick-${label.id}">Pick another label</label>
      <div class="label-pick-row">
        <select id="pick-${label.id}" class="label-pick-select" data-label-pick="${label.id}" ${busy ? 'disabled' : ''}>
          ${options.map(c => `<option value="${esc(c.setid)}" ${c.setid === selected ? 'selected' : ''}>${esc(optionText(c))}</option>`).join('')}
        </select>
        <button type="button" class="btn btn-secondary" data-label-action="pick" ${busy ? 'disabled' : ''}>${busy === 'pick' ? BUSY_TEXT.pick : 'Use'}</button>
      </div>
    </div>`;
}

function errorHtml(key) {
  const message = state.errors.get(key);
  return message ? `<p class="form-error" role="alert">${esc(message)}</p>` : '';
}

function button(action, label, kind, busy) {
  return `<button type="button" class="btn ${kind}" data-label-action="${action}" ${busy ? 'disabled' : ''}>${busy === action ? BUSY_TEXT[action] : label}</button>`;
}

function headHtml(name, badge) {
  return `<div class="label-head"><h2 class="label-name">${esc(name)}</h2>${badge}</div>`;
}

const ingredientName = label => sentenceCase(label.ingredient || '') || 'Unknown ingredient';

function pendingCardHtml(label) {
  const key = labelKey(label.id);
  const busy = state.busy.get(key);
  return `
    <li class="card doctor-card label-card" data-label-id="${label.id}" ${busy ? 'aria-busy="true"' : ''}>
      ${headHtml(ingredientName(label), statusBadge('Needs review', 'pending'))}
      ${label.title ? `<p class="label-product">${esc(productName(label.title))}</p>` : ''}
      ${factsHtml(label)}
      ${sourcesHtml(label.sources)}
      ${textHtml(label)}
      ${pickHtml(label, busy)}
      ${errorHtml(key)}
      <div class="form-actions">
        ${button('reject', 'Reject', 'btn-secondary', busy)}
        ${button('approve', 'Approve', 'btn-primary', busy)}
      </div>
    </li>`;
}

function fetchingCardHtml(label) {
  return `
    <li class="card doctor-card label-card" data-label-id="${label.id}" aria-busy="true">
      ${headHtml(ingredientName(label), statusBadge('Fetching', 'neutral'))}
      ${sourcesHtml(label.sources)}
      <p class="busy-note" role="status"><span class="spinner" aria-hidden="true"></span>Fetching the label from DailyMed...</p>
    </li>`;
}

function failedLabelCardHtml(label) {
  const key = labelKey(label.id);
  const busy = state.busy.get(key);
  const failed = label.status === 'failed';
  return `
    <li class="card doctor-card label-card" data-label-id="${label.id}" ${busy ? 'aria-busy="true"' : ''}>
      ${headHtml(ingredientName(label), statusBadge(failed ? 'Lookup failed' : 'No label found', 'problem'))}
      ${label.title ? `<p class="label-product">${esc(productName(label.title))}</p>` : ''}
      ${sourcesHtml(label.sources)}
      <p class="label-problem">${esc(failed ? (label.error || 'The lookup failed.') : NO_LABEL)}</p>
      ${pickHtml(label, busy)}
      ${errorHtml(key)}
      <div class="form-actions">${button('retry', 'Try again', 'btn-secondary', busy)}</div>
    </li>`;
}

const drugName = drug => drug.name || 'Medication';

function lookingUpCardHtml(drug) {
  return `
    <li class="card doctor-card label-card" data-drug-id="${drug.id}" aria-busy="true">
      ${headHtml(drugName(drug), statusBadge('Looking up', 'neutral'))}
      ${sourcesHtml([drug])}
      <p class="busy-note" role="status"><span class="spinner" aria-hidden="true"></span>Looking up...</p>
    </li>`;
}

function unresolvedCardHtml(drug) {
  if (drug.status === 'fetching') return lookingUpCardHtml(drug);
  const key = drugKey(drug.id);
  const busy = state.busy.get(key);
  const failed = drug.status === 'failed';
  return `
    <li class="card doctor-card label-card" data-drug-id="${drug.id}" ${busy ? 'aria-busy="true"' : ''}>
      ${headHtml(drugName(drug), statusBadge(failed ? 'Lookup failed' : 'Not matched', 'problem'))}
      ${sourcesHtml([drug])}
      <p class="label-problem">${esc(failed ? (drug.error || 'The lookup failed.') : NOT_MATCHED)}</p>
      ${errorHtml(key)}
      <div class="form-actions">${button('retry', 'Try again', 'btn-secondary', busy)}</div>
    </li>`;
}

function reviewedCardHtml(label) {
  const key = labelKey(label.id);
  const busy = state.busy.get(key);
  const approved = label.status === 'approved';
  const who = label.reviewed_by_name || 'A doctor';
  const when = formatDate(label.reviewed_at);
  return `
    <li class="card doctor-card label-card is-reviewed" data-label-id="${label.id}" ${busy ? 'aria-busy="true"' : ''}>
      ${headHtml(ingredientName(label), statusBadge(approved ? 'Approved' : 'Rejected', approved ? 'approved' : 'rejected'))}
      ${label.title ? `<p class="label-product">${esc(productName(label.title))}</p>` : ''}
      ${factsHtml(label)}
      <p class="doctor-meta"><span class="doctor-meta-label">${esc(who)}</span> ${approved ? 'approved' : 'rejected'} this label${when ? ' on ' + esc(when) : ''}.</p>
      ${approved ? textHtml({id: label.id, url: label.url}) : pickHtml(label, busy)}
      ${errorHtml(key)}
    </li>`;
}

function listHtml(data) {
  const pending = data.labels.filter(l => l.status === 'pending' || l.status === 'fetching');
  const problems = data.labels.filter(l => l.status === 'failed' || l.status === 'no_label');
  const reviewed = data.labels.filter(l => l.status === 'approved' || l.status === 'rejected')
    .sort((a, b) => String(b.reviewed_at || '').localeCompare(String(a.reviewed_at || '')));  // newest first
  const waiting = [
    ...pending.map(l => l.status === 'pending' ? pendingCardHtml(l) : fetchingCardHtml(l)),
    ...problems.map(failedLabelCardHtml),
    ...data.unresolved.map(unresolvedCardHtml),
  ];
  const waitingHtml = waiting.length
    ? `<ul class="item-list doctor-list label-list">${waiting.join('')}</ul>`
    : `<div class="card empty-inline"><p class="label-empty">${EMPTY}</p></div>`;
  const reviewedHtml = reviewed.length ? `
    <section class="label-reviewed" aria-labelledby="labelsReviewed">
      <h2 class="group-title" id="labelsReviewed">Reviewed <span class="count">${reviewed.length}</span></h2>
      <ul class="item-list doctor-list label-list">${reviewed.map(reviewedCardHtml).join('')}</ul>
    </section>` : '';
  return waitingHtml + reviewedHtml;
}

function render(main) {
  if (mounted !== main || !main.querySelector('.labels-view')) return;
  let body;
  if (state.data) {
    body = (state.error ? `<p class="form-error" role="alert">${esc(state.error)}</p>` : '') + listHtml(state.data);
  } else if (state.error) {
    body = `<p class="form-error" role="alert">${esc(state.error)}</p>`;
  } else {
    body = '<p class="busy-note" role="status"><span class="spinner" aria-hidden="true"></span>Loading...</p>';
  }
  main.querySelector('.labels-body').innerHTML = body;
}

function shellHtml() {
  return `
    <div class="split labels-view">
    <div class="split-aside">
    <div class="page-head"><p class="eyebrow">Education library</p><h1 class="page-title">Drug <em>labels</em></h1>
      <p class="lead">FDA labels for the medications on your patients' discharge documents. Once you approve a label, your AI clone can cite it to patients who take that drug.</p></div>
    </div>
    <div class="split-main labels-body"></div>
    </div>`;
}

// ---------- data ----------

function pendingCount(data) {
  return data ? data.labels.filter(l => l.status === 'pending').length : 0;
}

function setData(main, data) {
  state.data = data;
  state.error = '';
  // Forget per-card state of cards that left the list.
  const known = new Set(data.labels.map(l => l.id));
  const keys = new Set([...data.labels.map(l => labelKey(l.id)), ...data.unresolved.map(d => drugKey(d.id))]);
  for (const id of state.openText) if (!known.has(id)) state.openText.delete(id);
  for (const id of state.picks.keys()) if (!known.has(id)) state.picks.delete(id);
  for (const key of state.errors.keys()) if (!keys.has(key)) state.errors.delete(key);
  const view = views.get(main);
  if (view) view.onCount(pendingCount(data));
  render(main);
  schedulePoll(main);
}

function schedulePoll(main) {
  const view = views.get(main);
  if (!view) return;
  clearTimeout(view.poll);
  const running = state.data && (state.data.labels.some(l => l.status === 'fetching')
    || state.data.unresolved.some(d => d.status === 'fetching'));
  if (running && mounted === main) view.poll = setTimeout(() => load(main), POLL_MS);
}

async function load(main) {
  const seq = ++state.seq;
  try {
    const data = await api('/api/doctor/labels');
    if (seq !== state.seq) return;
    setData(main, data);
  } catch (error) {
    if (seq !== state.seq) return;
    state.error = error.message;
    render(main);
  }
}

// Replaces one label with the view an action returned.
function replaceLabel(main, view) {
  const labels = state.data.labels.map(l => l.id === view.id ? view : l);
  setData(main, {...state.data, labels});
}

async function runAction(main, key, action, request) {
  if (state.busy.has(key)) return;
  state.busy.set(key, action);
  state.errors.delete(key);
  state.seq++;  // a list response that started before this action is stale
  render(main);
  try {
    const out = await request();
    state.busy.delete(key);
    if (out.labels) setData(main, out); else replaceLabel(main, out);
  } catch (error) {
    state.busy.delete(key);
    state.errors.set(key, error.message);
    if (error.status === 409 || error.status === 404) {
      await load(main);  // someone else changed it: show where it is now
    } else {
      render(main);
    }
  }
}

function labelAction(main, id, action) {
  const key = labelKey(id);
  const url = `/api/doctor/labels/${id}/`;
  if (action === 'approve' || action === 'reject') {
    // Name the label text on screen, so a review never applies to text that
    // replaced it meanwhile (the server answers 409 and the list reloads).
    const label = state.data && state.data.labels.find(l => l.id === id);
    if (!label) return;
    runAction(main, key, action, () => api(url + action, {method: 'POST', json: {setid: label.setid}}));
  } else if (action === 'retry') {
    runAction(main, key, action, () => api(url + action, {method: 'POST'}));
  } else if (action === 'pick') {
    const select = main.querySelector(`[data-label-pick="${id}"]`);
    if (!select || !select.value) return;
    const setid = select.value;
    runAction(main, key, action, async () => {
      const view = await api(url + 'candidate', {method: 'POST', json: {setid}});
      state.picks.delete(id);
      return view;
    });
  }
}

function bind(main) {
  main.addEventListener('click', event => {
    if (mounted !== main) return;
    const btn = event.target.closest('button[data-label-action]');
    if (!btn) return;
    const card = btn.closest('.label-card');
    if (!card) return;
    if (card.dataset.labelId) {
      labelAction(main, Number(card.dataset.labelId), btn.dataset.labelAction);
    } else if (card.dataset.drugId) {
      const id = Number(card.dataset.drugId);
      runAction(main, drugKey(id), 'retry', () => api(`/api/doctor/document-drugs/${id}/retry`, {method: 'POST'}));
    }
  });
  main.addEventListener('change', event => {
    const id = event.target.dataset && event.target.dataset.labelPick;
    if (id) state.picks.set(Number(id), event.target.value);
  });
  main.addEventListener('toggle', event => {
    const id = event.target.dataset && event.target.dataset.labelText;
    if (!id) return;
    if (event.target.open) state.openText.add(Number(id)); else state.openText.delete(Number(id));
  }, true);
}

// ---------- public ----------

// Renders the tab into `main` and loads the list. onCount(n) receives the
// number of labels waiting for review whenever the list changes.
export function showLabels(main, {onCount = () => {}} = {}) {
  if (!views.has(main)) {
    views.set(main, {onCount, poll: null});
    bind(main);
  }
  views.get(main).onCount = onCount;
  mounted = main;
  main.innerHTML = shellHtml();
  render(main);
  load(main);
}

// Call when another tab replaces this one, so a late response or the
// background poll does not draw over it.
export function hideLabels(main) {
  if (mounted === main) mounted = null;
  const view = views.get(main);
  if (view) clearTimeout(view.poll);
}

// The number of labels waiting for review, for the tab badge on page load.
export async function countPendingLabels() {
  return pendingCount(await api('/api/doctor/labels'));
}
