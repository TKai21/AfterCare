// The document pane beside the instructions: the uploaded PDF itself (rendered
// with pdf.js) or, for pasted text, TXT files and documents uploaded before PDFs
// were kept, the document text. Either way the selected instruction's quote is
// highlighted where it appears.
import {clearSession, escapeHtml, getToken} from '/static/api.js';

const PDFJS = '/static/vendor/pdfjs/';
const esc = escapeHtml;

export function docPaneHtml(doc) {
  return `
    <aside class="doc-pane" aria-labelledby="docTitle">
      <div class="doc-head">
        <h2 class="doc-title" id="docTitle">Your discharge document</h2>
        ${doc.has_pdf ? '<a class="doc-open btn-link" id="docOpen" target="_blank" rel="noopener" hidden>Open PDF</a>' : ''}
      </div>
      <div class="doc-body${doc.has_pdf ? ' is-pdf' : ''}" id="docBody" tabindex="0"
           aria-label="${doc.has_pdf ? 'Document pages' : 'Document text'}"></div>
    </aside>`;
}

// Fills the pane rendered by docPaneHtml(). Returns {select(quote, {scroll}), first()}:
// select() highlights `quote` (null clears it); first() is the first highlight, if any.
export function mountDocPane(panel, doc) {
  const body = panel.querySelector('#docBody');
  return doc.has_pdf ? mountPdf(panel, body, doc) : mountText(body, doc);
}

// ---------- quote matching ----------
// Mirrors backend extraction.normalize(): case, quote/dash variants and
// whitespace runs are ignored, so a verified quote is always found again.

function foldChar(ch) {
  return ch.normalize('NFKC').toLowerCase()
    .replace(/[‘’‚‛′`´]/g, "'")
    .replace(/[“”„‟″]/g, '"')
    .replace(/[‐‑‒–—―−]/g, '-');
}

function normalize(text) {
  return Array.from(text, foldChar).join('').replace(/\s+/g, ' ').trim();
}

// Finds `quote` in a sequence of strings. Returns [{index, start, end}] per string
// covered by the match, or []. With `ignoreSpaces`, whitespace is dropped on both
// sides: a PDF's text runs split words and lines unpredictably.
function findIn(strings, quote, {ignoreSpaces = false} = {}) {
  let needle = normalize(quote || '');
  if (ignoreSpaces) needle = needle.replace(/ /g, '');
  if (!needle) return [];
  let hay = '';
  const map = [];  // map[k] = {index, i}: source of normalized char k
  let gap = null;
  strings.forEach((text, index) => {
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      if (/\s/.test(ch)) {
        if (hay && !ignoreSpaces) gap = gap || {index, i};
        continue;
      }
      if (gap) { hay += ' '; map.push(gap); gap = null; }
      for (const c of foldChar(ch)) { hay += c; map.push({index, i}); }
    }
    if (hay && !ignoreSpaces) gap = gap || {index, i: text.length};
  });
  const at = hay.indexOf(needle);
  if (at < 0) return [];
  const ranges = new Map();
  for (let k = at; k < at + needle.length; k++) {
    const {index, i} = map[k];
    if (i >= strings[index].length || /\s/.test(strings[index][i])) continue;
    const r = ranges.get(index) || {index, start: i, end: i + 1};
    r.start = Math.min(r.start, i);
    r.end = Math.max(r.end, i + 1);
    ranges.set(index, r);
  }
  return [...ranges.values()];
}

// ---------- text ----------

function paragraphHtml(text, range) {
  if (!range) return esc(text);
  return esc(text.slice(0, range.start)) +
    '<mark class="quote-mark">' + esc(text.slice(range.start, range.end)) + '</mark>' +
    esc(text.slice(range.end));
}

function mountText(body, doc) {
  const render = quote => {
    const byPara = new Map(findIn(doc.paragraphs, quote).map(r => [r.index, r]));
    body.innerHTML = doc.paragraphs
      .map((text, p) => `<p class="doc-para${byPara.has(p) ? ' has-mark' : ''}">${paragraphHtml(text, byPara.get(p))}</p>`)
      .join('');
  };
  render(null);
  const first = () => body.querySelector('mark');
  return {
    first,
    select(quote, {scroll = false} = {}) {
      render(quote);
      if (scroll) scrollBodyTo(body, first());
    },
  };
}

// Scrolls the pane (not the page) so `target` sits a third of the way down.
// Only when the pane is its own scroll area (wide screens).
function scrollBodyTo(body, target) {
  if (!target || body.scrollHeight <= body.clientHeight) return;
  const offset = target.getBoundingClientRect().top - body.getBoundingClientRect().top + body.scrollTop;
  body.scrollTo({top: Math.max(0, offset - body.clientHeight / 3), behavior: 'smooth'});
}

// ---------- PDF ----------

// One loaded PDF at a time, kept across re-renders of the tab so the pages do
// not flash while the patient works through the list.
const cache = {
  docId: null,
  loading: null,   // Promise<{pdf, url}>
  stages: new Map(),  // width -> Promise<HTMLElement> (rendered pages)
};

let pdfjsModule = null;
function pdfjs() {
  if (!pdfjsModule) {
    pdfjsModule = import(PDFJS + 'pdf.min.mjs').then(lib => {
      lib.GlobalWorkerOptions.workerSrc = PDFJS + 'pdf.worker.min.mjs';
      return lib;
    });
    pdfjsModule.catch(() => { pdfjsModule = null; });
  }
  return pdfjsModule;
}

async function fetchPdf(docId) {
  const token = getToken();
  const res = await fetch(`/api/documents/${docId}/file`, {
    headers: token ? {Authorization: 'Bearer ' + token} : {},
  });
  if (res.status === 401 && token) {
    clearSession();
    window.location.href = '/';
  }
  if (!res.ok) throw new Error('file ' + res.status);
  return new Uint8Array(await res.arrayBuffer());
}

function loadPdf(docId) {
  if (cache.docId !== docId) {
    const old = cache.loading;
    if (old) old.then(({pdf, url}) => { pdf.destroy(); URL.revokeObjectURL(url); }, () => {});
    cache.docId = docId;
    cache.stages = new Map();
    cache.loading = (async () => {
      const [lib, bytes] = await Promise.all([pdfjs(), fetchPdf(docId)]);
      const url = URL.createObjectURL(new Blob([bytes], {type: 'application/pdf'}));
      // pdf.js takes ownership of the buffer; the Blob above has its own copy.
      const pdf = await lib.getDocument({
        data: bytes,
        cMapUrl: PDFJS + 'cmaps/',
        standardFontDataUrl: PDFJS + 'standard_fonts/',
        wasmUrl: PDFJS + 'wasm/',
        iccUrl: PDFJS + 'iccs/',
        isEvalSupported: false,
      }).promise;
      return {pdf, url, lib};
    })();
    // A failed load is retried on the next mount instead of being cached.
    cache.loading.catch(() => { if (cache.docId === docId) cache.docId = null; });
  }
  return cache.loading;
}

async function renderPages({pdf, lib}, width) {
  const stage = document.createElement('div');
  stage.className = 'pdf-pages';
  const ratio = window.devicePixelRatio || 1;
  for (let n = 1; n <= pdf.numPages; n++) {
    const page = await pdf.getPage(n);
    const scale = width / page.getViewport({scale: 1}).width;
    const viewport = page.getViewport({scale});

    const sheet = document.createElement('div');
    sheet.className = 'pdf-page';
    sheet.style.width = viewport.width + 'px';
    sheet.style.height = viewport.height + 'px';
    sheet.style.setProperty('--total-scale-factor', String(scale));
    sheet.setAttribute('role', 'img');
    sheet.setAttribute('aria-label', `Page ${n} of ${pdf.numPages}`);

    const canvas = document.createElement('canvas');
    canvas.width = Math.floor(viewport.width * ratio);
    canvas.height = Math.floor(viewport.height * ratio);
    const marks = document.createElement('div');
    marks.className = 'pdf-marks';
    const text = document.createElement('div');
    text.className = 'textLayer';
    sheet.append(canvas, marks, text);
    stage.append(sheet);

    await page.render({
      canvasContext: canvas.getContext('2d'),
      viewport,
      transform: ratio === 1 ? null : [ratio, 0, 0, ratio, 0, 0],
    }).promise;
    await new lib.TextLayer({textContentSource: page.streamTextContent(), container: text, viewport}).render();
  }
  return stage;
}

function stageFor(loaded, width) {
  if (!cache.stages.has(width)) {
    cache.stages.clear();  // only the current width is worth keeping
    const stage = renderPages(loaded, width);
    stage.catch(() => cache.stages.delete(width));
    cache.stages.set(width, stage);
  }
  return cache.stages.get(width);
}

// Draws highlight boxes for `quote` over the rendered pages. Returns the first box.
function markQuote(stage, quote) {
  stage.querySelectorAll('.pdf-marks').forEach(layer => layer.replaceChildren());
  if (!quote) return null;
  const nodes = [];
  stage.querySelectorAll('.textLayer').forEach(layer => {
    const walker = document.createTreeWalker(layer, NodeFilter.SHOW_TEXT);
    for (let node = walker.nextNode(); node; node = walker.nextNode()) nodes.push(node);
  });
  const found = findIn(nodes.map(n => n.data), quote, {ignoreSpaces: true});
  let first = null;
  for (const {index, start, end} of found) {
    const node = nodes[index];
    const sheet = node.parentElement.closest('.pdf-page');
    const layer = sheet.querySelector('.pdf-marks');
    const origin = sheet.getBoundingClientRect();
    const range = document.createRange();
    range.setStart(node, start);
    range.setEnd(node, end);
    for (const rect of range.getClientRects()) {
      if (rect.width < 1 || rect.height < 1) continue;
      const box = document.createElement('div');
      box.className = 'pdf-mark';
      box.style.left = rect.left - origin.left + 'px';
      box.style.top = rect.top - origin.top + 'px';
      box.style.width = rect.width + 'px';
      box.style.height = rect.height + 'px';
      layer.append(box);
      first = first || box;
    }
  }
  return first;
}

function mountPdf(panel, body, doc) {
  let quote = null;
  let stage = null;
  let shownWidth = 0;
  let wantScroll = false;
  const open = panel.querySelector('#docOpen');

  const status = (html) => { body.innerHTML = `<div class="pdf-status">${html}</div>`; };
  status('<span class="spinner" aria-hidden="true"></span>Loading your document...');

  const apply = (scroll) => {
    if (!stage) return;
    const first = markQuote(stage, quote);
    if (scroll) scrollBodyTo(body, first);
  };

  const pageWidth = () => {
    const style = getComputedStyle(body);
    return Math.floor(body.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight));
  };

  async function show() {
    const width = pageWidth();
    if (width <= 0 || width === shownWidth) return;
    shownWidth = width;
    try {
      const loaded = await loadPdf(doc.id);
      if (open) { open.href = loaded.url; open.hidden = false; }
      const rendered = await stageFor(loaded, width);
      if (!body.isConnected || width !== shownWidth) return;
      const top = body.scrollTop;
      stage = rendered;
      body.replaceChildren(stage);
      body.scrollTop = top;
      apply(wantScroll);
      wantScroll = false;
    } catch (_) {
      if (!body.isConnected) return;
      shownWidth = 0;
      stage = null;
      // The text is always there: fall back to it rather than leave the pane empty.
      body.classList.remove('is-pdf');
      body.setAttribute('aria-label', 'Document text');
      const notice = '<p class="pdf-notice">Your PDF could not be shown here, so this is its text.</p>';
      const text = mountText(body, doc);
      text.select(quote);
      body.insertAdjacentHTML('afterbegin', notice);
      controller.first = text.first;
      controller.select = (q, opts) => {
        quote = q;
        text.select(q, opts);
        body.insertAdjacentHTML('afterbegin', notice);
      };
    }
  }

  const controller = {
    first: () => stage && stage.querySelector('.pdf-mark'),
    select(q, {scroll = false} = {}) {
      quote = q;
      if (stage) apply(scroll);
      else wantScroll = scroll;
    },
  };

  // Re-render at the new width when the pane is resized (window or layout change).
  let timer = null;
  const observer = new ResizeObserver(() => {
    if (!body.isConnected) { observer.disconnect(); return; }
    clearTimeout(timer);
    timer = setTimeout(show, 150);
  });
  observer.observe(body);
  show();
  return controller;
}
