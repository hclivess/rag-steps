// rag-sqlite web interface. Plain JavaScript, no build step, no libraries.
// The server (rag/server.py) does the work and sends one JSON line per
// finished step; this file only draws what arrives.
'use strict';

const $ = (sel, root = document) => root.querySelector(sel);

function h(tag, props, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (k === 'class') e.className = v;
    else if (k === 'text') e.textContent = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) e.setAttribute(k, v);
  }
  for (const kid of kids.flat()) if (kid != null) e.append(kid);
  return e;
}

// ---- talking to the server ----------------------------------------------------
async function request(url, body) {
  const res = await fetch(url, body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  return res;
}
async function json(url, body) {
  const res = await request(url, body);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}
// Reads newline-delimited JSON as it arrives: one event per finished step.
async function* stream(url, body) {
  const res = await request(url, body);
  if (!res.ok) throw new Error((await res.json()).error || res.statusText);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf('\n')) >= 0) {
      const line = buffer.slice(0, cut).trim();
      buffer = buffer.slice(cut + 1);
      if (line) yield JSON.parse(line);
    }
  }
  if (buffer.trim()) yield JSON.parse(buffer);
}

// ---- playback: the pipeline takes milliseconds, the display is slowed ----------
const DELAYS = { instant: [0, 0], normal: [450, 70], slow: [1400, 350] };
function pause(kind) {
  const ms = DELAYS[$('#speed').value][kind === 'step' ? 0 : 1];
  return ms ? new Promise((r) => setTimeout(r, ms)) : null;
}

// ---- shared state --------------------------------------------------------------
let files = [];                     // documents listed in the Read card
let selected = 0;
let busy = false;
const ingested = new Map();         // doc index -> chunks as they were ingested
const embedded = new Map();         // doc index -> Set of embedded chunk positions
let queryWords = [];

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const baseName = (p) => String(p).split(/[\\/]/).pop();
const settings = () => ({ chunk_size: +$('#size').value, overlap: +$('#overlap').value });

function setStatus(step, state, text) {
  const card = $('#s-' + step);
  card.classList.remove('run', 'done', 'skip');
  if (state) card.classList.add(state);
  $('.status', card).textContent = text || '';
}
// Mark one step as running; steps that were running become done.
function activate(group, step, text) {
  for (const id of group) {
    const card = $('#s-' + id);
    if (id === step) setStatus(id, 'run', text);
    else if (card.classList.contains('run')) setStatus(id, 'done', 'done');
  }
}
function setBusy(on) {
  busy = on;
  for (const id of ['#start', '#ask', '#reset', '#paste-add']) $(id).disabled = on;
}
function progress(which, fraction, text, isError) {
  $(`#${which}-bar`).style.width = Math.round(Math.min(1, fraction) * 100) + '%';
  const label = $(`#${which}-label`);
  label.textContent = text;
  label.classList.toggle('error', !!isError);
}
function showFacts(state) {
  if (state.embedder) $('#f-embedder').textContent = state.embedder;
  if (state.llm) $('#f-llm').textContent = state.llm;
  if (state.db) $('#f-db').textContent = state.db;
  $('#f-docs').textContent = state.stats.documents;
  $('#f-chunks').textContent = state.stats.chunks;
}
async function refreshState() {
  const state = await json('/api/state');
  showFacts(state);
  return state;
}

// ---- drawing: vectors ----------------------------------------------------------
// One vector as a small tile: every cell is one dimension, brighter = larger value.
function vectorTile(vector, label) {
  const dim = vector.length;
  const cols = Math.ceil(Math.sqrt(dim * 2));
  const rows = Math.ceil(dim / cols);
  const cell = dim > 2000 ? 2 : 3;
  const canvas = h('canvas', { width: cols * cell, height: rows * cell, title: `${dim} dimensions` });
  const ctx = canvas.getContext('2d');
  const max = vector.reduce((m, v) => Math.max(m, Math.abs(v)), 0) || 1;
  const plus = css('--accent'), minus = css('--kw');
  vector.forEach((v, i) => {
    if (!v) return;
    ctx.globalAlpha = 0.3 + 0.7 * Math.abs(v) / max;
    ctx.fillStyle = v > 0 ? plus : minus;
    ctx.fillRect((i % cols) * cell, Math.floor(i / cols) * cell, cell, cell);
  });
  return h('div', { class: 'tile' }, canvas, label ? h('span', { text: label }) : null);
}
function tokenChips(ev) {
  if (!ev.tokens) return null;
  const chips = ev.tokens.map((t) => h('span', { class: 'tok' }, t.token + ' → ', h('b', { text: t.bucket })));
  if (ev.token_count > ev.tokens.length) chips.push(h('span', { class: 'tok', text: `+${ev.token_count - ev.tokens.length} more` }));
  return h('div', { class: 'toks' }, chips);
}

// ---- A: build the index --------------------------------------------------------
const INGEST = ['read', 'chunk', 'embed', 'store'];

function renderFiles() {
  const box = $('#read-live');
  box.replaceChildren();
  if (!files.length) {
    box.append(h('p', { class: 'empty', text: 'No .md or .txt files here. Pick another folder or add your own text.' }));
    return;
  }
  files.forEach((f, i) => {
    const detail = f.read
      ? `${f.read.chars} characters · ${f.read.paragraphs} paragraphs`
      : `${f.chars != null ? f.chars + ' bytes · ' : ''}not read yet`;
    box.append(h('button', { type: 'button', class: 'file' + (i === selected ? ' sel' : ''), onclick: () => select(i) },
      h('span', { class: 'tick', text: f.read ? '✓' : '·' }),
      h('b', { text: f.title }),
      h('small', { text: f.source }),
      h('small', { text: detail })));
  });
}

function renderChunks(chunks, note) {
  const box = $('#chunk-live');
  box.replaceChildren();
  const f = files[selected];
  if (!f) { box.append(h('p', { class: 'empty', text: 'No document selected.' })); return; }
  const done = embedded.get(selected) || new Set();
  box.append(h('div', { class: 'chunk-head', text: `${f.title}: ${chunks.length} chunks · ${note}` }));
  box.append(h('div', { class: 'minimap', title: 'chunk lengths' },
    chunks.map((c) => h('i', { style: `flex: ${c.chars} 1 0`, class: done.has(c.position) ? 'emb' : '', 'data-pos': c.position }))));
  for (const c of chunks) {
    box.append(h('div', { class: 'chunk' + (done.has(c.position) ? ' emb' : ''), 'data-pos': c.position },
      h('div', null, h('span', { text: `chunk ${c.position + 1}` }),
        h('span', { text: `${c.chars} chars` + (c.overlap ? ` · ${c.overlap} overlap` : '') })),
      h('pre', null, c.overlap ? h('mark', { text: c.text.slice(0, c.overlap) }) : null, c.text.slice(c.overlap))));
  }
}

let previewTicket = 0;
async function preview() {
  const f = files[selected];
  if (!f) { renderChunks([], ''); return; }
  const ticket = ++previewTicket;
  const body = f.pasted ? { text: f.text, ...settings() } : { folder: $('#folder').value, source: f.source, ...settings() };
  try {
    const data = await json('/api/chunk', body);
    if (ticket === previewTicket) renderChunks(data.chunks, 'live preview, nothing stored yet');
  } catch (err) {
    if (ticket === previewTicket) $('#chunk-live').replaceChildren(h('p', { class: 'empty', text: String(err.message) }));
  }
}

function select(i) {
  selected = i;
  renderFiles();
  if (ingested.has(i)) renderChunks(ingested.get(i), 'as ingested');
  else preview();
}

async function loadFiles() {
  const pasted = files.filter((f) => f.pasted);
  ingested.clear(); embedded.clear();
  try {
    const data = await json('/api/files', { folder: $('#folder').value });
    files = [...data.files, ...pasted];
    if (!busy) progress('ingest', 0, `${files.length} documents listed. The chunk card shows a live preview of the selected one.`);
  } catch (err) {
    files = pasted;
    progress('ingest', 0, err.message, true);
  }
  select(0);
}

function resetIngestCards() {
  ingested.clear(); embedded.clear();
  for (const f of files) delete f.read;
  for (const id of INGEST) setStatus(id, '', '');
  $('#embed-now').replaceChildren();
  $('#embed-tiles').replaceChildren();
  $('#store-rows').replaceChildren();
  $('#store-live .totals')?.remove();
}

async function onIngest(ev) {
  const total = Math.max(1, files.length);
  switch (ev.step) {
    case 'plan':
      if (ev.files.length !== files.length) files = ev.files;
      if (!ev.files.length) throw new Error('No documents to ingest: the folder has no .md or .txt files.');
      renderFiles();
      break;
    case 'read':
      activate(INGEST, 'read', `${ev.doc + 1} of ${files.length}`);
      files[ev.doc].read = { chars: ev.chars, paragraphs: ev.paragraphs };
      selected = ev.doc;
      renderFiles();
      progress('ingest', ev.doc / total, `Reading ${ev.source}`);
      await pause('step');
      break;
    case 'chunk':
      activate(INGEST, 'chunk', `${ev.chunks.length} chunks`);
      ingested.set(ev.doc, ev.chunks);
      embedded.set(ev.doc, new Set());
      renderChunks(ev.chunks, 'as ingested');
      progress('ingest', (ev.doc + 0.25) / total, `Split ${files[ev.doc].title} into ${ev.chunks.length} chunks`);
      await pause('step');
      break;
    case 'embed': {
      activate(INGEST, 'embed', `${ev.position + 1} of ${ev.of}`);
      embedded.get(ev.doc).add(ev.position);
      const live = $('#chunk-live');
      for (const node of live.querySelectorAll(`[data-pos="${ev.position}"]`)) node.classList.add('emb');
      const block = $(`.chunk[data-pos="${ev.position}"]`, live);
      if (block) live.scrollTop = block.offsetTop - live.offsetTop - 60;
      $('#embed-now').replaceChildren(
        h('div', { class: 'line', text: `chunk ${ev.position + 1} of ${files[ev.doc].title} → ${ev.dim} numbers, ${ev.nonzero} not zero` }),
        tokenChips(ev));
      const tiles = $('#embed-tiles');
      $('.tile.last', tiles)?.classList.remove('last');
      const tile = vectorTile(ev.vector, `d${ev.doc + 1} c${ev.position + 1}`);
      tile.classList.add('last');
      tiles.append(tile);
      if (tiles.children.length > 300) tiles.firstChild.remove();
      progress('ingest', (ev.doc + 0.25 + 0.65 * (ev.position + 1) / ev.of) / total,
        `Embedding ${files[ev.doc].title}: chunk ${ev.position + 1} of ${ev.of}`);
      await pause('item');
      break;
    }
    case 'store':
      activate(INGEST, 'store', ev.skipped ? 'empty document' : `document ${ev.document_id}`);
      $('#store-rows').append(h('tr', null,
        h('td', { text: ev.skipped ? '–' : ev.document_id }),
        h('td', { text: baseName(ev.source) }),
        h('td', { text: ev.rows }),
        h('td', { text: ev.skipped ? '–' : `${ev.blob_bytes} bytes each` })));
      if (ev.stats) {
        showFacts({ stats: ev.stats });
        $('#store-live .totals')?.remove();
        $('#store-live').append(h('div', { class: 'totals',
          text: `In the database now: ${ev.stats.documents} documents, ${ev.stats.chunks} chunks` }));
      }
      progress('ingest', (ev.doc + 1) / total, `Stored ${baseName(ev.source)}`);
      await pause('step');
      break;
    case 'done':
      for (const id of INGEST) setStatus(id, 'done', 'done');
      $('.tile.last')?.classList.remove('last');
      showFacts({ stats: ev.stats });
      progress('ingest', 1, `Done: ${ev.chunks} chunks from ${ev.documents} documents in ${ev.ms} ms (pipeline time, without playback).`);
      break;
    case 'error':
      throw new Error(ev.message);
  }
}

async function runIngest() {
  if (busy) return;
  setBusy(true);
  resetIngestCards();
  renderFiles();
  progress('ingest', 0, 'Starting…');
  const body = {
    folder: $('#folder').value,
    documents: files.filter((f) => f.pasted).map((f) => ({ source: f.source, title: f.title, text: f.text })),
    ...settings(),
  };
  try {
    for await (const ev of stream('/api/ingest', body)) await onIngest(ev);
  } catch (err) {
    for (const id of INGEST) if ($('#s-' + id).classList.contains('run')) setStatus(id, '', 'failed');
    progress('ingest', 0, 'Failed: ' + err.message, true);
  } finally {
    setBusy(false);
  }
}

// ---- B: ask --------------------------------------------------------------------
const ASK = ['embed_query', 'vector_search', 'keyword_search', 'fusion', 'prompt', 'answer'];
const ASK_NAMES = {
  embed_query: 'Question embedded', vector_search: 'Vector search finished', keyword_search: 'Keyword search finished',
  fusion: 'Rankings fused', prompt: 'Prompt built', answer: 'Answer generated',
};

function highlighted(text) {
  if (!queryWords.length) return [text];
  const pattern = new RegExp(`\\b(${queryWords.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})\\b`, 'gi');
  return text.split(pattern).map((part, i) => (i % 2 ? h('mark', { class: 'q', text: part }) : part));
}
function hitList(hits, scale) {
  if (!hits.length) return [h('p', { class: 'empty', text: 'No chunk matched.' })];
  return hits.map((hit, i) => h('div', { class: 'hit' },
    h('div', { class: 'meta' },
      h('span', { text: `#${i + 1} · chunk ${hit.chunk_id} · ${baseName(hit.source)}` }),
      h('span', { text: hit.score.toFixed(4) })),
    h('div', { class: 'meter' }, h('i', { style: `width: ${Math.max(0, Math.min(1, hit.score / scale)) * 100}%` })),
    h('p', null, highlighted(hit.text.length > 240 ? hit.text.slice(0, 240) + '…' : hit.text))));
}
const rankCell = (rank, c) => (rank == null ? 'not found' : `#${rank + 1} → 1/(${c}+${rank})`);

function resetAskCards() {
  for (const id of ASK) setStatus(id, '', '');
  for (const id of ['embed_query', 'fusion', 'prompt', 'answer']) $(`#${id}-live`).replaceChildren();
  for (const id of ['vector_search', 'keyword_search']) $(`#${id}-live .hits`).replaceChildren();
  $('#keyword_search-live .fts').replaceChildren();
}

async function onAsk(ev, count) {
  if (ev.step === 'error') throw new Error(ev.message);
  activate(ASK, ev.step, 'working');
  if (ev.skipped) {
    setStatus(ev.step, 'skip', 'not used in this mode');
  } else if (ev.step === 'embed_query') {
    if (ev.tokens) queryWords = ev.tokens.map((t) => t.token).filter((t) => !t.includes(' '));
    $('#embed_query-live').replaceChildren(
      h('div', { class: 'now' },
        h('div', { class: 'line', text: `the question → ${ev.dim} numbers, ${ev.nonzero} not zero` }),
        tokenChips(ev)),
      vectorTile(ev.vector, 'question vector'));
    setStatus(ev.step, 'run', `${ev.dim} dimensions`);
  } else if (ev.step === 'vector_search') {
    $('#vector_search-live .hits').replaceChildren(...hitList(ev.hits, 1));
    setStatus(ev.step, 'run', `${ev.scanned} chunks scanned · ${ev.ms} ms`);
  } else if (ev.step === 'keyword_search') {
    $('#keyword_search-live .fts').replaceChildren('MATCH ', h('b', { text: ev.fts_query }));
    const box = $('#keyword_search-live .hits');
    box.classList.add('kw');
    box.replaceChildren(...hitList(ev.hits, Math.max(...ev.hits.map((x) => x.score), 1e-9)));
    setStatus(ev.step, 'run', `${ev.hits.length} matches · ${ev.ms} ms`);
  } else if (ev.step === 'fusion') {
    $('#fusion-live').replaceChildren(h('table', { class: 'rows' },
      h('thead', null, h('tr', null, ['chunk', 'vector rank', 'keyword rank', 'sum'].map((t) => h('th', { text: t })))),
      h('tbody', null, ev.rows.map((r) => h('tr', { class: r.kept ? 'kept' : 'cut' },
        h('td', { text: `${r.chunk_id} · ${baseName(r.source)}` }),
        h('td', { text: rankCell(r.vector_rank, ev.c) }),
        h('td', { text: rankCell(r.keyword_rank, ev.c) }),
        h('td', { text: r.score.toFixed(4) }))))),
      h('div', { class: 'totals', text: `The top ${ev.k} go into the prompt.` }));
    setStatus(ev.step, 'run', `${ev.rows.length} candidates`);
  } else if (ev.step === 'prompt') {
    $('#prompt-live').replaceChildren(
      h('div', { class: 'label', text: 'System instruction' }), h('pre', { text: ev.system }),
      h('div', { class: 'label', text: `Prompt (${ev.hits.length} chunks + question)` }), h('pre', { text: ev.prompt }));
    setStatus(ev.step, 'run', `${ev.prompt.length} characters`);
  } else if (ev.step === 'answer') {
    $('#answer-live').replaceChildren(h('div', { class: 'answer', text: ev.text }));
    setStatus(ev.step, 'done', `${ev.llm} · ${ev.ms} ms`);
  }
  progress('ask', count / ASK.length, ev.skipped ? 'Skipped a step this mode does not use' : ASK_NAMES[ev.step]);
  if (ev.step !== 'answer') await pause('step');
}

async function runAsk() {
  if (busy) return;
  const question = $('#question').value.trim();
  if (!question) { progress('ask', 0, 'Type a question first.', true); return; }
  if ($('#f-chunks').textContent === '0') { progress('ask', 0, 'The index is empty. Press Start ingest first.', true); return; }
  setBusy(true);
  resetAskCards();
  queryWords = (question.toLowerCase().match(/[a-z0-9]+/g) || []).filter((w) => w.length > 2);
  progress('ask', 0, 'Starting…');
  let count = 0;
  try {
    for await (const ev of stream('/api/ask', { question, k: +$('#k').value, mode: $('#mode').value })) {
      await onAsk(ev, ++count);
    }
    for (const id of ASK) if ($('#s-' + id).classList.contains('run')) setStatus(id, 'done', $('#s-' + id + ' .status').textContent);
    progress('ask', 1, 'Done. Every card below shows what its step produced.');
  } catch (err) {
    for (const id of ASK) if ($('#s-' + id).classList.contains('run')) setStatus(id, '', 'failed');
    progress('ask', 0, 'Failed: ' + err.message, true);
  } finally {
    setBusy(false);
  }
}

// ---- wiring --------------------------------------------------------------------
function showSettings() {
  const size = $('#size'), overlap = $('#overlap');
  if (+overlap.value >= +size.value) overlap.value = Math.max(0, +size.value - 20);
  $('#size-out').textContent = size.value + ' chars';
  $('#overlap-out').textContent = overlap.value + ' chars';
}
let debounce;
function settingsChanged() {
  showSettings();
  if (busy) return;
  ingested.clear(); embedded.clear();           // the stored chunks no longer match the sliders
  clearTimeout(debounce);
  debounce = setTimeout(preview, 60);
}

async function init() {
  const state = await refreshState();
  $('#folder').value = state.defaults.folder;
  $('#size').value = state.defaults.chunk_size;
  $('#overlap').value = state.defaults.overlap;
  showSettings();

  $('#size').addEventListener('input', settingsChanged);
  $('#overlap').addEventListener('input', settingsChanged);
  $('#folder').addEventListener('change', () => { if (!busy) loadFiles(); });
  $('#ingest-form').addEventListener('submit', (e) => { e.preventDefault(); runIngest(); });
  $('#ask-form').addEventListener('submit', (e) => { e.preventDefault(); runAsk(); });
  $('#reset').addEventListener('click', async () => {
    if (busy) return;
    showFacts(await json('/api/reset', {}));
    resetIngestCards();
    renderFiles();
    progress('ingest', 0, 'Database emptied.');
  });
  $('#paste-add').addEventListener('click', () => {
    const text = $('#paste-text').value;
    if (!text.trim() || busy) return;
    const name = $('#paste-name').value.trim() || `pasted-${files.filter((f) => f.pasted).length + 1}`;
    files.push({ source: name, title: name, pasted: true, text });
    $('#paste-text').value = ''; $('#paste-name').value = '';
    ingested.clear(); embedded.clear();
    select(files.length - 1);
  });

  await loadFiles();

  // http://127.0.0.1:8765/?autorun  runs both pipelines once, without playback delay.
  const params = new URLSearchParams(location.search);
  if (params.has('autorun')) {
    $('#speed').value = 'instant';
    if (params.get('autorun')) $('#question').value = params.get('autorun');
    await runIngest();
    await runAsk();
    document.title = 'ready · ' + document.title;
  }
}

init().catch((err) => progress('ingest', 0, 'Could not reach the server: ' + err.message, true));
