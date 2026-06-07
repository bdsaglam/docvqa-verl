"use strict";

const $ = (id) => document.getElementById(id);
const app = $("app");

// ----------------------------------------------------------------------- //
// utilities
// ----------------------------------------------------------------------- //
function esc(s) {
  return String(s ?? "").replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
function fmtPct(x) { return x == null ? "–" : (100 * x).toFixed(1) + "%"; }
function fmtNum(x, d = 0) { return x == null ? "–" : Number(x).toFixed(d); }
function fmtAnls(x) { return x == null ? "–" : Number(x).toFixed(3); }
async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return r.json();
}
function correctBadge(c, anls) {
  if (c === true) return `<span class="badge ok">✓ correct</span>`;
  if (c === false) {
    const cls = anls > 0 ? "mid" : "bad";
    return `<span class="badge ${cls}">✗ ${fmtAnls(anls)}</span>`;
  }
  return `<span class="pill">–</span>`;
}

// very light python highlighter.
// single-pass tokenizer: strings/comments are matched first and emitted as-is,
// keyword/fn/number highlighting only touches the code *between* them — so we
// never re-scan our own inserted <span> markup (which corrupted the output).
function highlightPy(code) {
  const codeSpan = (s) => {
    let h = esc(s);
    h = h.replace(/\b(def|return|for|in|if|else|elif|while|import|from|with|as|class|None|True|False|and|or|not|print|lambda)\b/g,
      '<span class="kw">$1</span>');
    h = h.replace(/\b(batch_look|SUBMIT|search)\b/g, '<span class="fn">$1</span>');
    h = h.replace(/\b(\d+\.?\d*)\b/g, '<span class="num2">$1</span>');
    return h;
  };
  // comment OR (triple-quoted / single / double) string literal
  const re = /(#[^\n]*)|('''[\s\S]*?'''|"""[\s\S]*?"""|'(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*")/g;
  let out = "", last = 0, m;
  while ((m = re.exec(code)) !== null) {
    out += codeSpan(code.slice(last, m.index));
    if (m[1]) out += `<span class="com">${esc(m[1])}</span>`;
    else out += `<span class="str">${esc(m[2])}</span>`;
    last = re.lastIndex;
  }
  out += codeSpan(code.slice(last));
  return out;
}

// split assistant content into {think, body-segments}
function parseAssistant(content) {
  let think = null, rest = content;
  const ti = content.indexOf("</think>");
  if (ti !== -1) {
    think = content.slice(0, ti).replace(/^<think>/, "").trim();
    rest = content.slice(ti + 8);
  } else if (content.startsWith("<think>")) {
    think = content.slice(7).trim();
    rest = "";
  }
  // split rest into prose / ```python code``` segments
  const segs = [];
  const re = /```(\w*)\n?([\s\S]*?)```/g;
  let last = 0, m;
  while ((m = re.exec(rest)) !== null) {
    if (m.index > last) {
      const t = rest.slice(last, m.index).trim();
      if (t) segs.push({ type: "prose", text: t });
    }
    segs.push({ type: "code", lang: m[1] || "python", text: m[2].replace(/\n$/, "") });
    last = re.lastIndex;
  }
  const tail = rest.slice(last).trim();
  if (tail) segs.push({ type: "prose", text: tail });
  return { think, segs };
}

function renderTurn(msg, idx) {
  const role = msg.role;
  if (role === "assistant") {
    const { think, segs } = parseAssistant(msg.content || "");
    let inner = "";
    if (think) {
      inner += `<details class="think" open><summary>&lt;think&gt;</summary>${esc(think)}</details>`;
    }
    for (const s of segs) {
      if (s.type === "prose") inner += `<div class="prose">${esc(s.text)}</div>`;
      else inner += `<pre class="code">${highlightPy(s.text)}</pre>`;
    }
    return turnShell("assistant", `Assistant · turn ${idx}`, inner);
  }
  if (role === "user") {
    // tool output; show verbatim
    return turnShell("user", `User`, `<pre class="out">${esc(msg.content || "")}</pre>`);
  }
  // system
  return turnShell("system",
    `System prompt <span class="pill">${(msg.content || "").length} chars</span>`,
    `<details class="think"><summary>show</summary><pre class="out">${esc(msg.content || "")}</pre></details>`);
}
function turnShell(cls, head, body) {
  return `<div class="turn ${cls}"><div class="turn-h">${head}</div><div class="turn-body">${body}</div></div>`;
}

// ----------------------------------------------------------------------- //
// router
// ----------------------------------------------------------------------- //
let RUNSDIR = "";
function setCrumbs(parts) {
  $("crumbs").innerHTML = parts.map((p, i) => {
    const sep = i ? `<span class="sep">/</span>` : "";
    return sep + (p.hash ? `<a onclick="location.hash='${p.hash}'">${esc(p.label)}</a>` : esc(p.label));
  }).join("");
}

window.addEventListener("hashchange", route);
window.addEventListener("DOMContentLoaded", () => {
  $("home").onclick = () => (location.hash = "#/");
  $("lightbox").onclick = () => $("lightbox").classList.add("hidden");
  route();
});

function route() {
  const h = location.hash.replace(/^#/, "") || "/";
  const parts = h.split("/").filter(Boolean).map(decodeURIComponent);
  if (parts.length === 0) return viewRuns();
  if (parts[0] === "run" && parts.length === 2) return viewTriage(parts[1]);
  if (parts[0] === "run" && parts[1] && parts[2] === "r")
    return viewReader(parts[1], parts[3], parts[4], +parts[5]);
  if (parts[0] === "compare")
    return viewCompare(parts[1], parts[2], parts[3], parts[4], +parts[5]);
  viewRuns();
}

// ----------------------------------------------------------------------- //
// view: run list
// ----------------------------------------------------------------------- //
async function viewRuns() {
  setCrumbs([{ label: "runs" }]);
  app.innerHTML = `<div class="loading">Loading runs…</div>`;
  const data = await getJSON("/api/runs");
  RUNSDIR = data.runs_dir;
  $("runsdir").textContent = data.runs_dir;
  if (!data.runs.length) {
    app.innerHTML = `<div class="empty">No runs found in <code>${esc(data.runs_dir)}</code></div>`;
    return;
  }
  const cards = data.runs.map((r) => {
    const acc = r.overall_accuracy != null
      ? `<span class="badge ${r.overall_accuracy >= 0.3 ? "ok" : r.overall_accuracy > 0 ? "mid" : "bad"}">${fmtPct(r.overall_accuracy)}</span>`
      : `<span class="pill">no results.json</span>`;
    return `<div class="card run-card" onclick="location.hash='#/run/${encodeURIComponent(r.run)}'">
      <h3>${esc(r.run)}</h3>
      <div class="kv">
        <span>${acc}</span>
        <span><b>${r.n_tasks}</b> docs</span>
        <span>n=<b>${r.n ?? "–"}</b></span>
        <span><b>${esc(r.model || "–")}</b></span>
      </div>
      <div class="kv" style="margin-top:6px">
        <span>${esc(r.dataset || "")}/${esc(r.split || "")}</span>
        <span>${esc(r.created_at || "")}</span>
      </div>
    </div>`;
  }).join("");
  app.innerHTML = `<h2>Eval runs</h2><div class="run-grid">${cards}</div>`;
}

// ----------------------------------------------------------------------- //
// view: triage table
// ----------------------------------------------------------------------- //
const triageState = { sort: "anls", dir: 1, filter: "all", q: "" };
let triageRows = [];

async function viewTriage(run) {
  setCrumbs([{ label: "runs", hash: "/" }, { label: run }]);
  app.innerHTML = `<div class="loading">Loading rollouts…</div>`;
  const data = await getJSON(`/api/runs/${encodeURIComponent(run)}/rollouts`);
  triageRows = data.rollouts;
  renderTriage(run);
}

function renderTriage(run) {
  const cols = [
    ["category", "cat"], ["question_id", "question"], ["is_correct", "ok"],
    ["anls", "anls"], ["num_turns", "turns"], ["vlm_calls", "looks"],
    ["wall_clock_s", "wall_s"], ["termination", "term"],
  ];
  let rows = triageRows.slice();
  if (triageState.filter === "correct") rows = rows.filter((r) => r.is_correct === true);
  else if (triageState.filter === "wrong") rows = rows.filter((r) => r.is_correct === false);
  else if (triageState.filter === "partial") rows = rows.filter((r) => r.is_correct === false && r.anls > 0);
  if (triageState.q) {
    const q = triageState.q.toLowerCase();
    rows = rows.filter((r) =>
      (r.question || "").toLowerCase().includes(q) ||
      (r.question_id || "").toLowerCase().includes(q) ||
      (r.gold_answer || "").toLowerCase().includes(q) ||
      (r.submitted_answer || "").toLowerCase().includes(q));
  }
  const s = triageState.sort, dir = triageState.dir;
  rows.sort((a, b) => {
    let x = a[s], y = b[s];
    if (x == null) x = -Infinity; if (y == null) y = -Infinity;
    if (typeof x === "string") return dir * x.localeCompare(y);
    return dir * (x - y);
  });

  const nCorrect = triageRows.filter((r) => r.is_correct === true).length;
  const head = cols.map(([k, lab]) =>
    `<th class="sortable" data-k="${k}">${lab}${s === k ? (dir > 0 ? " ▲" : " ▼") : ""}</th>`).join("");
  const body = rows.map((r) => {
    const sidx = r.sample_idx ?? 0;
    const href = `#/run/${encodeURIComponent(run)}/r/${encodeURIComponent(r.doc_id)}/${encodeURIComponent(r.question_id)}/${sidx}`;
    return `<tr onclick="location.hash='${href}'">
      <td>${esc(r.category)}</td>
      <td><div class="mono">${esc(r.question_id)}${sidx ? " #" + sidx : ""}</div>
          <div style="color:var(--muted);font-size:12px">${esc((r.question || "").slice(0, 90))}</div></td>
      <td>${correctBadge(r.is_correct, r.anls)}</td>
      <td class="num">${fmtAnls(r.anls)}</td>
      <td class="num">${fmtNum(r.num_turns)}</td>
      <td class="num">${fmtNum(r.vlm_calls)}</td>
      <td class="num">${fmtNum(r.wall_clock_s, 0)}</td>
      <td class="mono">${esc(r.termination)}</td>
    </tr>`;
  }).join("");

  app.innerHTML = `
    <h2>${esc(run)} <span class="count">· ${nCorrect}/${triageRows.length} correct</span></h2>
    <div class="toolbar">
      <input type="text" id="q" placeholder="filter question / id / answer…" value="${esc(triageState.q)}" />
      <button class="btn ${triageState.filter === "all" ? "active" : ""}" data-f="all">all (${triageRows.length})</button>
      <button class="btn ${triageState.filter === "correct" ? "active" : ""}" data-f="correct">correct</button>
      <button class="btn ${triageState.filter === "wrong" ? "active" : ""}" data-f="wrong">wrong</button>
      <button class="btn ${triageState.filter === "partial" ? "active" : ""}" data-f="partial">partial anls</button>
      <span class="count">${rows.length} shown</span>
    </div>
    <table><thead><tr>${head}</tr></thead><tbody>${body || `<tr><td colspan="8" class="empty">none</td></tr>`}</tbody></table>`;

  $("q").oninput = (e) => { triageState.q = e.target.value; renderTriage(run); };
  app.querySelectorAll("[data-f]").forEach((b) =>
    (b.onclick = () => { triageState.filter = b.dataset.f; renderTriage(run); }));
  app.querySelectorAll("th[data-k]").forEach((th) =>
    (th.onclick = () => {
      const k = th.dataset.k;
      if (triageState.sort === k) triageState.dir *= -1;
      else { triageState.sort = k; triageState.dir = (k === "anls" || k === "wall_clock_s" || k === "num_turns") ? -1 : 1; }
      renderTriage(run);
    }));
}

// ----------------------------------------------------------------------- //
// view: reader
// ----------------------------------------------------------------------- //
async function fetchRollout(run, doc, qid, sidx) {
  return getJSON(`/api/runs/${encodeURIComponent(run)}/tasks/${encodeURIComponent(doc)}/${encodeURIComponent(qid)}/${sidx}`);
}

function readerHeadHTML(rec, run, doc, qid, sidx) {
  return `<div class="reader-head"><div class="qbox">
    <div class="q"><b>Q:</b> ${esc(rec.question)}</div>
    <div class="answers">
      <span><span class="lab">gold</span><b>${esc(rec.gold_answer)}</b></span>
      <span><span class="lab">submitted</span>${esc(rec.submitted_answer)}</span>
      <span><span class="lab">result</span>${correctBadge(rec.is_correct, rec.anls)}</span>
    </div>
    <div class="meta-strip">
      <span class="pill">${esc(rec.category)}</span>
      <span class="pill">${esc(rec.doc_id)}</span>
      <span class="pill">${rec.num_turns} turns</span>
      <span class="pill">${rec.vlm_calls} looks</span>
      <span class="pill">${fmtNum(rec.wall_clock_s, 0)}s</span>
      <span class="pill">term: ${esc(rec.termination)}</span>
      <span class="pill">${esc(rec.model)}</span>
    </div></div></div>`;
}

function turnsHTML(rec) {
  return `<div class="turns">${(rec.messages || []).map((m, i) => renderTurn(m, i)).join("")}</div>`;
}

async function viewReader(run, doc, qid, sidx) {
  setCrumbs([{ label: "runs", hash: "/" }, { label: run, hash: "/run/" + encodeURIComponent(run) }, { label: qid }]);
  app.innerHTML = `<div class="loading">Loading trajectory…</div>`;
  let rec;
  try { rec = await fetchRollout(run, doc, qid, sidx); }
  catch (e) { app.innerHTML = `<div class="empty">Not found: ${esc(e.message)}</div>`; return; }

  app.innerHTML =
    readerHeadHTML(rec, run, doc, qid, sidx) +
    `<div class="toolbar">
       <button class="btn" id="cmp-btn">compare with another run…</button>
       <button class="btn" id="pages-btn">show document pages</button>
     </div>` +
    turnsHTML(rec) +
    `<div id="pages-panel" class="pages-panel"></div>`;

  $("cmp-btn").onclick = () => openComparePicker(run, doc, qid, sidx);
  $("pages-btn").onclick = () => togglePages(run, doc);
}

async function togglePages(run, doc) {
  const panel = $("pages-panel");
  if (panel.dataset.open === "1") { panel.innerHTML = ""; panel.dataset.open = "0"; return; }
  panel.dataset.open = "1";
  panel.innerHTML = `<div class="loading">Loading pages…</div>`;
  const data = await getJSON(`/api/runs/${encodeURIComponent(run)}/doc/${encodeURIComponent(doc)}/pages`);
  if (!data.pages.length) { panel.innerHTML = `<div class="empty">No page images for ${esc(doc)}</div>`; return; }
  const base = `/api/runs/${encodeURIComponent(run)}/doc/${encodeURIComponent(doc)}/page/`;
  panel.innerHTML = `<h2>${esc(doc)} · ${data.pages.length} pages</h2><div class="pages-grid">` +
    data.pages.map((p) =>
      `<figure onclick="showLightbox('${base}${p}')"><img loading="lazy" src="${base}${p}" /><figcaption>page ${p}</figcaption></figure>`).join("") +
    `</div>`;
}
function showLightbox(src) {
  $("lightbox-img").src = src;
  $("lightbox").classList.remove("hidden");
}

// ----------------------------------------------------------------------- //
// view: compare
// ----------------------------------------------------------------------- //
async function openComparePicker(run, doc, qid, sidx) {
  const data = await getJSON("/api/runs");
  const others = data.runs.filter((r) => r.run !== run);
  const opts = others.map((r) => `<option value="${esc(r.run)}">${esc(r.run)}</option>`).join("");
  const panel = $("pages-panel");
  panel.dataset.open = "1";
  panel.innerHTML = `<div class="card"><b>Compare</b> this rollout against the same question in:
    <select id="cmp-sel">${opts}</select>
    <button class="btn" id="cmp-go">go</button></div>`;
  $("cmp-go").onclick = () => {
    const other = $("cmp-sel").value;
    location.hash = `#/compare/${encodeURIComponent(run)}/${encodeURIComponent(other)}/${encodeURIComponent(doc)}/${encodeURIComponent(qid)}/${sidx}`;
  };
}

async function viewCompare(runA, runB, doc, qid, sidx) {
  setCrumbs([{ label: "runs", hash: "/" }, { label: `compare ${qid}` }]);
  app.innerHTML = `<div class="loading">Loading both trajectories…</div>`;
  const [a, b] = await Promise.allSettled([
    fetchRollout(runA, doc, qid, sidx),
    // sample_idx may differ across runs; fall back to 0
    fetchRollout(runB, doc, qid, sidx).catch(() => fetchRollout(runB, doc, qid, 0)),
  ]);
  const col = (run, res) => {
    if (res.status !== "fulfilled")
      return `<div class="col"><h3>${esc(run)}</h3><div class="empty">not found in this run</div></div>`;
    const rec = res.value;
    return `<div class="col"><h3>${esc(run)}</h3>${readerHeadHTML(rec)}${turnsHTML(rec)}</div>`;
  };
  app.innerHTML = `<h2>Compare · <span class="mono">${esc(qid)}</span></h2>
    <div class="cmp">${col(runA, a)}${col(runB, b)}</div>`;
}
