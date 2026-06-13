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

// extract page indices referenced via `pages[N]` in code segments (first-seen
// order, deduped). `dynamic` flags `pages[<var/expr>]` we can't resolve statically.
function extractPageRefs(segs) {
  const seen = new Set(), pages = [];
  let dynamic = false;
  for (const s of segs) {
    if (s.type !== "code") continue;
    let m;
    const re = /pages\[\s*(\d+)\s*\]/g;
    while ((m = re.exec(s.text)) !== null) {
      const n = +m[1];
      if (!seen.has(n)) { seen.add(n); pages.push(n); }
    }
    if (/pages\[\s*[^\]\d\s][^\]]*\]/.test(s.text)) dynamic = true; // pages[i], pages[a:b]
  }
  return { pages, dynamic };
}

function lookStripHTML(segs, ctx) {
  if (!ctx || !ctx.run || !ctx.doc) return "";
  const { pages, dynamic } = extractPageRefs(segs);
  if (!pages.length && !dynamic) return "";
  const base = `/api/runs/${encodeURIComponent(ctx.run)}/doc/${encodeURIComponent(ctx.doc)}/page/`;
  const thumbs = pages.map((p) =>
    `<figure onclick="showLightbox('${base}${p}')"><img loading="lazy" src="${base}${p}" ` +
    `onerror="this.closest('figure').style.display='none'"/><figcaption>p${p}</figcaption></figure>`).join("");
  const note = dynamic ? `<span class="dyn-note">+ pages via loop/variable (not shown)</span>` : "";
  const lab = `<div class="look-lab">👁 looked at ${pages.length} page${pages.length === 1 ? "" : "s"} ${note}</div>`;
  return `<div class="lookpages">${lab}<div class="look-grid">${thumbs}</div></div>`;
}

function renderTurn(msg, idx, ctx) {
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
    inner += lookStripHTML(segs, ctx);
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
const triageState = { sort: "rate", dir: 1, filter: "all", q: "", expanded: new Set() };
let triageRows = [];

async function viewTriage(run) {
  setCrumbs([{ label: "runs", hash: "/" }, { label: run }]);
  app.innerHTML = `<div class="loading">Loading rollouts…</div>`;
  const data = await getJSON(`/api/runs/${encodeURIComponent(run)}/rollouts`);
  triageRows = data.rollouts;
  triageState.expanded = new Set();
  renderTriage(run);
}

const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);

// collapse sample-level rollouts into one group per question_id
function buildGroups(rows) {
  const map = new Map();
  for (const r of rows) {
    let g = map.get(r.question_id);
    if (!g) {
      g = { question_id: r.question_id, doc_id: r.doc_id, category: r.category,
            question: r.question, gold_answer: r.gold_answer, samples: [] };
      map.set(r.question_id, g);
    }
    g.samples.push(r);
  }
  const groups = [];
  for (const g of map.values()) {
    g.samples.sort((a, b) => (a.sample_idx ?? 0) - (b.sample_idx ?? 0));
    const anls = g.samples.map((s) => s.anls ?? 0);
    g.n = g.samples.length;
    g.nCorrect = g.samples.filter((s) => s.is_correct === true).length;
    g.rate = g.nCorrect / g.n;
    g.anyCorrect = g.nCorrect > 0;
    g.allCorrect = g.nCorrect === g.n;
    g.meanAnls = mean(anls);
    g.maxAnls = Math.max(...anls);
    g.meanTurns = mean(g.samples.map((s) => s.num_turns ?? 0));
    g.meanLooks = mean(g.samples.map((s) => s.vlm_calls ?? 0));
    g.meanWall = mean(g.samples.map((s) => s.wall_clock_s ?? 0));
    groups.push(g);
  }
  return groups;
}

function groupBadge(g) {
  if (g.allCorrect) return `<span class="badge ok">✓ ${g.nCorrect}/${g.n}</span>`;
  if (g.anyCorrect) return `<span class="badge mid">~ ${g.nCorrect}/${g.n}</span>`;
  return `<span class="badge bad">✗ 0/${g.n}</span>`;
}

function renderTriage(run) {
  if (!triageRows.length) {
    app.innerHTML = `<h2>${esc(run)}</h2>
      <div class="empty">No trajectories on disk for this run yet —
      <code>${esc(run)}/tasks/*/trajectories.jsonl</code> is empty.<br/>
      (Collection/eval may still be running, or this run was cleaned up.)</div>`;
    return;
  }
  const allGroups = buildGroups(triageRows);

  // run-level stats: docs / questions / samples are three different things
  const nSamples = triageRows.length;
  const nQuestions = allGroups.length;
  const nDocs = new Set(triageRows.map((r) => r.doc_id)).size;
  const sampleCorrect = triageRows.filter((r) => r.is_correct === true).length;
  const nSolved = allGroups.filter((g) => g.anyCorrect).length;
  const nUnsolved = allGroups.filter((g) => !g.anyCorrect).length;
  const nMixed = allGroups.filter((g) => g.anyCorrect && !g.allCorrect).length;

  // filter groups
  let groups = allGroups;
  if (triageState.filter === "solved") groups = groups.filter((g) => g.anyCorrect);
  else if (triageState.filter === "unsolved") groups = groups.filter((g) => !g.anyCorrect);
  else if (triageState.filter === "mixed") groups = groups.filter((g) => g.anyCorrect && !g.allCorrect);
  if (triageState.q) {
    const q = triageState.q.toLowerCase();
    groups = groups.filter((g) =>
      (g.question || "").toLowerCase().includes(q) ||
      (g.question_id || "").toLowerCase().includes(q) ||
      (g.gold_answer || "").toLowerCase().includes(q) ||
      g.samples.some((s) => (s.submitted_answer || "").toLowerCase().includes(q)));
  }

  const s = triageState.sort, dir = triageState.dir;
  groups = groups.slice().sort((a, b) => {
    let x = a[s], y = b[s];
    if (x == null) x = -Infinity; if (y == null) y = -Infinity;
    if (typeof x === "string") return dir * x.localeCompare(y);
    return dir * (x - y);
  });

  const cols = [
    ["", ""], ["category", "cat"], ["", "question"], ["rate", "solved"],
    ["meanAnls", "anls μ"], ["maxAnls", "anls↑"], ["meanTurns", "turns μ"],
    ["meanLooks", "looks μ"], ["meanWall", "wall μ"], ["n", "n"],
  ];
  const head = cols.map(([k, lab]) =>
    k ? `<th class="sortable" data-k="${k}">${lab}${s === k ? (dir > 0 ? " ▲" : " ▼") : ""}</th>`
      : `<th>${lab}</th>`).join("");

  const body = groups.map((g) => {
    const open = triageState.expanded.has(g.question_id);
    const grow = `<tr class="grp" data-q="${esc(g.question_id)}">
      <td class="caret">${open ? "▾" : "▸"}</td>
      <td>${esc(g.category)}</td>
      <td><div class="mono">${esc(g.question_id)}</div>
          <div style="color:var(--muted);font-size:12px">${esc((g.question || "").slice(0, 90))}</div></td>
      <td>${groupBadge(g)}</td>
      <td class="num">${fmtAnls(g.meanAnls)}</td>
      <td class="num">${fmtAnls(g.maxAnls)}</td>
      <td class="num">${fmtNum(g.meanTurns, 1)}</td>
      <td class="num">${fmtNum(g.meanLooks, 1)}</td>
      <td class="num">${fmtNum(g.meanWall, 0)}</td>
      <td class="num">${g.n}</td>
    </tr>`;
    if (!open) return grow;
    const subs = g.samples.map((r) => {
      const sidx = r.sample_idx ?? 0;
      const href = `#/run/${encodeURIComponent(run)}/r/${encodeURIComponent(r.doc_id)}/${encodeURIComponent(r.question_id)}/${sidx}`;
      return `<tr class="samp" onclick="location.hash='${href}'">
        <td></td><td></td>
        <td class="mono samp-q">↳ sample #${sidx} · ${esc(r.termination ?? "")}</td>
        <td>${correctBadge(r.is_correct, r.anls)}</td>
        <td class="num">${fmtAnls(r.anls)}</td>
        <td class="num"></td>
        <td class="num">${fmtNum(r.num_turns)}</td>
        <td class="num">${fmtNum(r.vlm_calls)}</td>
        <td class="num">${fmtNum(r.wall_clock_s, 0)}</td>
        <td class="num"></td>
      </tr>`;
    }).join("");
    return grow + subs;
  }).join("");

  const fbtn = (f, lab) =>
    `<button class="btn ${triageState.filter === f ? "active" : ""}" data-f="${f}">${lab}</button>`;
  app.innerHTML = `
    <h2>${esc(run)} <span class="count">· sample acc ${fmtAnls(sampleCorrect / nSamples)}</span></h2>
    <div class="meta-strip" style="margin-bottom:12px">
      <span class="pill"><b>${nDocs}</b> docs</span>
      <span class="pill"><b>${nQuestions}</b> questions</span>
      <span class="pill"><b>${nSamples}</b> samples</span>
      <span class="pill">solved <b>${nSolved}</b>/${nQuestions} (${fmtAnls(nSolved / nQuestions)})</span>
      <span class="pill">sample-correct ${sampleCorrect}/${nSamples}</span>
    </div>
    <div class="toolbar">
      <input type="text" id="q" placeholder="filter question / id / answer…" value="${esc(triageState.q)}" />
      ${fbtn("all", `all (${nQuestions})`)}
      ${fbtn("solved", `solved (${nSolved})`)}
      ${fbtn("unsolved", `unsolved (${nUnsolved})`)}
      ${fbtn("mixed", `mixed (${nMixed})`)}
      <span class="spacer"></span>
      <button class="btn" id="expand-all">${triageState.expanded.size ? "collapse all" : "expand all"}</button>
    </div>
    <table class="grouped"><thead><tr>${head}</tr></thead>
      <tbody>${body || `<tr><td colspan="10" class="empty">none</td></tr>`}</tbody></table>`;

  $("q").oninput = (e) => { triageState.q = e.target.value; renderTriage(run); };
  app.querySelectorAll("[data-f]").forEach((b) =>
    (b.onclick = () => { triageState.filter = b.dataset.f; renderTriage(run); }));
  $("expand-all").onclick = () => {
    if (triageState.expanded.size) triageState.expanded.clear();
    else groups.forEach((g) => triageState.expanded.add(g.question_id));
    renderTriage(run);
  };
  app.querySelectorAll("tr.grp").forEach((tr) =>
    (tr.onclick = () => {
      const q = tr.dataset.q;
      if (triageState.expanded.has(q)) triageState.expanded.delete(q);
      else triageState.expanded.add(q);
      renderTriage(run);
    }));
  app.querySelectorAll("th[data-k]").forEach((th) =>
    (th.onclick = () => {
      const k = th.dataset.k;
      if (triageState.sort === k) triageState.dir *= -1;
      else { triageState.sort = k; triageState.dir = (k === "category") ? 1 : -1; }
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

function turnsHTML(rec, run) {
  const ctx = { run, doc: rec.doc_id };
  return `<div class="turns">${(rec.messages || []).map((m, i) => renderTurn(m, i, ctx)).join("")}</div>`;
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
       <label class="toggle"><input type="checkbox" id="pages-chk"/> document pages</label>
     </div>` +
    turnsHTML(rec, run) +
    `<div id="pages-panel" class="pages-panel"></div>`;

  $("cmp-btn").onclick = () => openComparePicker(run, doc, qid, sidx);
  $("pages-chk").onchange = (e) => setPagesPanel(run, doc, e.target.checked);
}

async function setPagesPanel(run, doc, show) {
  const panel = $("pages-panel");
  if (!show) { panel.innerHTML = ""; panel.dataset.open = "0"; return; }
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
    return `<div class="col"><h3>${esc(run)}</h3>${readerHeadHTML(rec)}${turnsHTML(rec, run)}</div>`;
  };
  app.innerHTML = `<h2>Compare · <span class="mono">${esc(qid)}</span></h2>
    <div class="cmp">${col(runA, a)}${col(runB, b)}</div>`;
}
