"use strict";

const CFG = window.APP_CONFIG || {};
const API = (CFG.API_BASE || "").replace(/\/$/, "");
const $ = (sel) => document.querySelector(sel);
const STEPS = [
  ["search", "Search literature"],
  ["annotate", "Annotate genes"],
  ["verify", "Verify human genes"],
  ["network", "Interaction network"],
  ["llm", "LLM evidence review"],
  ["rank", "Rank"],
];
const PART_CLASS = { literature: "lit", llm_relevance: "rel", llm_evidence: "evi", network: "net" };
const PART_LABEL = { literature: "Literature", llm_relevance: "LLM relevance", llm_evidence: "LLM evidence", network: "Network" };

let current = null;     // the result being shown
let source = null;      // active EventSource
let timer = null;

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (n) => (typeof n === "number" ? n.toLocaleString() : esc(n));

/* ---------- backend status ---------- */
async function checkServer() {
  const pill = $("#server-status");
  const set = (cls, text) => { pill.className = "status-pill " + cls; pill.lastElementChild.textContent = text; };
  const slow = setTimeout(() => set("waking", "Waking server…"), 2500);
  try {
    const res = await fetch(API + "/api/health", { signal: AbortSignal.timeout(90000) });
    const body = await res.json();
    set("ok", body.llm_enabled ? "Live · LLM on" : "Live · literature only");
    return true;
  } catch {
    set("down", "Server offline · examples only");
    return false;
  } finally {
    clearTimeout(slow);
  }
}

/* ---------- examples (static JSON, no backend needed) ---------- */
async function loadExamples() {
  try {
    const list = await (await fetch("examples/index.json")).json();
    $("#example-chips").innerHTML = list
      .map((e) => `<button type="button" class="chip" data-file="${esc(e.file)}">${esc(e.topic)}</button>`)
      .join("");
    $("#example-chips").addEventListener("click", (ev) => {
      const btn = ev.target.closest(".chip");
      if (btn) openExample(btn.dataset.file);
    });
    return list;
  } catch {
    $(".examples").hidden = true;
    return [];
  }
}

async function openExample(file) {
  stopSearch();
  $("#error").hidden = true;
  try {
    const result = await (await fetch("examples/" + encodeURIComponent(file))).json();
    $("#topic").value = result.topic;
    $("#k").value = result.k;
    history.replaceState(null, "", "#example=" + encodeURIComponent(file));
    show({ ...result, _example: true });
  } catch {
    showError("That example could not be loaded.");
  }
}

/* ---------- search ---------- */
function stopSearch() {
  if (source) source.close();
  source = null;
  clearInterval(timer);
  $("#submit").disabled = false;
  $("#progress").hidden = true;
}

function startSearch(topic, k) {
  stopSearch();
  $("#error").hidden = true;
  $("#results").hidden = true;
  $("#progress").hidden = false;
  $("#progress-topic").textContent = `“${topic}”`;
  $("#progress-message").textContent = "Connecting to the server…";
  $("#coldstart").hidden = true;
  $("#steps").innerHTML = STEPS.map(([id, label]) => `<li data-step="${id}">${label}</li>`).join("");
  $("#submit").disabled = true;

  const t0 = Date.now();
  timer = setInterval(() => {
    const s = Math.round((Date.now() - t0) / 1000);
    $("#elapsed").textContent = `${s} s`;
    if (s > 12 && $("#progress-message").textContent.startsWith("Connecting")) $("#coldstart").hidden = false;
  }, 500);

  const url = `${API}/api/rank/stream?topic=${encodeURIComponent(topic)}&k=${k}`;
  source = new EventSource(url);
  let finished = false;

  source.addEventListener("progress", (ev) => {
    const { stage, message } = JSON.parse(ev.data);
    $("#progress-message").textContent = message;
    $("#coldstart").hidden = true;
    markStep(stage);
  });
  source.addEventListener("result", (ev) => {
    finished = true;
    const result = JSON.parse(ev.data);
    stopSearch();
    show(result);
  });
  source.addEventListener("failure", (ev) => {
    finished = true;
    stopSearch();
    showError(JSON.parse(ev.data).message);
  });
  source.onerror = () => {
    if (finished) return;
    stopSearch();
    showError("Could not reach the ranking server. It may be waking up or over its rate limit. Wait a minute and retry, or open a precomputed example above.");
  };
}

function markStep(stage) {
  const ids = STEPS.map((s) => s[0]);
  const idx = stage === "done" ? ids.length : ids.indexOf(stage);
  if (idx < 0) return;
  document.querySelectorAll("#steps li").forEach((li, i) => {
    li.classList.toggle("done", i < idx);
    li.classList.toggle("active", i === idx);
  });
}

function showError(message) {
  const box = $("#error");
  box.innerHTML = `<strong>The search could not finish.</strong> ${esc(message)}`;
  box.hidden = false;
}

/* ---------- results ---------- */
function show(result) {
  current = result;
  const s = result.stats;
  const llm = result.mode === "llm";
  $("#results-title").innerHTML = `Top ${result.genes.length} genes for “${esc(result.topic)}”`;
  $("#results-meta").innerHTML = [
    llm ? `<span class="badge llm">LLM-reviewed · ${esc(result.model)}</span>` : `<span class="badge">Literature signals only</span>`,
    result._example ? `<span class="badge">Precomputed example</span>` : "",
    `Generated ${esc(new Date(result.created_at).toLocaleString())}`,
  ].filter(Boolean).join(" ");

  const stat = (v, l) => `<div class="stat"><b>${fmt(v)}</b><span>${l}</span></div>`;
  $("#stats").innerHTML =
    stat(s.abstracts_matched, "abstracts match the topic") +
    stat(s.abstracts_used, "most relevant analysed") +
    stat(s.human_genes_found, "human genes mentioned") +
    stat(s.llm_reviewed, `genes reviewed by LLM (${s.llm_calls} calls)`) +
    stat(s.network_edges, "candidate interactions") +
    stat(`${s.elapsed_s} s`, "pipeline time");

  const w = $("#warnings");
  w.hidden = !result.warnings.length;
  w.querySelector("summary").textContent = `Notes on this run (${result.warnings.length})`;
  w.querySelector("ul").innerHTML = result.warnings.map((x) => `<li>${esc(x)}</li>`).join("");

  const max = Math.max(...result.genes.map((g) => g.score), 1e-9);
  $("#gene-list").innerHTML = result.genes.map((g) => geneCard(g, max)).join("");
  const others = result.other_candidates || [];
  $("#others").hidden = !others.length;
  $("#others summary").textContent = `Other reviewed candidates (${others.length})`;
  $("#others tbody").innerHTML = others.map((g) => `<tr>
      <td class="mono">${g.rank}</td>
      <td><a class="mono" href="${esc(g.ncbi_url)}" target="_blank" rel="noopener" title="${esc(g.name)}">${esc(g.symbol)}</a></td>
      <td><span class="rel-tag ${g.relevance || "none"}">${g.relevance || "not reviewed"}</span></td>
      <td class="mono">${g.papers}</td><td class="mono">${g.score.toFixed(3)}</td>
      <td>${linkPmids(esc(g.rationale || ""))}</td></tr>`).join("");
  drawNetwork(result);
  $("#results").hidden = false;
  $("#results").scrollIntoView({ behavior: "smooth", block: "start" });
}

function geneCard(g, max) {
  const rel = g.relevance || "none";
  const segs = Object.entries(g.score_parts)
    .map(([k, v]) => `<span class="seg ${PART_CLASS[k]}" style="width:${(100 * v) / max}%" title="${PART_LABEL[k]}: ${v.toFixed(3)}"></span>`)
    .join("");
  return `<li class="gene" data-symbol="${esc(g.symbol)}">
    <button class="gene-row" type="button" aria-expanded="false">
      <span class="rank">${g.rank}</span>
      <span><span class="sym">${esc(g.symbol)}</span><span class="rel-tag ${rel}">${rel === "none" ? "not reviewed" : rel}</span>
        <span class="gname">${esc(g.name)}</span></span>
      <span class="bar" aria-hidden="true">${segs}</span>
      <span><span class="score">${g.score.toFixed(3)}</span><span class="papers">${g.papers} papers</span></span>
      <svg class="chev" viewBox="0 0 24 24" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg>
    </button>
    ${g.rationale ? `<p class="rationale">${linkPmids(esc(g.rationale))}</p>` : ""}
    <div class="detail" hidden>${geneDetail(g)}</div>
  </li>`;
}

function linkPmids(html) {
  return html.replace(/PMID (\d+)/g, '<a href="https://pubmed.ncbi.nlm.nih.gov/$1/" target="_blank" rel="noopener">PMID $1</a>');
}

function highlight(text, terms) {
  const unique = [...new Set(terms.filter((x) => x && x.length > 1).map((x) => x.toLowerCase()))];
  if (!unique.length) return esc(text);
  // One combined pass so overlapping terms never produce nested <mark> tags.
  const re = new RegExp(`(${unique.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`, "gi");
  return String(text).split(re).map((part, i) => (i % 2 ? `<mark>${esc(part)}</mark>` : esc(part))).join("");
}

function geneDetail(g) {
  const parts = Object.entries(g.score_parts).map(([k, v]) => `<span>${PART_LABEL[k]} <b>${v.toFixed(3)}</b></span>`).join("");
  const ev = g.evidence.map((e) => `<div class="ev">
      <span class="ev-label ${e.label || "none"}">${e.label || "not assessed"}</span>
      <div><p>${highlight(e.text, [e.mention, g.symbol])}</p>
      <a href="https://pubmed.ncbi.nlm.nih.gov/${esc(e.pmid)}/" target="_blank" rel="noopener">PubMed ${esc(e.pmid)} ↗</a></div>
    </div>`).join("");
  return `<div><h4>Score breakdown</h4><div class="parts">${parts}</div></div>
    <div><h4>Evidence sentences</h4>${ev || '<p class="muted small">No sentences returned.</p>'}</div>
    <div class="parts"><span>${g.mentions} mentions in ${g.papers} papers</span>
      <span>Interacts with: ${g.partners.length ? esc(g.partners.join(", ")) : "none among candidates"}</span>
      <a href="${esc(g.ncbi_url)}" target="_blank" rel="noopener">NCBI Gene ${esc(g.ncbi_id)} ↗</a></div>`;
}

/* ---------- network (circular layout, no dependencies) ---------- */
const REL_COLOR = { core: "--core", supporting: "--supporting", peripheral: "--peripheral", unrelated: "--unrelated" };

function drawNetwork(result) {
  const svg = $("#network");
  const genes = result.genes;
  const css = getComputedStyle(document.documentElement);
  if (genes.length < 2) {
    svg.innerHTML = `<text class="empty" x="180" y="180" text-anchor="middle">Not enough genes to draw.</text>`;
    return;
  }
  const cx = 180, cy = 180, r = genes.length > 24 ? 140 : 124;
  const max = Math.max(...genes.map((g) => g.score), 1e-9);
  const pos = {};
  genes.forEach((g, i) => {
    const a = (2 * Math.PI * i) / genes.length - Math.PI / 2;
    pos[g.symbol] = [cx + r * Math.cos(a), cy + r * Math.sin(a), a];
  });
  const edges = result.network
    .map(([a, b, s]) => `<line class="edge" x1="${pos[a][0]}" y1="${pos[a][1]}" x2="${pos[b][0]}" y2="${pos[b][1]}" stroke-opacity="${(0.15 + 0.5 * (s - 0.7) / 0.3).toFixed(2)}" stroke-width="${(0.8 + 1.6 * (s - 0.7) / 0.3).toFixed(2)}" data-a="${esc(a)}" data-b="${esc(b)}"/>`)
    .join("");
  const nodes = genes.map((g) => {
    const [x, y, a] = pos[g.symbol];
    const rad = 5 + 9 * (g.score / max);
    const fill = g.relevance ? css.getPropertyValue(REL_COLOR[g.relevance]).trim() : css.getPropertyValue("--line").trim();
    const dash = g.relevance ? "" : `stroke="${css.getPropertyValue("--muted").trim()}" stroke-dasharray="3 2" style="stroke-width:1.5"`;
    const lx = x + (rad + 4) * Math.cos(a), ly = y + (rad + 4) * Math.sin(a);
    const anchor = Math.cos(a) > 0.3 ? "start" : Math.cos(a) < -0.3 ? "end" : "middle";
    const dy = Math.sin(a) > 0.3 ? 10 : Math.sin(a) < -0.3 ? -3 : 4;
    return `<g class="node" data-symbol="${esc(g.symbol)}"><title>#${g.rank} ${esc(g.symbol)} · score ${g.score.toFixed(3)}</title>
      <circle cx="${x}" cy="${y}" r="${rad.toFixed(1)}" fill="${fill}" ${dash}/>
      <text x="${lx.toFixed(1)}" y="${(ly + dy).toFixed(1)}" text-anchor="${anchor}">${esc(g.symbol)}</text></g>`;
  }).join("");
  const empty = result.network.length ? "" : `<text class="empty" x="180" y="184" text-anchor="middle">No high-confidence interactions among these genes.</text>`;
  svg.innerHTML = `<g>${edges}</g><g>${nodes}</g>${empty}`;
}

function highlightGene(symbol, on) {
  document.querySelectorAll(`[data-symbol="${CSS.escape(symbol)}"]`).forEach((el) => el.classList.toggle("hl", on));
  document.querySelectorAll("#network .edge").forEach((e) => {
    if (e.dataset.a === symbol || e.dataset.b === symbol) e.style.stroke = on ? "var(--accent)" : "";
  });
}

/* ---------- downloads ---------- */
function download(name, type, text) {
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(new Blob([text], { type })), download: name });
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

function toCSV(result) {
  const cols = ["rank", "symbol", "name", "ncbi_id", "score", "relevance", "papers", "mentions", "degree", "rationale"];
  const q = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  return [cols.join(","), ...result.genes.map((g) => cols.map((c) => q(g[c])).join(","))].join("\n");
}

const slug = () => (current?.topic || "genes").toLowerCase().replace(/[^a-z0-9]+/g, "_");

/* ---------- wiring ---------- */
document.addEventListener("DOMContentLoaded", () => {
  if (CFG.REPO_URL) $("#repo-link").href = CFG.REPO_URL;
  checkServer();
  loadExamples().then(() => {
    // Shareable links: #example=file.json opens a saved result; ?topic=…&k=… runs a search.
    const hash = new URLSearchParams(location.hash.slice(1));
    const query = new URLSearchParams(location.search);
    if (hash.get("example")) openExample(hash.get("example"));
    else if (query.get("topic")) {
      $("#topic").value = query.get("topic");
      $("#k").value = query.get("k") || 10;
      $("#search-form").requestSubmit();
    }
  });

  $("#search-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const topic = $("#topic").value.trim().replace(/\s+/g, " ");
    const k = Math.min(50, Math.max(1, parseInt($("#k").value, 10) || 10));
    $("#k").value = k;
    if (topic) startSearch(topic, k);
  });
  document.querySelectorAll(".stepper button").forEach((b) =>
    b.addEventListener("click", () => {
      const k = $("#k");
      k.value = Math.min(50, Math.max(1, (parseInt(k.value, 10) || 10) + Number(b.dataset.step)));
    })
  );
  $("#gene-list").addEventListener("click", (ev) => {
    if (ev.target.closest("a")) return;
    const row = ev.target.closest(".gene-row");
    if (!row) return;
    const li = row.parentElement;
    const open = li.classList.toggle("open");
    row.setAttribute("aria-expanded", open);
    li.querySelector(".detail").hidden = !open;
  });
  for (const root of ["#gene-list", "#network"]) {
    $(root).addEventListener("mouseover", (ev) => { const el = ev.target.closest("[data-symbol]"); if (el) highlightGene(el.dataset.symbol, true); });
    $(root).addEventListener("mouseout", (ev) => { const el = ev.target.closest("[data-symbol]"); if (el) highlightGene(el.dataset.symbol, false); });
  }
  $("#network").addEventListener("click", (ev) => {
    const node = ev.target.closest(".node");
    if (!node) return;
    const li = document.querySelector(`#gene-list [data-symbol="${CSS.escape(node.dataset.symbol)}"]`);
    li?.scrollIntoView({ behavior: "smooth", block: "center" });
    if (li && !li.classList.contains("open")) li.querySelector(".gene-row").click();
  });
  $("#dl-csv").addEventListener("click", () => current && download(`${slug()}_genes.csv`, "text/csv", toCSV(current)));
  $("#dl-json").addEventListener("click", () => current && download(`${slug()}_genes.json`, "application/json", JSON.stringify(current, null, 2)));
});
