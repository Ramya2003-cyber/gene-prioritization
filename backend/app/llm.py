"""LLM evidence review through any OpenAI-compatible chat endpoint.

Defaults to Google Gemini's free tier (OpenAI-compatible endpoint). Groq, Mistral or
OpenRouter work by changing LLM_BASE_URL / LLM_MODEL / LLM_API_KEY.

The model reads the exact sentences in which each candidate gene is mentioned and
returns, per gene, a topic-relevance label, a per-passage evidence label and a short
rationale. Every item is validated against what was sent: unknown genes or passage IDs
and invalid labels are dropped and reported, never allowed to fail the whole search.
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import httpx

from .http import request

RELEVANCE = {"core": 1.0, "supporting": 0.6, "peripheral": 0.25, "unrelated": 0.0}
EVIDENCE = {"direct": 1.0, "indirect": 0.5, "background": 0.1}
GENES_PER_CALL = 12

SYSTEM_PROMPT = """You are a careful biomedical literature curator.
For a research topic you receive candidate human genes, each with sentences from PubMed
abstracts in which the gene is mentioned. Judge ONLY from the supplied sentences.

For each gene return:
- relevance: "core" (the gene is a central, well-evidenced player in the topic),
  "supporting" (clear experimental link but not central), "peripheral" (mentioned in
  passing, as a generic marker, or only loosely related) or "unrelated" (mention is
  incidental or the passages do not concern the topic).
- evidence: for each passage ID, "direct" (the sentence reports a finding linking this
  gene to the topic), "indirect" (link via another factor, association, or hypothesis)
  or "background" (generic mention, method, or no finding).
- rationale: at most 25 words, citing passage IDs in square brackets like [P3].

Generic hub genes (e.g. TP53, AKT1, TNF) should only be "core" if the passages show a
specific role in this topic. Return JSON only:
{"genes":[{"symbol":"...","relevance":"...","rationale":"...","evidence":[{"pid":"P1","label":"..."}]}]}"""


class LLMConfig:
    def __init__(self) -> None:
        self.api_key = os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
        self.base_url = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/")
        self.model = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
        self.max_calls = int(os.getenv("LLM_MAX_CALLS_PER_SEARCH", "4"))

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


def build_batches(genes: list[dict], size: int = GENES_PER_CALL) -> list[list[dict]]:
    """Assign stable passage IDs (P1, P2, …) per batch and split genes into batches."""
    batches = []
    for start in range(0, len(genes), size):
        counter = 0
        batch = []
        for gene in genes[start : start + size]:
            passages = []
            for ctx in gene["contexts"][:3]:
                counter += 1
                passages.append({"pid": f"P{counter}", "pmid": ctx["pmid"], "text": ctx["text"]})
            batch.append({"symbol": gene["symbol"], "name": gene["name"], "passages": passages})
        batches.append(batch)
    return batches


def parse_json(text: str) -> dict:
    """Parse a JSON object, tolerating ```json fences or text around the object."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def validate(batch: list[dict], payload: dict) -> tuple[dict[str, dict], list[str]]:
    """Keep only well-formed assessments of genes/passages that were actually sent."""
    issues: list[str] = []
    sent = {g["symbol"]: {p["pid"]: p for p in g["passages"]} for g in batch}
    accepted: dict[str, dict] = {}
    for item in payload.get("genes", []) if isinstance(payload, dict) else []:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).strip()
        if symbol not in sent:
            issues.append(f"Ignored assessment of unknown gene '{symbol[:30]}'.")
            continue
        if symbol in accepted:
            continue
        relevance = str(item.get("relevance", "")).lower()
        if relevance not in RELEVANCE:
            issues.append(f"{symbol}: invalid relevance label dropped.")
            continue
        evidence = []
        for ev in item.get("evidence", []) or []:
            pid, label = str(ev.get("pid", "")), str(ev.get("label", "")).lower()
            if pid in sent[symbol] and label in EVIDENCE and pid not in {e["pid"] for e in evidence}:
                evidence.append({"pid": pid, "label": label, "pmid": sent[symbol][pid]["pmid"]})
            else:
                issues.append(f"{symbol}: dropped evidence item with unknown passage or label.")
        rationale = " ".join(str(item.get("rationale", "")).split())[:300]
        # Citations in the rationale must point at this gene's passages; replace with PMIDs.
        def cite(m: re.Match) -> str:
            p = sent[symbol].get(m.group(1))
            return f"[PMID {p['pmid']}]" if p else ""
        rationale = re.sub(r"\[(P\d+)\]", cite, rationale).strip()
        accepted[symbol] = {"relevance": relevance, "rationale": rationale, "evidence": evidence}
    missing = [s for s in sent if s not in accepted]
    if missing:
        issues.append(f"Not assessed by the model: {', '.join(missing)}.")
    return accepted, issues


def _call(client: httpx.Client, config: LLMConfig, topic: str, batch: list[dict]) -> dict:
    user = json.dumps(
        {"topic": topic, "genes": [{k: v for k, v in g.items()} for g in batch]}, ensure_ascii=False
    )
    response = request(
        client,
        "POST",
        f"{config.base_url}/chat/completions",
        attempts=3,
        backoff=4.0,  # free tiers rate-limit; back off generously
        headers={"Authorization": f"Bearer {config.api_key}"},
        json={
            "model": config.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
        },
        timeout=90,
    )
    body = response.json()
    content = body["choices"][0]["message"]["content"]
    return {"payload": parse_json(content), "usage": body.get("usage", {})}


def review(client: httpx.Client, config: LLMConfig, topic: str, genes: list[dict]) -> dict:
    """Assess genes in a few batched calls (bounded by LLM_MAX_CALLS_PER_SEARCH)."""
    batches = build_batches(genes)[: config.max_calls]
    assessments: dict[str, dict] = {}
    issues: list[str] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    calls = failed = 0

    def run(batch):
        return batch, _call(client, config, topic, batch)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, b) for b in batches]
        for future in futures:
            calls += 1
            try:
                batch, out = future.result()
            except Exception as exc:  # one failed batch must not sink the search
                failed += 1
                issues.append(f"An LLM review batch failed ({type(exc).__name__}); those genes use literature signals only.")
                continue
            accepted, batch_issues = validate(batch, out["payload"])
            assessments.update(accepted)
            issues.extend(batch_issues)
            for key in usage:
                usage[key] += int(out["usage"].get(key) or 0)
    return {"assessments": assessments, "issues": issues, "calls": calls, "failed_calls": failed, "usage": usage}
