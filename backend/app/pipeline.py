"""End-to-end pipeline: topic + k -> ranked, evidence-backed human genes."""

from __future__ import annotations

import time
from collections import OrderedDict
from datetime import datetime, timezone
from threading import Lock

import httpx

from . import literature, llm, network, ranking

ARTICLE_LIMIT = 250
MAX_K = 50
_cache: "OrderedDict[tuple, tuple[float, dict]]" = OrderedDict()
_cache_lock = Lock()
CACHE_TTL = 24 * 3600


def _cache_get(key):
    with _cache_lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < CACHE_TTL:
            _cache.move_to_end(key)
            return hit[1]
    return None


def _cache_put(key, value):
    with _cache_lock:
        _cache[key] = (time.time(), value)
        while len(_cache) > 128:
            _cache.popitem(last=False)


def run(topic: str, k: int, progress=lambda stage, message: None, *, client: httpx.Client | None = None,
        config: llm.LLMConfig | None = None) -> dict:
    topic = literature.clean_topic(topic)
    if not isinstance(k, int) or not 1 <= k <= MAX_K:
        raise ValueError(f"k must be an integer from 1 to {MAX_K}.")
    # Tests pass an explicit config; the server picks Claude (while its search budget lasts) then Gemini.
    claude, gemini = llm.configs() if config is None else (None, config)
    llm_available = bool((claude and claude.enabled) or gemini.enabled)
    key = (topic.casefold(), k, "llm" if llm_available else "literature-only")
    cached = _cache_get(key)
    if cached:
        progress("done", "Loaded a result computed in the last 24 hours.")
        return cached

    started = time.time()
    own_client = client is None
    client = client or literature_client()
    warnings: list[str] = []
    try:
        found = literature.discover(client, topic, ARTICLE_LIMIT, progress)
        genes = found["genes"]
        if not genes:
            raise LookupError("No human genes were found in abstracts for this topic. Try a broader or more common term.")
        if found["total_hits"] > len(found["pmids"]):
            warnings.append(
                f"Used the {len(found['pmids'])} most relevant of {found['total_hits']:,} matching abstracts; "
                "a larger corpus could change lower ranks."
            )

        # Review more candidates than requested so the LLM can promote/demote around the cut.
        pool = ranking.shortlist(genes, min(len(genes), max(2 * k, k + 10, 20), 60))

        progress("network", f"Fetching STRING interactions among {len(pool)} candidate genes…")
        try:
            edges = network.fetch_network(client, [g["symbol"] for g in pool])
        except (httpx.HTTPError, ValueError) as exc:
            edges = []
            warnings.append(f"STRING network unavailable ({type(exc).__name__}); network bonus set to 0.")

        review = {"assessments": {}, "issues": [], "calls": 0, "failed_calls": 0, "usage": {}}
        candidates = []
        if claude is not None and claude.enabled and llm.claude_budget.try_acquire():
            candidates.append(claude)
        if gemini.enabled:
            candidates.append(gemini)
        used = None
        for cfg in candidates:
            to_review = pool[: llm.GENES_PER_CALL * cfg.max_calls]
            progress("llm", f"LLM ({cfg.model}) is reviewing evidence sentences for {len(to_review)} genes…")
            review = llm.review(client, cfg, topic, to_review)
            if review["assessments"]:
                used = cfg
                break
            warnings.append(f"The LLM review with {cfg.model} failed for every batch.")
        if used:
            warnings.extend(review["issues"][:8])
        elif candidates:
            warnings.append("Ranking uses literature and network signals only.")
        else:
            warnings.append("No LLM API key is configured on the server; ranking uses literature and network signals only.")

        progress("rank", "Combining literature, LLM and network signals…")
        ranked = ranking.score(pool, edges, review["assessments"])
        top = ranked[:k]
        if len(top) < k:
            warnings.append(f"Only {len(top)} verified human genes were found (requested {k}); no genes were invented.")
        top_symbols = {g["symbol"] for g in top}
        result = {
            "topic": topic,
            "k": k,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mode": "llm" if review["assessments"] else "literature-only",
            "model": used.model if used else None,
            "stats": {
                "abstracts_matched": found["total_hits"],
                "abstracts_used": len(found["pmids"]),
                "abstracts_annotated": found["annotated"],
                "human_genes_found": len(genes),
                "candidates_ranked": len(pool),
                "llm_reviewed": len(review["assessments"]),
                "llm_calls": review["calls"],
                "llm_tokens": sum(review["usage"].values()) if review["usage"] else 0,
                "network_edges": len(edges),
                "elapsed_s": round(time.time() - started, 1),
            },
            "weights": ranking.WEIGHTS,
            "genes": top,
            # Shortlisted genes that did not make the top k, so users can see what was demoted and why.
            "other_candidates": [
                {key: g[key] for key in ("rank", "symbol", "name", "score", "papers", "relevance", "rationale", "ncbi_url")}
                for g in ranked[k:]
            ],
            "network": [e for e in edges if e[0] in top_symbols and e[1] in top_symbols],
            "warnings": warnings,
        }
        _cache_put(key, result)
        progress("done", f"Ranked {len(top)} genes in {result['stats']['elapsed_s']} s.")
        return result
    finally:
        if own_client:
            client.close()


def literature_client() -> httpx.Client:
    from .http import make_client

    return make_client()
