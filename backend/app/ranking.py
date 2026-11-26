"""Transparent, additive gene scoring.

score = 0.40 · literature  +  0.40 · LLM relevance  +  0.15 · LLM evidence  +  0.05 · network

* literature: log paper breadth (plus a little mention volume), scaled 0–1 within the topic.
* LLM relevance: core 1.0 / supporting 0.6 / peripheral 0.25 / unrelated 0.
* LLM evidence: mean passage label (direct 1.0 / indirect 0.5 / background 0.1).
* network: STRING degree among the candidates, scaled 0–1, deliberately a small bonus so
  highly connected "hub" genes cannot outrank genes the literature is actually about.

When the LLM is unavailable, relevance/evidence are 0 for every gene and the ranking
falls back to literature + network (the mode is reported to the user).
"""

from __future__ import annotations

import math

from .llm import EVIDENCE, RELEVANCE

WEIGHTS = {"literature": 0.40, "llm_relevance": 0.40, "llm_evidence": 0.15, "network": 0.05}


def literature_signal(papers: int, mentions: int) -> float:
    return math.log1p(papers) + 0.25 * math.log1p(mentions)


def shortlist(genes: dict, size: int) -> list[dict]:
    """Order verified candidates by literature signal and keep the top `size`."""
    rows = [
        {
            "symbol": c.symbol,
            "name": c.name,
            "ncbi_id": c.ncbi_id,
            "papers": len(c.pmids),
            "mentions": c.mentions,
            "contexts": c.contexts,
        }
        for c in genes.values()
    ]
    rows.sort(key=lambda r: (-literature_signal(r["papers"], r["mentions"]), -r["papers"], r["symbol"]))
    return rows[:size]


def score(rows: list[dict], edges: list[list], assessments: dict[str, dict]) -> list[dict]:
    degree = {r["symbol"]: 0 for r in rows}
    partners = {r["symbol"]: [] for r in rows}
    for a, b, _ in edges:
        if a in degree and b in degree:
            degree[a] += 1
            degree[b] += 1
            partners[a].append(b)
            partners[b].append(a)
    lit = {r["symbol"]: literature_signal(r["papers"], r["mentions"]) for r in rows}
    lit_max = max(lit.values(), default=1) or 1
    deg_max = max(degree.values(), default=1) or 1

    out = []
    for r in rows:
        a = assessments.get(r["symbol"])
        labels = [EVIDENCE[e["label"]] for e in a["evidence"]] if a else []
        parts = {
            "literature": WEIGHTS["literature"] * lit[r["symbol"]] / lit_max,
            "llm_relevance": WEIGHTS["llm_relevance"] * (RELEVANCE[a["relevance"]] if a else 0.0),
            "llm_evidence": WEIGHTS["llm_evidence"] * (sum(labels) / len(labels) if labels else 0.0),
            "network": WEIGHTS["network"] * degree[r["symbol"]] / deg_max,
        }
        label_by_pmid = {e["pmid"]: e["label"] for e in a["evidence"]} if a else {}
        out.append(
            {
                "symbol": r["symbol"],
                "name": r["name"],
                "ncbi_id": r["ncbi_id"],
                "ncbi_url": f"https://www.ncbi.nlm.nih.gov/gene/{r['ncbi_id']}",
                "papers": r["papers"],
                "mentions": r["mentions"],
                "degree": degree[r["symbol"]],
                "partners": sorted(partners[r["symbol"]]),
                "score": round(sum(parts.values()), 4),
                "score_parts": {k: round(v, 4) for k, v in parts.items()},
                "relevance": a["relevance"] if a else None,
                "rationale": a["rationale"] if a else "",
                "evidence": [
                    {"pmid": c["pmid"], "text": c["text"], "mention": c.get("mention", ""), "label": label_by_pmid.get(c["pmid"])}
                    for c in r["contexts"]
                ],
            }
        )
    out.sort(key=lambda g: (-g["score"], -g["papers"], g["symbol"]))
    for rank, g in enumerate(out, 1):
        g["rank"] = rank
    return out
