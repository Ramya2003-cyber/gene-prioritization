"""STRING protein-interaction network among the shortlisted candidate genes."""

from __future__ import annotations

import httpx

from .http import request

STRING = "https://string-db.org/api/json/network"


def fetch_network(client: httpx.Client, symbols: list[str], min_score: float = 0.7) -> list[list]:
    """Return de-duplicated [a, b, score] edges between the given human gene symbols."""
    if len(symbols) < 2:
        return []
    response = request(
        client,
        "POST",
        STRING,
        data={
            "identifiers": "\r".join(sorted(symbols)),
            "species": 9606,
            "required_score": int(min_score * 1000),
            "caller_identity": "gene_prioritizer",
        },
    )
    rows = response.json()
    if not isinstance(rows, list):
        return []
    wanted = set(symbols)
    edges: dict[tuple[str, str], float] = {}
    for row in rows:
        a, b = row.get("preferredName_A"), row.get("preferredName_B")
        try:
            score = float(row.get("score") or 0)
        except (TypeError, ValueError):
            continue
        if a in wanted and b in wanted and a != b and score >= min_score:
            pair = tuple(sorted((a, b)))
            edges[pair] = max(edges.get(pair, 0.0), score)
    return [[a, b, round(s, 3)] for (a, b), s in sorted(edges.items())]
