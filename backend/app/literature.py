"""Topic -> publications -> annotated human gene mentions.

1. Europe PMC: relevance-ordered PubMed IDs whose title/abstract contain the topic.
2. PubTator3: NCBI Gene annotations for those abstracts (with the sentence they occur in).
3. MyGene.info: verify each NCBI Gene ID is a human gene and fetch its symbol/name.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field

import httpx

from .http import request

EUROPE_PMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
PUBTATOR = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson"
MYGENE = "https://mygene.info/v3/gene"
MAX_CONTEXTS_PER_GENE = 6


@dataclass
class Candidate:
    ncbi_id: str
    symbol: str = ""
    name: str = ""
    mentions: int = 0
    pmids: set[str] = field(default_factory=set)
    contexts: list[dict] = field(default_factory=list)  # {pmid, text, mention}


def clean_topic(topic: str) -> str:
    topic = " ".join(str(topic).split())
    if not topic or len(topic) > 200:
        raise ValueError("Enter a topic of 1-200 characters.")
    if re.search(r'["\\{}<>]', topic):
        raise ValueError("Please enter plain text without quotes or query syntax.")
    return topic


def search_pmids(client: httpx.Client, topic: str, limit: int) -> tuple[list[str], int]:
    """Return up to `limit` relevance-ordered PMIDs and the total hit count."""
    response = request(
        client,
        "GET",
        EUROPE_PMC,
        params={
            "query": f'TITLE_ABS:"{topic}" AND SRC:MED AND HAS_ABSTRACT:Y',
            "format": "json",
            "resultType": "idlist",  # IDs only: far faster than full records
            "pageSize": min(limit, 1000),
        },
    )
    body = response.json()
    results = body.get("resultList", {}).get("result", []) or []
    pmids: list[str] = []
    for row in results:
        pmid = str(row.get("pmid") or row.get("id") or "")
        if pmid.isdigit() and pmid not in pmids:
            pmids.append(pmid)
    return pmids[:limit], int(body.get("hitCount") or 0)


def _documents(payload) -> list[dict]:
    """PubTator BioC JSON may be a list, a {'PubTator3': [...]} wrapper or a collection."""
    if isinstance(payload, list):
        return [doc for item in payload for doc in _documents(item)]
    if isinstance(payload, dict):
        if "PubTator3" in payload:
            return _documents(payload["PubTator3"])
        if "documents" in payload:
            return _documents(payload["documents"])
        if "passages" in payload:
            return [payload]
    return []


def _sentence(text: str, start: int, end: int) -> str:
    """The sentence containing [start, end), trimmed to a readable length."""
    left = max(text.rfind(". ", 0, start), text.rfind("? ", 0, start), text.rfind("; ", 0, start))
    left = 0 if left < 0 else left + 2
    right_candidates = [i for i in (text.find(". ", end), text.find("? ", end)) if i >= 0]
    right = min(right_candidates) + 1 if right_candidates else len(text)
    sentence = text[left:right].strip()
    if len(sentence) > 420:  # keep the mention in view
        lo = max(0, start - left - 200)
        sentence = ("…" if lo else "") + sentence[lo : lo + 400] + "…"
    return sentence


def annotate(client: httpx.Client, pmids: list[str], sleep=None) -> tuple[dict[str, Candidate], int]:
    """Collect gene mentions per NCBI Gene ID. Returns candidates and #annotated articles."""
    import time

    pause = sleep or time.sleep
    candidates: dict[str, Candidate] = {}
    annotated = 0
    seen: set[tuple] = set()
    for start in range(0, len(pmids), 100):
        if start:
            pause(0.35)  # PubTator asks clients to stay under ~3 requests/second
        response = request(client, "GET", PUBTATOR, params={"pmids": ",".join(pmids[start : start + 100])})
        try:
            body = response.json()
        except (ValueError, json.JSONDecodeError):
            body = [json.loads(line) for line in response.text.splitlines() if line.strip()]
        for doc in _documents(body):
            pmid = str(doc.get("id") or doc.get("pmid") or "")
            if not pmid:
                continue
            annotated += 1
            for passage in doc.get("passages", []):
                text = passage.get("text") or ""
                base = int(passage.get("offset") or 0)
                for ann in passage.get("annotations", []):
                    info = ann.get("infons", {})
                    if str(info.get("type", "")).lower() != "gene":
                        continue
                    gene = str(info.get("identifier") or "")
                    if not gene.isdigit():  # skip ambiguous multi-ID annotations
                        continue
                    locs = ann.get("locations") or []
                    key = (pmid, gene, tuple((l.get("offset"), l.get("length")) for l in locs))
                    if key in seen:
                        continue
                    seen.add(key)
                    cand = candidates.setdefault(gene, Candidate(ncbi_id=gene))
                    cand.mentions += 1
                    first_in_paper = pmid not in cand.pmids
                    cand.pmids.add(pmid)
                    # Prefer one context per distinct paper, so evidence spans studies.
                    if text and locs and first_in_paper and len(cand.contexts) < MAX_CONTEXTS_PER_GENE:
                        s = int(locs[0].get("offset", base)) - base
                        e = s + int(locs[0].get("length", 0))
                        cand.contexts.append(
                            {"pmid": pmid, "text": _sentence(text, max(0, s), max(0, e)), "mention": ann.get("text") or ""}
                        )
    return candidates, annotated


def verify_human(client: httpx.Client, candidates: dict[str, Candidate]) -> dict[str, Candidate]:
    """Keep only IDs that MyGene confirms are human (taxid 9606); merge by symbol."""
    ids = sorted(candidates)
    by_symbol: dict[str, Candidate] = {}
    for start in range(0, len(ids), 1000):
        response = request(
            client,
            "POST",
            MYGENE,
            data={"ids": ",".join(ids[start : start + 1000]), "fields": "symbol,name,taxid", "species": "all"},
        )
        for row in response.json():
            gene = str(row.get("query") or row.get("_id") or "")
            if row.get("notfound") or row.get("taxid") != 9606 or not row.get("symbol") or gene not in candidates:
                continue
            cand = candidates[gene]
            cand.symbol, cand.name = row["symbol"], row.get("name", "")
            if cand.symbol in by_symbol:  # rare: two IDs share a symbol
                other = by_symbol[cand.symbol]
                other.mentions += cand.mentions
                other.pmids |= cand.pmids
                other.contexts = (other.contexts + cand.contexts)[:MAX_CONTEXTS_PER_GENE]
            else:
                by_symbol[cand.symbol] = cand
    return by_symbol


def discover(client: httpx.Client, topic: str, article_limit: int, progress=lambda *_: None) -> dict:
    progress("search", f"Searching Europe PMC for “{topic}”…")
    pmids, total = search_pmids(client, topic, article_limit)
    progress("annotate", f"Found {total:,} matching abstracts; reading gene annotations in the top {len(pmids)}…")
    raw, annotated = annotate(client, pmids) if pmids else ({}, 0)
    progress("verify", f"Verifying {len(raw)} gene identifiers as human genes…")
    genes = verify_human(client, raw) if raw else {}
    return {"pmids": pmids, "total_hits": total, "annotated": annotated, "genes": genes}
