"""Offline tests: every external API (literature services and the LLM) is mocked."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app import http, llm, pipeline, ranking
from app.literature import Candidate, _sentence, clean_topic


# ---------- http retries ----------
def test_retries_transient_503_then_succeeds():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(503) if len(calls) < 3 else httpx.Response(200, json={"ok": True})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    r = http.request(client, "GET", "https://x.test", sleep=lambda s: None)
    assert r.json() == {"ok": True} and len(calls) == 3


def test_gives_up_after_attempts():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    with pytest.raises(httpx.HTTPStatusError):
        http.request(client, "GET", "https://x.test", attempts=2, sleep=lambda s: None)


# ---------- literature helpers ----------
def test_topic_validation():
    assert clean_topic("  ferroptosis   cancer ") == "ferroptosis cancer"
    for bad in ["", 'a"b', "x" * 201]:
        with pytest.raises(ValueError):
            clean_topic(bad)


def test_sentence_extraction_keeps_mention():
    text = "First sentence. GPX4 loss triggers ferroptosis in cells. Last one."
    s = text.index("GPX4")
    assert _sentence(text, s, s + 4) == "GPX4 loss triggers ferroptosis in cells."


# ---------- LLM validation ----------
GENES = [
    {"symbol": "GPX4", "name": "glutathione peroxidase 4", "contexts": [{"pmid": "1", "text": "GPX4 ..."}, {"pmid": "2", "text": "GPX4 ..."}]},
    {"symbol": "TP53", "name": "tumor protein p53", "contexts": [{"pmid": "3", "text": "TP53 ..."}]},
]


def test_validation_drops_bad_items_but_keeps_good_ones():
    batch = llm.build_batches(GENES)[0]
    payload = {"genes": [
        {"symbol": "GPX4", "relevance": "core", "rationale": "Key regulator [P1].",
         "evidence": [{"pid": "P1", "label": "direct"}, {"pid": "P9", "label": "direct"}]},
        {"symbol": "MADEUP", "relevance": "core", "evidence": []},
        {"symbol": "TP53", "relevance": "very-important", "evidence": []},
    ]}
    accepted, issues = llm.validate(batch, payload)
    assert set(accepted) == {"GPX4"}
    assert accepted["GPX4"]["evidence"] == [{"pid": "P1", "label": "direct", "pmid": "1"}]
    assert accepted["GPX4"]["rationale"] == "Key regulator [PMID 1]."
    assert any("MADEUP" in i for i in issues) and any("TP53" in i for i in issues)


def test_parse_json_tolerates_fences():
    assert llm.parse_json('```json\n{"genes": []}\n```') == {"genes": []}
    assert llm.parse_json('Here you go: {"genes": []} thanks') == {"genes": []}


def test_review_survives_a_failed_batch(monkeypatch):
    config = llm.LLMConfig()
    config.api_key, config.max_calls = "test", 4
    genes = [{"symbol": f"G{i}", "name": "", "contexts": [{"pmid": str(i), "text": "x"}]} for i in range(20)]
    calls = {"n": 0}

    def fake_call(client, cfg, topic, batch):
        calls["n"] += 1
        if batch[0]["symbol"] == "G0":
            raise httpx.ReadTimeout("slow")
        return {"payload": {"genes": [{"symbol": g["symbol"], "relevance": "supporting", "evidence": []} for g in batch]},
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

    monkeypatch.setattr(llm, "_call", fake_call)
    out = llm.review(None, config, "t", genes)
    assert out["failed_calls"] == 1 and len(out["assessments"]) == 20 - llm.GENES_PER_CALL


# ---------- ranking ----------
def _rows():
    return [
        {"symbol": "GPX4", "name": "", "ncbi_id": "2879", "papers": 54, "mentions": 121, "contexts": []},
        {"symbol": "TP53", "name": "", "ncbi_id": "7157", "papers": 5, "mentions": 10, "contexts": []},
        {"symbol": "CTNNB1", "name": "", "ncbi_id": "1499", "papers": 1, "mentions": 1, "contexts": []},
    ]


def test_network_hubs_cannot_outrank_literature_focus():
    hub_edges = [["TP53", "CTNNB1", 0.99], ["TP53", "GPX4", 0.9]] + [["CTNNB1", "GPX4", 0.8]]
    ranked = ranking.score(_rows(), hub_edges, {})
    assert [g["symbol"] for g in ranked] == ["GPX4", "TP53", "CTNNB1"]


def test_llm_relevance_reorders_candidates():
    assessments = {"TP53": {"relevance": "core", "rationale": "", "evidence": []},
                   "GPX4": {"relevance": "unrelated", "rationale": "", "evidence": []}}
    ranked = ranking.score(_rows(), [], assessments)
    assert ranked[0]["symbol"] == "TP53"
    assert sum(ranked[0]["score_parts"].values()) == pytest.approx(ranked[0]["score"], abs=1e-3)


# ---------- pipeline + API ----------
@pytest.fixture
def fake_discovery(monkeypatch):
    def discover(client, topic, limit, progress):
        progress("search", "searching")
        genes = {}
        for i, (sym, papers) in enumerate([("GPX4", 30), ("SLC7A11", 20), ("ACSL4", 10)]):
            c = Candidate(ncbi_id=str(i + 1), symbol=sym, name=sym.lower(), mentions=papers * 2)
            c.pmids = {f"{i}{p}" for p in range(papers)}
            c.contexts = [{"pmid": f"{i}0", "text": f"{sym} drives ferroptosis.", "mention": sym}]
            genes[sym] = c
        return {"pmids": ["1"] * 10, "total_hits": 500, "annotated": 10, "genes": genes}

    monkeypatch.setattr(pipeline.literature, "discover", discover)
    monkeypatch.setattr(pipeline.network, "fetch_network", lambda client, syms: [["GPX4", "SLC7A11", 0.9]])
    pipeline._cache.clear()


def test_pipeline_literature_only_mode(fake_discovery, monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = pipeline.run("ferroptosis", 2, client=httpx.Client())
    assert result["mode"] == "literature-only"
    assert [g["symbol"] for g in result["genes"]] == ["GPX4", "SLC7A11"]
    assert result["network"] == [["GPX4", "SLC7A11", 0.9]]
    assert "C:\\" not in json.dumps(result)  # no local paths leak to the public API


def test_pipeline_reports_shortfall(fake_discovery, monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = pipeline.run("ferroptosis", 5, client=httpx.Client())
    assert len(result["genes"]) == 3 and any("Only 3" in w for w in result["warnings"])


def test_stream_endpoint_emits_progress_and_result(fake_discovery, monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    from app.main import app

    client = TestClient(app)
    body = client.get("/api/rank/stream", params={"topic": "ferroptosis", "k": 2}).text
    assert "event: progress" in body and "event: result" in body
    bad = client.get("/api/rank/stream", params={"topic": 'bad"topic', "k": 2}).text
    assert "event: failure" in bad
    assert client.get("/api/health").json()["status"] == "ok"
