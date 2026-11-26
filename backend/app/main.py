"""FastAPI backend for the gene prioritization app.

GET  /api/health              -> service status and configured LLM
GET  /api/rank/stream?topic=&k= -> Server-Sent Events: progress…, then result (or error)
POST /api/rank {topic, k}     -> the same result as plain JSON
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from collections import defaultdict, deque

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()

from . import pipeline  # noqa: E402
from . import llm  # noqa: E402

app = FastAPI(title="Gene Prioritizer API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Free public APIs + a free-tier LLM: keep usage polite and bounded.
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_HOUR", "20"))
_slots = threading.BoundedSemaphore(int(os.getenv("MAX_CONCURRENT_SEARCHES", "2")))
_hits: dict[str, deque] = defaultdict(deque)
_hits_lock = threading.Lock()


class RankRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=200)
    k: int = Field(default=10, ge=1, le=pipeline.MAX_K)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    return forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "?")


def _check_rate(request: Request) -> None:
    now, ip = time.time(), _client_ip(request)
    with _hits_lock:
        hits = _hits[ip]
        while hits and now - hits[0] > 3600:
            hits.popleft()
        if len(hits) >= RATE_LIMIT:
            raise HTTPException(429, "Hourly search limit reached. Please try again later or use an example topic.")
        hits.append(now)


@app.get("/")
@app.get("/api/health")
def health() -> dict:
    model = llm.next_model()
    return {"status": "ok", "llm_enabled": model is not None, "model": model,
            "claude_searches_remaining": llm.claude_budget.remaining if llm.configs()[0].enabled else 0,
            "max_k": pipeline.MAX_K}


def _run_guarded(topic: str, k: int, progress) -> dict:
    if not _slots.acquire(timeout=30):
        raise HTTPException(503, "The server is busy with other searches. Please retry in a minute.")
    try:
        return pipeline.run(topic, k, progress)
    finally:
        _slots.release()


@app.post("/api/rank")
def rank(body: RankRequest, request: Request) -> dict:
    _check_rate(request)
    try:
        return _run_guarded(body.topic, body.k, lambda *_: None)
    except (ValueError, LookupError) as exc:
        raise HTTPException(422, str(exc)) from exc


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/api/rank/stream")
def rank_stream(topic: str, request: Request, k: int = 10) -> StreamingResponse:
    _check_rate(request)
    events: queue.Queue = queue.Queue()

    def worker():
        try:
            result = _run_guarded(topic, k, lambda stage, msg: events.put(("progress", {"stage": stage, "message": msg})))
            events.put(("result", result))
        except HTTPException as exc:
            events.put(("failure", {"message": exc.detail}))
        except (ValueError, LookupError) as exc:
            events.put(("failure", {"message": str(exc)}))
        except Exception as exc:  # surface a readable message, never a stack trace
            events.put(("failure", {"message": f"A literature service did not respond ({type(exc).__name__}). Please try again."}))
        finally:
            events.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def stream():
        yield ": connected\n\n"
        while True:
            try:
                item = events.get(timeout=15)
            except queue.Empty:
                yield ": keep-alive\n\n"  # stops proxies from closing an idle stream
                continue
            if item is None:
                return
            yield _sse(*item)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
