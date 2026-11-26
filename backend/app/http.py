"""HTTP helper with bounded retries for free public APIs (Europe PMC, PubTator, MyGene, STRING).

Public bioinformatics services regularly return transient 429/5xx responses or stall.
A single hiccup should not fail a whole search, so we retry a few times with
exponential backoff and honour Retry-After when the server sends it.
"""

from __future__ import annotations

import time

import httpx

RETRY_STATUS = {429, 500, 502, 503, 504}
USER_AGENT = "gene-prioritizer/1.0 (+https://github.com/)"


def make_client(timeout: float = 45.0) -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(timeout, connect=15.0),
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    )


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def request(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    attempts: int = 4,
    backoff: float = 1.5,
    sleep=time.sleep,
    **kwargs,
) -> httpx.Response:
    """Send a request, retrying transient failures. Raises on the final failure."""
    for attempt in range(attempts):
        last = attempt == attempts - 1
        try:
            response = client.request(method, url, **kwargs)
        except (httpx.TimeoutException, httpx.TransportError):
            if last:
                raise
            sleep(backoff * 2**attempt)
            continue
        if response.status_code in RETRY_STATUS and not last:
            sleep(min(_retry_after(response) or backoff * 2**attempt, 20.0))
            continue
        response.raise_for_status()
        return response
    raise RuntimeError("unreachable")  # pragma: no cover
