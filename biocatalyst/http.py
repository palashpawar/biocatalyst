"""Shared HTTP session with per-host rate limiting and retries."""
from __future__ import annotations

import time
from urllib.parse import urlparse

import requests

from .config import HTTP_UA, MAX_RETRIES, RATE_LIMIT, REQUEST_TIMEOUT, SEC_USER_AGENT

_last_hit: dict[str, float] = {}
_session = requests.Session()


def _host_key(url: str) -> str:
    host = urlparse(url).netloc.lower()
    for key in RATE_LIMIT:
        if host.endswith(key):
            return key
    return host


def _wait(url: str) -> None:
    key = _host_key(url)
    delay = RATE_LIMIT.get(key, 0.5)
    elapsed = time.monotonic() - _last_hit.get(key, 0.0)
    if elapsed < delay:
        time.sleep(delay - elapsed)
    _last_hit[key] = time.monotonic()


def get(url: str, **kwargs) -> requests.Response:
    """GET with rate limiting and exponential backoff on 429/5xx."""
    headers = {"User-Agent": SEC_USER_AGENT if "sec.gov" in url else HTTP_UA}
    headers.update(kwargs.pop("headers", {}))
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        _wait(url)
        try:
            resp = _session.get(url, headers=headers, timeout=REQUEST_TIMEOUT, **kwargs)
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(2**attempt)
                continue
            # Client errors are permanent. Retrying a 404 just multiplies the
            # wait for every company that has no XBRL facts filed.
            if 400 <= resp.status_code < 500:
                resp.raise_for_status()
            resp.raise_for_status()
            return resp
        except requests.HTTPError as exc:
            if exc.response is not None and 400 <= exc.response.status_code < 500:
                raise
        except requests.RequestException as exc:  # pragma: no cover - network
            last_exc = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"GET failed after {MAX_RETRIES} attempts: {url}") from last_exc
