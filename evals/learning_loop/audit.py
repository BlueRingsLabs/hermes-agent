"""Per-call audit from Hermes' own log lines: who served each call, its tokens, and what the review fork did.

Hermes logs one INFO line per API call (``agent/turn_usage.py``, logger ``agent.conversation_loop``) naming the
upstream provider OpenRouter reports, and one line per finished background review (``agent/background_review.py``)
with the fork's own calls and tokens. The episode worker attaches :class:`CaptureHandler` to both loggers before
the run, so the audit does not depend on how a programmatic session configures logging, and keeps only those
two kinds of line: neither carries message content or credentials.
"""
from __future__ import annotations

import logging
import re

LOGGERS = ("agent.conversation_loop", "agent.background_review")
REVIEW_THREAD = "bg-review"
_CALL = re.compile(
    r"API call #\d+: model=(?P<model>\S+) provider=\S+ in=(?P<inp>\d+) out=(?P<out>\d+) total=\d+ latency=[\d.]+s"
    r"(?: cache=(?P<cache>\d+)/\d+ \(\d+%\))?(?: write=(?P<write>\d+))?(?: id=\S+)?(?: upstream=(?P<upstream>.+))?$")
_REVIEW = re.compile(
    r"Background review complete: thread=bg-review calls=(?P<calls>\d+) in=(?P<inp>\d+) out=(?P<out>\d+) "
    r"cache_read=(?P<cache>\d+) result=(?P<result>\S+)")
_KEEP = ("API call #", "Background review complete:")


class CaptureHandler(logging.Handler):
    """Keeps (thread name, message) for the two audited line kinds only."""

    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.lines: list[tuple[str, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if message.startswith(_KEEP):
            self.lines.append((record.threadName, message))


def attach() -> CaptureHandler:
    handler = CaptureHandler()
    for name in LOGGERS:
        target = logging.getLogger(name)
        target.addHandler(handler)
        if target.getEffectiveLevel() > logging.INFO:
            target.setLevel(logging.INFO)
    return handler


def _call(match: re.Match, thread: str) -> dict:
    as_int = lambda key: int(match.group(key) or 0)
    return {"role": "review" if thread.startswith(REVIEW_THREAD) else "main", "model": match.group("model"),
            "input": as_int("inp"), "output": as_int("out"), "cache_read": as_int("cache"),
            "cache_write": as_int("write"), "upstream": (match.group("upstream") or "").strip()}


def summarise(lines: list[tuple[str, str]], expected_upstream: str) -> dict:
    """Calls by role, token totals, the upstreams seen, and whether every call stayed on the pinned route."""
    calls = [_call(m, thread) for thread, line in lines if (m := _CALL.search(line))]
    reviews = [{key: (m.group(key) if key == "result" else int(m.group(key)))
                for key in ("calls", "inp", "out", "cache", "result")}
               for _, line in lines if (m := _REVIEW.search(line))]
    totals = {role: {k: sum(c[k] for c in calls if c["role"] == role)
                     for k in ("input", "output", "cache_read", "cache_write")} for role in ("main", "review")}
    upstreams = sorted({c["upstream"] for c in calls})
    return {"calls_main": sum(c["role"] == "main" for c in calls),
            "calls_review": sum(c["role"] == "review" for c in calls),
            "tokens": totals, "upstreams": upstreams,
            "off_route": [u for u in upstreams if u != expected_upstream],
            "route_ok": bool(calls) and upstreams == [expected_upstream],
            "reviews": reviews}
