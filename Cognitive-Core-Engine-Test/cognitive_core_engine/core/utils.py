"""Standalone utility functions extracted from NON_RSI_AGI_CORE_v5."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, List


def stable_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def now_ms() -> int:
    return int(time.time() * 1000)


def tokenize(text: str) -> List[str]:
    text = text.lower()
    buf: List[str] = []
    cur: List[str] = []
    for ch in text:
        if ch.isalnum() or ch in ("_", "-"):
            cur.append(ch)
        else:
            if cur:
                buf.append("".join(cur))
                cur = []
    if cur:
        buf.append("".join(cur))
    return buf
