"""Cross-process JSON GET cache for the external odds sources.

Up to four strategy profiles poll the same sportsbook endpoints (Pinnacle,
FanDuel, Action Network). With POLYBOT_SHARED_DIR set, a response fetched by
one process is reused by the others until it is older than the CALLER's
max_age, so in-play polling at ~20 s costs one request per URL per window
instead of one per profile. Pregame callers pass a long max_age and read the
same files. Without the env var this is a plain in-process cache.

Failures return None (callers treat that as "no quotes"); a stale response is
never returned past the caller's max_age.
"""

import hashlib
import json
import os
import tempfile
import time
from typing import Any, Dict, Optional

import requests

_memo: Dict[str, tuple] = {}


def _key(url: str, params: Optional[dict]) -> str:
    raw = url + "?" + json.dumps(params or {}, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode()).hexdigest()


def get_json(url: str, params: Optional[dict] = None, headers: Optional[dict] = None,
             max_age: float = 60.0, timeout: float = 15.0) -> Optional[Any]:
    key = _key(url, params)
    now = time.time()
    hit = _memo.get(key)
    if hit and now - hit[0] <= max_age:
        return hit[1]
    shared = os.environ.get("POLYBOT_SHARED_DIR")
    path = os.path.join(shared, "http", key + ".json") if shared else None
    if path:
        try:
            mtime = os.path.getmtime(path)
            if now - mtime <= max_age:
                with open(path) as fh:
                    data = json.load(fh)
                _memo[key] = (mtime, data)
                return data
        except (OSError, ValueError):
            pass
    try:
        resp = requests.get(url, params=params, headers=headers or {}, timeout=timeout)
        if resp.status_code != 200:
            return None
        data = json.loads(resp.text, strict=False)   # FanDuel emits raw control chars
    except Exception:
        return None
    _memo[key] = (now, data)
    if path:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".h-")
            with os.fdopen(fd, "w") as fh:
                json.dump(data, fh)
            os.replace(tmp, path)
        except OSError:
            pass
    return data
