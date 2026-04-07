from __future__ import annotations

import argparse
import ast
import collections
import difflib
import hashlib
import json
import math
import copy
import os
import random
import re
import subprocess
import shutil
import sys
import tempfile
import textwrap
import time
import traceback
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Set, Union
import multiprocessing as mp


def now_ms() -> int:
    return int(time.time() * 1000)

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_json(p: Path) -> Dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def write_json(p: Path, obj: Any, indent: int = 2):
    safe_mkdir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=indent, default=str), encoding="utf-8")

def unified_diff(old: str, new: str, name: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(True),
            new.splitlines(True),
            fromfile=name,
            tofile=name,
        )
    )
