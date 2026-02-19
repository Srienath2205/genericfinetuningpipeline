"""
prepare_dataset.py
Generic dataset validator / light normalizer for message-based SFT.

Each JSONL line must be:
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
"""

from typing import Iterable, Dict, Any, Tuple
from pathlib import Path
import json


REQUIRED_ROLES: Tuple[str, str] = ("user", "assistant")


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield i, json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"[{path}] JSON parse error on line {i}: {e}") from e


def _validate_messages_struct(rec: Dict[str, Any], line_no: int, path: Path) -> None:
    if "messages" not in rec or not isinstance(rec["messages"], list):
        raise ValueError(f"[{path}] line {line_no}: missing or invalid 'messages' list")

    roles_present = set()
    for m in rec["messages"]:
        if not isinstance(m, dict):
            raise ValueError(f"[{path}] line {line_no}: each message must be an object")
        if "role" not in m or "content" not in m:
            raise ValueError(f"[{path}] line {line_no}: each message must have 'role' and 'content'")
        if not isinstance(m["role"], str) or not isinstance(m["content"], str):
            raise ValueError(f"[{path}] line {line_no}: 'role' and 'content' must be strings")
        roles_present.add(m["role"])

    # Require at least one user and one assistant message
    for r in REQUIRED_ROLES:
        if r not in roles_present:
            raise ValueError(f"[{path}] line {line_no}: required role '{r}' not found in 'messages'")


def validate_or_raise(path_str: str) -> None:
    """
    Validates the JSONL dataset. Raises ValueError with a helpful message if invalid.
    Returns None if the file is valid.
    """
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p}")
    if p.suffix.lower() not in (".jsonl", ".json"):
        # we allow .json too for convenience, but prefer jsonl
        pass

    count = 0
    for line_no, rec in _iter_jsonl(p):
        _validate_messages_struct(rec, line_no, p)
        count += 1

    if count == 0:
        raise ValueError(f"[{p}] contains 0 valid records")

    print(f"[OK] {p} validated with {count} records")