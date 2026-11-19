"""ISO timestamp parsing consistent across Python runtimes."""

import re
from datetime import datetime


def parse_iso_datetime(value: str) -> datetime:
    if not isinstance(value, str):
        raise TypeError("ISO timestamp must be a string")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    text = re.sub(r"([+-][0-9]{2})([0-9]{2})$", r"\1:\2", text)
    return datetime.fromisoformat(text)
