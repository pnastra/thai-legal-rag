"""Settings loaded from a local .env file (never committed)."""

import os
from pathlib import Path


def load_env(path: str | Path = ".env") -> None:
    """Read KEY=VALUE lines from `path` into os.environ.

    Existing environment variables win, blank lines and # comments are skipped.
    A missing file is fine (e.g. in CI, where variables come from secrets).
    """
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
