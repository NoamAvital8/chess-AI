"""Append-only JSONL training records, flushed after every event."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class MetricsWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = uuid4().hex

    def write(self, event: str, **fields) -> None:
        record = {"schema_version": 1, "run_id": self.run_id,
                  "timestamp": datetime.now(timezone.utc).isoformat(),
                  "event": event, **fields}
        # Closing each append flushes it, including between long self-play games.
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")


def replay_summary(replay) -> dict:
    decisive = sum(ex[3] in (-1.0, 1.0) for ex in replay)
    draws = sum(ex[3] == 0.0 for ex in replay)
    return {"positions": len(replay), "decisive": decisive, "draws": draws,
            "unlabeled": len(replay) - decisive - draws,
            "value_label_fraction": (decisive + draws) / len(replay) if replay else 0.0}
