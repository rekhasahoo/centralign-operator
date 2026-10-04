"""Append-only event store. Events are never updated or deleted."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime

from pydantic import BaseModel

from .schemas import DDL, Event, EventType


class ConcurrencyError(RuntimeError):
    """Two writers raced for the same (run_id, seq)."""


class EventStore:
    def __init__(self, path: str = ":memory:"):
        self._c = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._c.executescript(DDL)
        self._lock = threading.Lock()

    def append(self, run_id: str, type: EventType, payload: BaseModel) -> Event:
        with self._lock:
            (last,) = self._c.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM events WHERE run_id=?", (run_id,)
            ).fetchone()
            ev = Event.make(run_id, last + 1, payload, type)
            try:
                self._c.execute(
                    "INSERT INTO events (run_id, seq, event_id, ts, type, payload) VALUES (?,?,?,?,?,?)",
                    (ev.run_id, ev.seq, ev.event_id, ev.ts.isoformat(), ev.type.value, json.dumps(ev.payload)),
                )
            except sqlite3.IntegrityError as exc:  # another process won the race
                raise ConcurrencyError(str(exc)) from exc
        return ev

    def read(self, run_id: str, after_seq: int = 0) -> list[Event]:
        rows = self._c.execute(
            "SELECT run_id, seq, event_id, ts, type, payload FROM events "
            "WHERE run_id=? AND seq>? ORDER BY seq",
            (run_id, after_seq),
        ).fetchall()
        return [
            Event(run_id=r, seq=s, event_id=i, ts=datetime.fromisoformat(ts),
                  type=EventType(t), payload=json.loads(p))
            for r, s, i, ts, t, p in rows
        ]

    def list_runs(self) -> list[str]:
        return [r for (r,) in self._c.execute("SELECT DISTINCT run_id FROM events ORDER BY ts")]
