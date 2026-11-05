"""Append-only audit trail of every budget-relevant decision."""
from __future__ import annotations

import csv
from typing import Dict, List, Optional

AUDIT_FIELDS = ["t", "action", "target", "cost", "pool", "spent", "remaining", "p_prime", "s_mon",
                "taint", "H", "net", "density", "theta", "reason"]


class AuditLog:
    def __init__(self) -> None:
        self._rows: List[Dict] = []

    def append(self, **fields) -> None:
        unknown = set(fields) - set(AUDIT_FIELDS)
        if unknown:
            raise KeyError(f"unknown audit fields: {sorted(unknown)}")
        self._rows.append({k: fields.get(k) for k in AUDIT_FIELDS})

    def __len__(self) -> int:
        return len(self._rows)

    def rows(self) -> List[Dict]:
        return [dict(r) for r in self._rows]

    def overspent_rows(self) -> int:
        """Independent recount: charged rows whose pool went below zero (always 0 unless the ledger is bugged)."""
        return sum(1 for r in self._rows if r["remaining"] is not None and r["remaining"] < 0)

    def to_csv(self, path: str, extra: Optional[Dict] = None) -> None:
        extra = extra or {}
        with open(path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(extra) + AUDIT_FIELDS)
            wr.writeheader()
            for r in self._rows:
                wr.writerow({**extra, **r})
