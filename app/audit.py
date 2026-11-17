"""Meter occupancy and cross-device conflict audit.

Everything here is derived from the *current* revisions of the readings and
handover tables (the highest revision of every ``source_id``).  It never reads
the computed ``intervals`` table, so a rejected/failed rebuild or a service
restart cannot change (or resurrect) an audit conclusion, and an accepted
correction changes the conclusion immediately.

Occupancy model
---------------
For one device, its authoritative readings are ordered by time.  For every
pair of adjacent readings:

* same meter  -> the meter is occupied by that device from the first reading
  until the second one;
* cross meter -> the segment can only be attributed when exactly one matching
  handover of the same device (old meter, new meter, timestamp strictly inside
  the segment) exists.  In that case the segment is split at the handover
  timestamp: the old meter is occupied up to the handover, the new meter from
  the handover on.  With no / more than one matching handover the attribution
  is undeterminable and the transition is reported as *pending* -- never as
  "no conflict".

For every meter, occupancy segments of *different* devices whose time spans
overlap are cross-device conflicts.  Contiguous overlaps of the same device
pair are merged into one period.  Each reported period carries the bounding
readings and the handover (if any) used to split a segment.

A reading on a meter that falls inside another device's occupancy but cannot
be bounded by that device's own readings is also reported as pending review,
because its attribution is indeterminate.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Iterable

# Highest revision of every reading source, i.e. the authoritative rows.
_LATEST_READINGS_SQL = """
WITH latest AS (
    SELECT source_id, MAX(revision) AS max_revision
    FROM readings GROUP BY source_id
)
SELECT r.* FROM readings r
JOIN latest l
  ON r.source_id = l.source_id AND r.revision = l.max_revision
ORDER BY r.device_id, r.ts, r.id
"""

_LATEST_HANDOVERS_SQL = """
WITH latest AS (
    SELECT source_id, MAX(revision) AS max_revision
    FROM handovers GROUP BY source_id
)
SELECT h.* FROM handovers h
JOIN latest l
  ON h.source_id = l.source_id AND h.revision = l.max_revision
ORDER BY h.device_id, h.ts, h.id
"""


@dataclass(frozen=True)
class Prov:
    """A pointer to the authoritative reading/handover bounding a segment."""

    kind: str  # "reading" | "handover"
    source_id: str
    revision: int
    ts: str


@dataclass
class Occupancy:
    meter_no: str
    device_id: str
    start_ts: str
    end_ts: str
    start: Prov
    end: Prov
    handover: Prov | None = None


@dataclass
class Pending:
    kind: str  # cross_meter | unresolved_reading | no_data
    device_id: str | None
    meter_no: str | None
    start_ts: str | None
    end_ts: str | None
    reason: str
    refs: list[Prov] = field(default_factory=list)
    old_meter_no: str | None = None
    new_meter_no: str | None = None


@dataclass
class Conflict:
    meter_no: str
    start_ts: str
    end_ts: str
    occupancies: list[Occupancy] = field(default_factory=list)

    @property
    def devices(self) -> list[str]:
        return sorted({o.device_id for o in self.occupancies})


@dataclass
class AuditData:
    occupancies: list[Occupancy]
    pendings: list[Pending]
    known_meters: set[str]
    reading_rows: dict[tuple[str, int], sqlite3.Row]
    handover_rows: dict[tuple[str, int], sqlite3.Row]


def _reading_prov(row: sqlite3.Row) -> Prov:
    return Prov("reading", row["source_id"], row["revision"], row["ts"])


def _handover_prov(row: sqlite3.Row) -> Prov:
    return Prov("handover", row["source_id"], row["revision"], row["ts"])


def build_audit_data(conn: sqlite3.Connection) -> AuditData:
    """Build all occupancy segments / pending reviews from committed data."""
    reading_rows: dict[tuple[str, int], sqlite3.Row] = {}
    handover_rows: dict[tuple[str, int], sqlite3.Row] = {}
    readings_by_device: dict[str, list[sqlite3.Row]] = {}
    handovers_by_device: dict[str, list[sqlite3.Row]] = {}
    readings_by_meter: dict[str, list[sqlite3.Row]] = {}
    known_meters: set[str] = set()

    for row in conn.execute(_LATEST_READINGS_SQL).fetchall():
        reading_rows[(row["source_id"], row["revision"])] = row
        readings_by_device.setdefault(row["device_id"], []).append(row)
        readings_by_meter.setdefault(row["meter_no"], []).append(row)
        known_meters.add(row["meter_no"])

    for row in conn.execute(_LATEST_HANDOVERS_SQL).fetchall():
        handover_rows[(row["source_id"], row["revision"])] = row
        handovers_by_device.setdefault(row["device_id"], []).append(row)
        known_meters.add(row["old_meter_no"])
        known_meters.add(row["new_meter_no"])

    occupancies: list[Occupancy] = []
    pendings: list[Pending] = []

    for device_id, readings in readings_by_device.items():
        handovers = handovers_by_device.get(device_id, [])
        for a, b in zip(readings, readings[1:]):
            if b["ts"] <= a["ts"]:
                # No positive-length period can be attributed.
                continue
            if a["meter_no"] == b["meter_no"]:
                occupancies.append(
                    Occupancy(
                        meter_no=a["meter_no"],
                        device_id=device_id,
                        start_ts=a["ts"],
                        end_ts=b["ts"],
                        start=_reading_prov(a),
                        end=_reading_prov(b),
                    )
                )
                continue

            matches = [
                h
                for h in handovers
                if h["old_meter_no"] == a["meter_no"]
                and h["new_meter_no"] == b["meter_no"]
                and a["ts"] < h["ts"] < b["ts"]
            ]
            if len(matches) != 1:
                reason = (
                    "ambiguous_handover: more than one matching handover inside "
                    "the cross-meter transition"
                    if matches
                    else "handover_missing: no matching handover inside the "
                    "cross-meter transition"
                )
                pendings.append(
                    Pending(
                        kind="cross_meter",
                        device_id=device_id,
                        meter_no=None,
                        start_ts=a["ts"],
                        end_ts=b["ts"],
                        reason=reason,
                        refs=[_reading_prov(a), _reading_prov(b)],
                        old_meter_no=a["meter_no"],
                        new_meter_no=b["meter_no"],
                    )
                )
                continue

            h = matches[0]
            hp = _handover_prov(h)
            occupancies.append(
                Occupancy(
                    meter_no=a["meter_no"],
                    device_id=device_id,
                    start_ts=a["ts"],
                    end_ts=h["ts"],
                    start=_reading_prov(a),
                    end=hp,
                    handover=hp,
                )
            )
            occupancies.append(
                Occupancy(
                    meter_no=b["meter_no"],
                    device_id=device_id,
                    start_ts=h["ts"],
                    end_ts=b["ts"],
                    start=hp,
                    end=_reading_prov(b),
                    handover=hp,
                )
            )

    _add_unresolved_readings(occupancies, pendings, readings_by_meter)

    return AuditData(
        occupancies=occupancies,
        pendings=pendings,
        known_meters=known_meters,
        reading_rows=reading_rows,
        handover_rows=handover_rows,
    )


def _add_unresolved_readings(
    occupancies: list[Occupancy],
    pendings: list[Pending],
    readings_by_meter: dict[str, list[sqlite3.Row]],
) -> None:
    """Flag readings on a meter that cannot be bounded by their own device.

    A reading that is not inside one of its own device's occupancy segments,
    on a meter that another device also uses, leaves the attribution
    indeterminate at that instant.  Reporting "clear" could hide a duplicate
    registration, so such a reading is reported as pending review.
    """
    own_segments: dict[tuple[str, str], list[Occupancy]] = {}
    for occ in occupancies:
        own_segments.setdefault((occ.meter_no, occ.device_id), []).append(occ)

    pending_windows: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for p in pendings:
        if p.kind != "cross_meter" or p.device_id is None:
            continue
        for meter in (p.old_meter_no, p.new_meter_no):
            if meter:
                pending_windows.setdefault((meter, p.device_id), []).append(
                    (p.start_ts or "", p.end_ts or "")
                )

    devices_by_meter: dict[str, set[str]] = {}
    for rows in readings_by_meter.values():
        for row in rows:
            devices_by_meter.setdefault(row["meter_no"], set()).add(row["device_id"])

    for meter_no, rows in readings_by_meter.items():
        for row in rows:
            device_id = row["device_id"]
            ts = row["ts"]
            if any(
                o.start_ts <= ts <= o.end_ts
                for o in own_segments.get((meter_no, device_id), [])
            ):
                continue
            if any(
                a <= ts <= b
                for a, b in pending_windows.get((meter_no, device_id), [])
            ):
                # Already reported as an undeterminable cross-meter transition.
                continue
            other_devices = devices_by_meter.get(meter_no, set()) - {device_id}
            if not other_devices:
                # No other device uses this meter at all: no ambiguity.
                continue
            pendings.append(
                Pending(
                    kind="unresolved_reading",
                    device_id=device_id,
                    meter_no=meter_no,
                    start_ts=ts,
                    end_ts=ts,
                    reason=(
                        "unbounded_reading: device has a reading on a meter also "
                        "used by another device, but no adjacent reading pair "
                        "bounds its attribution"
                    ),
                    refs=[_reading_prov(row)],
                )
            )


def _overlap(a: Occupancy, b: Occupancy) -> tuple[str, str] | None:
    if a.device_id == b.device_id:
        return None
    start = max(a.start_ts, b.start_ts)
    end = min(a.end_ts, b.end_ts)
    if start < end:  # a touching boundary is not an overlap period
        return start, end
    return None


def find_conflicts(data: AuditData) -> list[Conflict]:
    """Every cross-device overlap period, per meter and device pair.

    Contiguous overlaps between the same pair of devices on the same meter are
    merged into one period so the caller sees the whole "crossed meters"
    window rather than one row per adjacent reading pair.
    """
    by_meter: dict[str, list[Occupancy]] = {}
    for occ in data.occupancies:
        by_meter.setdefault(occ.meter_no, []).append(occ)

    conflicts: list[Conflict] = []
    for meter_no, segments in by_meter.items():
        by_device: dict[str, list[Occupancy]] = {}
        for occ in segments:
            by_device.setdefault(occ.device_id, []).append(occ)
        devices = sorted(by_device)
        for i, dev_a in enumerate(devices):
            for dev_b in devices[i + 1:]:
                windows: list[tuple[str, str, list[Occupancy]]] = []
                for a in by_device[dev_a]:
                    for b in by_device[dev_b]:
                        window = _overlap(a, b)
                        if window is not None:
                            windows.append((window[0], window[1], [a, b]))
                if not windows:
                    continue
                windows.sort(key=lambda w: (w[0], w[1]))
                merged: list[tuple[str, str, list[Occupancy]]] = []
                for start, end, segs in windows:
                    if merged and start <= merged[-1][1]:
                        p_start, p_end, p_segs = merged[-1]
                        seen = {(s.device_id, s.start_ts, s.end_ts, s.meter_no) for s in p_segs}
                        for s in segs:
                            if (s.device_id, s.start_ts, s.end_ts, s.meter_no) not in seen:
                                p_segs.append(s)
                                seen.add((s.device_id, s.start_ts, s.end_ts, s.meter_no))
                        merged[-1] = (p_start, max(p_end, end), p_segs)
                    else:
                        merged.append((start, end, list(segs)))
                for start, end, segs in merged:
                    conflicts.append(
                        Conflict(
                            meter_no=meter_no,
                            start_ts=start,
                            end_ts=end,
                            occupancies=sorted(
                                segs,
                                key=lambda s: (s.start_ts, s.end_ts, s.device_id),
                            ),
                        )
                    )
    conflicts.sort(key=lambda c: (c.meter_no, c.start_ts, c.end_ts))
    return conflicts


def _prov_dict(p: Prov) -> dict:
    return {
        "kind": p.kind,
        "source_id": p.source_id,
        "revision": p.revision,
        "ts": p.ts,
    }


def _reading_dict(row: sqlite3.Row) -> dict:
    return {
        "source_id": row["source_id"],
        "revision": row["revision"],
        "device_id": row["device_id"],
        "ts": row["ts"],
        "meter_no": row["meter_no"],
        "cumulative_kwh": row["cumulative"],
    }


def _handover_dict(row: sqlite3.Row) -> dict:
    return {
        "source_id": row["source_id"],
        "revision": row["revision"],
        "device_id": row["device_id"],
        "ts": row["ts"],
        "old_meter_no": row["old_meter_no"],
        "old_final_kwh": row["old_final"],
        "new_meter_no": row["new_meter_no"],
        "new_initial_kwh": row["new_initial"],
    }


def _collect_refs(
    data: AuditData, provs: Iterable[Prov]
) -> tuple[list[dict], list[dict]]:
    reading_keys: set[tuple[str, int]] = set()
    handover_keys: set[tuple[str, int]] = set()
    for p in provs:
        if p.kind == "reading":
            reading_keys.add((p.source_id, p.revision))
        else:
            handover_keys.add((p.source_id, p.revision))
    readings = sorted(
        (_reading_dict(data.reading_rows[k]) for k in reading_keys),
        key=lambda d: (d["ts"], d["source_id"], d["revision"]),
    )
    handovers = sorted(
        (_handover_dict(data.handover_rows[k]) for k in handover_keys),
        key=lambda d: (d["ts"], d["source_id"], d["revision"]),
    )
    return readings, handovers


def conflict_to_dict(c: Conflict, data: AuditData) -> dict:
    provs: list[Prov] = []
    occupancies: list[dict] = []
    for o in c.occupancies:
        provs.extend([o.start, o.end])
        if o.handover is not None:
            provs.append(o.handover)
        occupancies.append(
            {
                "device_id": o.device_id,
                "meter_no": o.meter_no,
                "start_ts": o.start_ts,
                "end_ts": o.end_ts,
                "start_ref": _prov_dict(o.start),
                "end_ref": _prov_dict(o.end),
                "handover_source_id": o.handover.source_id if o.handover else None,
                "handover_revision": o.handover.revision if o.handover else None,
            }
        )
    readings, handovers = _collect_refs(data, provs)
    return {
        "meter_no": c.meter_no,
        "start_ts": c.start_ts,
        "end_ts": c.end_ts,
        "devices": c.devices,
        "device_ids": c.devices,
        "occupancies": occupancies,
        "readings": readings,
        "handovers": handovers,
        # convenience: the reading/handover source ids backing this period
        "reading_sources": [
            {"source_id": r["source_id"], "revision": r["revision"]} for r in readings
        ],
        "handover_sources": [
            {"source_id": h["source_id"], "revision": h["revision"]} for h in handovers
        ],
    }


def pending_to_dict(p: Pending, data: AuditData) -> dict:
    readings, _ = _collect_refs(data, p.refs)
    return {
        "kind": p.kind,
        "device_id": p.device_id,
        "meter_no": p.meter_no,
        "start_ts": p.start_ts,
        "end_ts": p.end_ts,
        "old_meter_no": p.old_meter_no,
        "new_meter_no": p.new_meter_no,
        "reason": p.reason,
        "readings": readings,
        "sources": [{"source_id": r["source_id"], "revision": r["revision"]} for r in readings],
    }


def _in_window(start: str | None, end: str | None, from_ts: str, to_ts: str) -> bool:
    if start is None or end is None:
        return True
    return start < to_ts and end > from_ts


def meter_audit(
    conn: sqlite3.Connection,
    meter_no: str,
    from_ts: str,
    to_ts: str,
) -> dict:
    """Audit one meter over ``[from_ts, to_ts)`` against committed data.

    ``status`` is ``conflict`` when a cross-device overlap exists, otherwise
    ``pending`` when the attribution of this meter cannot be determined in the
    window, and only ``clear`` when neither is true.
    """
    data = build_audit_data(conn)
    conflicts = [
        c
        for c in find_conflicts(data)
        if c.meter_no == meter_no and c.start_ts < to_ts and c.end_ts > from_ts
    ]
    pending = [
        p
        for p in data.pendings
        if _pending_matches_meter(p, meter_no) and _in_window(p.start_ts, p.end_ts, from_ts, to_ts)
    ]
    meter_known = meter_no in data.known_meters
    if not meter_known:
        pending.append(
            Pending(
                kind="no_data",
                device_id=None,
                meter_no=meter_no,
                start_ts=None,
                end_ts=None,
                reason=(
                    "no_data: no committed reading or handover references this "
                    "meter, so a conflict-free conclusion cannot be established"
                ),
            )
        )
    if conflicts:
        status = "conflict"
    elif pending:
        status = "pending"
    else:
        status = "clear"

    conflict_dicts = [conflict_to_dict(c, data) for c in conflicts]
    pending_dicts = [pending_to_dict(p, data) for p in pending]

    reading_keys: dict[tuple[str, int], dict] = {}
    handover_keys: dict[tuple[str, int], dict] = {}
    for c in conflict_dicts:
        for r in c["readings"]:
            reading_keys[(r["source_id"], r["revision"])] = r
        for h in c["handovers"]:
            handover_keys[(h["source_id"], h["revision"])] = h
    for p in pending_dicts:
        for r in p["readings"]:
            reading_keys[(r["source_id"], r["revision"])] = r

    return {
        "meter_no": meter_no,
        "from_ts": from_ts,
        "to_ts": to_ts,
        "status": status,
        "conflict": bool(conflicts),
        "has_conflict": bool(conflicts),
        "conflict_count": len(conflicts),
        "meter_known": meter_known,
        "conflicts": conflict_dicts,
        "conflict_periods": conflict_dicts,
        "pending": pending_dicts,
        "pending_reviews": pending_dicts,
        "readings": sorted(
            reading_keys.values(), key=lambda d: (d["ts"], d["source_id"], d["revision"])
        ),
        "handovers": sorted(
            handover_keys.values(), key=lambda d: (d["ts"], d["source_id"], d["revision"])
        ),
    }


def _pending_matches_meter(p: Pending, meter_no: str) -> bool:
    if p.kind == "cross_meter":
        return meter_no in (p.old_meter_no, p.new_meter_no)
    return p.meter_no == meter_no


def conflicts_for_interval(
    start_ts: str,
    end_ts: str,
    meter_nos: Iterable[str],
    conflicts: Iterable[Conflict],
) -> list[Conflict]:
    """Conflicts overlapping an interval on any of its meters."""
    meters = set(meter_nos)
    return [
        c
        for c in conflicts
        if c.meter_no in meters and start_ts < c.end_ts and end_ts > c.start_ts
    ]


def conflict_detail(hits: Iterable[Conflict]) -> str | None:
    items = [
        {
            "meter_no": c.meter_no,
            "start_ts": c.start_ts,
            "end_ts": c.end_ts,
            "devices": c.devices,
        }
        for c in hits
    ]
    return json.dumps(items, ensure_ascii=False, sort_keys=True) if items else None


def conflicts_by_device(conflicts: Iterable[Conflict]) -> dict[str, frozenset]:
    """Map device -> fingerprints of every conflict period it participates in.

    Used to detect exactly which devices' conflict involvement changed after a
    submission, so only those devices lose current-supply eligibility.
    """
    out: dict[str, set[tuple[str, str, str]]] = {}
    for c in conflicts:
        key = (c.meter_no, c.start_ts, c.end_ts)
        for device_id in c.devices:
            out.setdefault(device_id, set()).add(key)
    return {d: frozenset(keys) for d, keys in out.items()}


def changed_conflict_devices(
    old: dict[str, frozenset], new: dict[str, frozenset]
) -> set[str]:
    return {
        device_id
        for device_id in set(old) | set(new)
        if old.get(device_id, frozenset()) != new.get(device_id, frozenset())
    }
