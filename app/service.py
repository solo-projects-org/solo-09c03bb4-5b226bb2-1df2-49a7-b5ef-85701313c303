"""Domain logic: revision rules, interval settlement, rebuild transactions.

Key guarantees implemented here:

* Submissions are versioned by (source_id, revision).  Revisions must be
  strictly increasing; re-posting the exact same payload is idempotent;
  posting different content under an existing (source_id, revision) is
  rejected with 409.
* As soon as a late reading/correction is *accepted*, every current interval
  of every affected device is marked ``stale`` inside the same transaction.
* The rebuild that follows runs in its own transaction per device.  A failure
  (or a process kill, or a restart) leaves the stale rows in place — old
  results never come back as current, and devices not involved keep serving.
* Undeterminable intervals are stored with ``energy_kwh = NULL``; zero is
  only ever written for a genuinely measured zero difference.
* Cross-device meter conflicts (the same meter occupied by two devices) are
  derived from the committed current revisions in :mod:`app.audit`.  An
  interval overlapping such a conflict is stored as ``conflict`` and is never
  eligible to supply current energy; accepting a correction changes that
  conclusion immediately, independent of any rebuild or restart.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3

from .errors import ConflictError, RebuildFailure
from .models import HandoverIn, ReadingIn, to_utc_iso, utc_now_iso
from . import audit


def _hash(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# Revisioned submission handling
# ---------------------------------------------------------------------------

def _check_revision(
    conn: sqlite3.Connection,
    table: str,
    source_id: str,
    revision: int,
    content_hash: str,
) -> bool:
    """Validate the revision chain.

    Revisions must be consecutive: the first submission is revision 1 and
    every later submission must use exactly ``latest + 1``.  Re-posting the
    exact same payload under an existing revision is idempotent.  A content
    mismatch on an existing (source_id, revision) or any rollback/skipping is
    rejected with ConflictError (HTTP 409).
    """
    row = conn.execute(
        f"SELECT revision, content_hash FROM {table} WHERE source_id = ? "
        f"ORDER BY revision DESC LIMIT 1",
        (source_id,),
    ).fetchone()
    if row is not None:
        latest = row["revision"]
        if revision < latest:
            raise ConflictError(
                f"revision rollback rejected: latest revision of source "
                f"'{source_id}' is {latest}, got {revision}"
            )
        if revision == latest:
            existing = conn.execute(
                f"SELECT content_hash FROM {table} WHERE source_id = ? AND revision = ?",
                (source_id, revision),
            ).fetchone()
            if existing["content_hash"] != content_hash:
                raise ConflictError(
                    f"content conflict: source '{source_id}' revision {revision} "
                    f"already exists with different content"
                )
            return True  # identical re-submission
        if revision != latest + 1:
            raise ConflictError(
                f"non-consecutive revision rejected: latest revision of source "
                f"'{source_id}' is {latest}, expected {latest + 1}, got {revision}"
            )
    elif revision != 1:
        raise ConflictError(
            f"first revision of source '{source_id}' must be 1, got {revision}"
        )
    return False


def _stale_current_intervals(conn: sqlite3.Connection, device_ids: list[str]) -> int:
    """Immediately revoke current-supply eligibility on affected devices.

    Rows already marked as conflicting also lose that state: their conclusion
    must be recomputed from the newly committed data, never inherited.
    """
    if not device_ids:
        return 0
    placeholders = ",".join("?" for _ in device_ids)
    cur = conn.execute(
        f"UPDATE intervals SET status = 'stale', "
        f"superseded_at = ? WHERE status IN ('current','conflict') "
        f"AND device_id IN ({placeholders})",
        [utc_now_iso(), *device_ids],
    )
    return cur.rowcount


def _conflict_map(conn: sqlite3.Connection) -> dict[str, frozenset]:
    return audit.conflicts_by_device(audit.find_conflicts(audit.build_audit_data(conn)))


def _moved_source_devices(
    conn: sqlite3.Connection, table: str, source_id: str, current_device: str
) -> set[str]:
    """Other devices that a source used to (or still does) belong to.

    Revisions of one source may name different devices over time; a change
    must revoke the intervals of the device the earlier revision pointed at,
    even when the same transaction also moves the source forward.
    """
    return {
        r["device_id"]
        for r in conn.execute(
            f"SELECT DISTINCT device_id FROM {table} WHERE source_id = ? "
            f"AND device_id <> ?",
            (source_id, current_device),
        )
    }


def submit_reading(conn: sqlite3.Connection, payload: ReadingIn) -> dict:
    ts_iso = to_utc_iso(payload.ts)
    content = {
        "device_id": payload.device_id,
        "ts": ts_iso,
        "meter_no": payload.meter_no,
        "cumulative_kwh": payload.cumulative_kwh,
    }
    digest = _hash(content)
    affected: list[str] = []

    conn.execute("BEGIN IMMEDIATE")
    try:
        idempotent = _check_revision(conn, "readings", payload.source_id, payload.revision, digest)
        if not idempotent:
            before = _conflict_map(conn)
            conn.execute(
                "INSERT INTO readings (source_id, revision, device_id, ts, meter_no, "
                "cumulative, content_hash) VALUES (?,?,?,?,?,?,?)",
                (
                    payload.source_id, payload.revision, payload.device_id, ts_iso,
                    payload.meter_no, payload.cumulative_kwh, digest,
                ),
            )
            # The new reading belongs to its own device.  A correction may also
            # move the reading to a different device, so stale whatever device
            # the previous revision pointed at.
            prev = conn.execute(
                "SELECT device_id FROM readings WHERE source_id = ? AND revision < ? "
                "ORDER BY revision DESC LIMIT 1",
                (payload.source_id, payload.revision),
            ).fetchone()
            devices = {payload.device_id}
            if prev is not None:
                devices.add(prev["device_id"])
            affected = sorted(devices)
            _stale_current_intervals(conn, affected)
            moved = _moved_source_devices(conn, "readings", payload.source_id, payload.device_id)
            # A newly introduced / resolved cross-device conflict changes the
            # eligibility of every device involved, even one whose own readings
            # did not move.  Their intervals are revoked in this transaction so
            # no stale conclusion can be served in the meantime.
            after = _conflict_map(conn)
            changed = audit.changed_conflict_devices(before, after) - set(affected) - moved
            if changed:
                _stale_current_intervals(conn, sorted(changed))
            affected = sorted(devices | changed | moved)
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return {
        "kind": "reading",
        "source_id": payload.source_id,
        "revision": payload.revision,
        "idempotent": idempotent,
        "affected_devices": affected,
    }


def submit_handover(conn: sqlite3.Connection, payload: HandoverIn) -> dict:
    ts_iso = to_utc_iso(payload.ts)
    content = {
        "device_id": payload.device_id,
        "ts": ts_iso,
        "old_meter_no": payload.old_meter_no,
        "old_final_kwh": payload.old_final_kwh,
        "new_meter_no": payload.new_meter_no,
        "new_initial_kwh": payload.new_initial_kwh,
    }
    digest = _hash(content)
    affected: list[str] = []

    conn.execute("BEGIN IMMEDIATE")
    try:
        idempotent = _check_revision(conn, "handovers", payload.source_id, payload.revision, digest)
        if not idempotent:
            before = _conflict_map(conn)
            conn.execute(
                "INSERT INTO handovers (source_id, revision, device_id, ts, old_meter_no, "
                "old_final, new_meter_no, new_initial, content_hash) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    payload.source_id, payload.revision, payload.device_id, ts_iso,
                    payload.old_meter_no, payload.old_final_kwh,
                    payload.new_meter_no, payload.new_initial_kwh, digest,
                ),
            )
            prev = conn.execute(
                "SELECT device_id, old_meter_no, new_meter_no FROM handovers "
                "WHERE source_id = ? AND revision < ? ORDER BY revision DESC LIMIT 1",
                (payload.source_id, payload.revision),
            ).fetchone()
            devices = {payload.device_id}
            if prev is not None:
                devices.add(prev["device_id"])
            affected = sorted(devices)
            _stale_current_intervals(conn, affected)
            moved = _moved_source_devices(conn, "readings", payload.source_id, payload.device_id)
            # A handover decides how a cross-meter segment is split, so it can
            # create or remove a cross-device occupancy overlap on other
            # devices as well.
            after = _conflict_map(conn)
            changed = audit.changed_conflict_devices(before, after) - set(affected) - moved
            if changed:
                _stale_current_intervals(conn, sorted(changed))
            affected = sorted(devices | changed | moved)
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return {
        "kind": "handover",
        "source_id": payload.source_id,
        "revision": payload.revision,
        "idempotent": idempotent,
        "affected_devices": affected,
    }


# ---------------------------------------------------------------------------
# Interval computation
# ---------------------------------------------------------------------------

def _latest_readings(conn: sqlite3.Connection, device_id: str) -> list[sqlite3.Row]:
    """Authoritative readings of a device.

    The highest revision of every ``source_id`` is resolved *globally* first
    and only then filtered by device: a correction that moves a source to a
    different device must remove it from the old device instead of leaving the
    superseded revision behind.
    """
    return conn.execute(
        "WITH latest AS ("
        "  SELECT source_id, MAX(revision) AS max_rev FROM readings GROUP BY source_id"
        ") SELECT r.* FROM readings r JOIN latest l "
        "ON r.source_id = l.source_id AND r.revision = l.max_rev "
        "WHERE r.device_id = ? "
        "ORDER BY r.ts ASC, r.id ASC",
        (device_id,),
    ).fetchall()


def _latest_handovers(conn: sqlite3.Connection, device_id: str) -> list[sqlite3.Row]:
    """Authoritative handovers of a device (current revision per source)."""
    return conn.execute(
        "WITH latest AS ("
        "  SELECT source_id, MAX(revision) AS max_rev FROM handovers GROUP BY source_id"
        ") SELECT h.* FROM handovers h JOIN latest l "
        "ON h.source_id = l.source_id AND h.revision = l.max_rev "
        "WHERE h.device_id = ? "
        "ORDER BY h.ts ASC, h.id ASC",
        (device_id,),
    ).fetchall()


def _settle_interval(
    conn: sqlite3.Connection,
    device_id: str,
    a: sqlite3.Row,
    b: sqlite3.Row,
    conflicts: list[audit.Conflict] | None = None,
    handovers: list[sqlite3.Row] | None = None,
) -> dict:
    """Compute the consumption between two adjacent readings.

    Same meter  -> simple difference; a backward reading is undeterminable
                   (never clamped to zero).
    Cross meter -> only settle when exactly one handover of this device falls
                   strictly inside the interval and its meter numbers match
                   both ends; otherwise undeterminable ('handover missing').
    Either way, if the interval overlaps a cross-device meter conflict the
    energy is undeterminable ('meter_conflict') and carries the conflict
    detail: such a row may never supply current energy.
    """
    common = {
        "device_id": device_id,
        "start_ts": a["ts"], "end_ts": b["ts"],
        "start_meter_no": a["meter_no"], "end_meter_no": b["meter_no"],
        "start_source_id": a["source_id"], "start_revision": a["revision"],
        "end_source_id": b["source_id"], "end_revision": b["revision"],
        "handover_source_id": None, "handover_revision": None,
        "conflict": 0, "conflict_detail": None,
    }

    hits = audit.conflicts_for_interval(
        a["ts"], b["ts"], {a["meter_no"], b["meter_no"]}, conflicts or []
    )
    if hits:
        return {
            **common,
            "energy_kwh": None,
            "conflict": 1,
            "conflict_detail": audit.conflict_detail(hits),
            "reason": "meter_conflict: interval overlaps a cross-device meter "
                      "registration conflict; not eligible for current supply",
        }

    if a["meter_no"] == b["meter_no"]:
        diff = b["cumulative"] - a["cumulative"]
        if diff < 0:
            return {**common, "energy_kwh": None,
                    "reason": "negative_difference: meter reading went backward"}
        return {**common, "energy_kwh": round(diff, 9), "reason": None}

    # Cross-meter interval: look for the unique matching handover inside it,
    # considering only the current revision of every handover source.
    if handovers is None:
        handovers = _latest_handovers(conn, device_id)
    unique = [
        h
        for h in handovers
        if h["ts"] > a["ts"]
        and h["ts"] < b["ts"]
        and h["old_meter_no"] == a["meter_no"]
        and h["new_meter_no"] == b["meter_no"]
    ]

    if len(unique) == 0:
        return {**common, "energy_kwh": None,
                "reason": "handover_missing: no matching handover inside cross-meter interval"}
    if len(unique) > 1:
        return {**common, "energy_kwh": None,
                "reason": "ambiguous_handover: more than one matching handover inside interval"}

    h = unique[0]
    seg1 = h["old_final"] - a["cumulative"]
    seg2 = b["cumulative"] - h["new_initial"]
    if seg1 < 0 or seg2 < 0:
        return {**common, "energy_kwh": None,
                "handover_source_id": h["source_id"], "handover_revision": h["revision"],
                "reason": "negative_segment: reading backward relative to handover values"}
    return {
        **common,
        "energy_kwh": round(seg1 + seg2, 9),
        "reason": None,
        "handover_source_id": h["source_id"],
        "handover_revision": h["revision"],
    }


def devices_with_stale_intervals(conn: sqlite3.Connection) -> list[str]:
    """Devices holding intervals that lost current eligibility.

    Includes ``conflict`` rows: they are not current energy either, and a
    retry must recompute them from the committed data.
    """
    return [
        r["device_id"]
        for r in conn.execute(
            "SELECT DISTINCT device_id FROM intervals "
            "WHERE status IN ('stale','conflict') ORDER BY 1"
        )
    ]


def rebuild_device(
    conn: sqlite3.Connection,
    device_id: str,
    *,
    fail_device: str | None = None,
    fail_header_devices: set[str] | None = None,
) -> int:
    """Atomically rebuild all intervals of one device.

    Previously stale rows are kept for audit; only ``current`` rows are
    replaced, and new rows are inserted as ``current`` in one transaction.
    An injected failure raises *before* commit to simulate a crash: rollback
    leaves the run at status ``running`` and every previously stale interval
    untouched.  On service restart init_db flips dangling ``running`` runs to
    ``interrupted`` without reviving any interval.
    """
    fail_header_devices = fail_header_devices or set()
    run = conn.execute(
        "INSERT INTO rebuild_runs (device_id, status) VALUES (?, 'running')",
        (device_id,),
    )
    run_id = run.lastrowid
    conn.commit()

    conn.execute("BEGIN IMMEDIATE")
    try:
        if fail_device == device_id or device_id in fail_header_devices:
            raise RebuildFailure(
                device_id,
                "injected failure before commit (test hook); stale intervals retained",
            )

        readings = _latest_readings(conn, device_id)
        # Touch the handover table so its latest revisions are considered even
        # when no cross-meter interval exists yet.
        handovers = _latest_handovers(conn, device_id)
        # Conflict conclusions are always derived from committed readings /
        # handovers inside this same transaction, so a concurrent submission
        # can never be half-observed.
        conflicts = audit.find_conflicts(audit.build_audit_data(conn))

        rows = [
            _settle_interval(
                conn, device_id, readings[i], readings[i + 1], conflicts, handovers
            )
            for i in range(len(readings) - 1)
        ]

        # Stale rows stay as an audit trail; only live rows are replaced.
        conn.execute(
            "DELETE FROM intervals WHERE device_id = ? AND status IN ('current','conflict')",
            (device_id,),
        )
        for r in rows:
            conn.execute(
                "INSERT INTO intervals (device_id, start_ts, end_ts, start_meter_no, "
                "end_meter_no, energy_kwh, status, reason, start_source_id, "
                "start_revision, end_source_id, end_revision, handover_source_id, "
                "handover_revision, conflict, conflict_detail) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    r["device_id"], r["start_ts"], r["end_ts"],
                    r["start_meter_no"], r["end_meter_no"], r["energy_kwh"],
                    "conflict" if r["conflict"] else "current", r["reason"],
                    r["start_source_id"], r["start_revision"],
                    r["end_source_id"], r["end_revision"],
                    r["handover_source_id"], r["handover_revision"],
                    r["conflict"], r["conflict_detail"],
                ),
            )
        conn.execute(
            "UPDATE rebuild_runs SET status = 'success', "
            "finished_at = ? WHERE id = ?",
            (utc_now_iso(), run_id),
        )
        conn.commit()
        return len(rows)
    except RebuildFailure:
        # Simulated crash: rollback and deliberately leave the run 'running'
        # so the restart path marks it 'interrupted'.  Stale rows survive.
        conn.rollback()
        raise
    except Exception as exc:
        conn.rollback()
        conn.execute(
            "UPDATE rebuild_runs SET status = 'failed', detail = ?, "
            "finished_at = ? WHERE id = ?",
            (str(exc), utc_now_iso(), run_id),
        )
        conn.commit()
        raise


def rebuild(
    conn: sqlite3.Connection,
    device_id: str | None = None,
    *,
    devices: list[str] | None = None,
    all_devices: bool = False,
    only_stale: bool = False,
    fail_device: str | None = None,
    fail_header_devices: set[str] | None = None,
) -> dict:
    """Rebuild one device, or every device that needs it.

    ``devices`` limits the run to an explicit set (used by automatic rebuilds
    after submissions, including a device's very first reading when no stale
    rows exist yet).  By default all devices that have readings are rebuilt;
    ``only_stale=True`` limits the run to devices currently holding stale
    intervals.  A failure on one device is recorded and does not abort the
    others; stale intervals of a failed device remain queryable as stale and
    never regain current eligibility.
    """
    if device_id is not None:
        targets = [device_id]
    elif devices is not None:
        targets = list(dict.fromkeys(devices))  # de-duplicate, keep order
    elif all_devices or not only_stale:
        targets = [
            r["device_id"]
            for r in conn.execute(
                "SELECT DISTINCT device_id FROM readings ORDER BY 1"
            )
        ]
    else:
        targets = devices_with_stale_intervals(conn)

    rebuilt: list[str] = []
    failed: dict[str, str] = {}
    for dev in targets:
        try:
            n = rebuild_device(
                conn, dev,
                fail_device=fail_device,
                fail_header_devices=fail_header_devices,
            )
            rebuilt.append(f"{dev} ({n} intervals)")
        except RebuildFailure as exc:
            failed[dev] = exc.reason

    success = not failed
    run_row = conn.execute(
        "INSERT INTO rebuild_runs (device_id, status, detail, finished_at) "
        "VALUES (?,?,?,?)",
        (
            device_id,
            "success" if success else "failed",
            json.dumps({"rebuilt": rebuilt, "failed": failed}),
            utc_now_iso(),
        ),
    )
    conn.commit()
    return {
        "run_id": run_row.lastrowid,
        "success": success,
        "rebuilt_devices": rebuilt,
        "failed_devices": failed,
    }


def list_intervals(
    conn: sqlite3.Connection,
    device_id: str | None,
    status: str,
) -> list[dict]:
    """List intervals with conflict eligibility derived from committed data.

    The stored status is authoritative for ``stale``/``conflict`` rows, but a
    row is *also* checked against a live audit of the committed readings and
    handovers.  An interval overlapping a cross-device meter conflict is never
    served as current energy, even if a rebuild has not run since the conflict
    appeared (for example a conflict introduced outside a rebuild, or a
    database upgraded from an older schema).
    """
    clauses: list[str] = []
    params: list[str] = []
    if status == "current":
        clauses.append("status = 'current'")
    elif status == "conflict":
        clauses.append("status = 'conflict'")
    elif status != "all":
        clauses.append("status = ?")
        params.append(status)
    if device_id is not None:
        clauses.append("device_id = ?")
        params.append(device_id)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = conn.execute(
        f"SELECT * FROM intervals{where} ORDER BY device_id, start_ts, id",
        params,
    ).fetchall()

    conflicts = audit.find_conflicts(audit.build_audit_data(conn))
    out: list[dict] = []
    for r in rows:
        d = dict(r)
        hits = audit.conflicts_for_interval(
            r["start_ts"], r["end_ts"],
            {r["start_meter_no"], r["end_meter_no"]}, conflicts,
        )
        if hits:
            d["conflict"] = 1
            d["conflict_detail"] = audit.conflict_detail(hits)
            if d["status"] == "current":
                if status == "current":
                    # Never supply current energy for a conflicting interval.
                    continue
                # The stored row predates the conflict (e.g. a conflict that
                # appeared without a rebuild, or a database upgraded from the
                # old schema).  Report it as gated, never as energy.
                d["status"] = "conflict"
                d["energy_kwh"] = None
                d["reason"] = (
                    "meter_conflict: interval overlaps a cross-device meter "
                    "registration conflict; not eligible for current supply"
                )
            # Stale rows keep their own historical conclusion untouched.
        elif d["status"] == "conflict":
            if status == "conflict":
                # Stored conclusion no longer matches committed data; keep it
                # out until a rebuild recomputes the row.
                continue
            if status == "all":
                # Keep audit history truthful: an interval marked conflicting
                # by an earlier rebuild is not a current-energy row.  Its
                # eligibility is re-established only by a successful rebuild.
                d["status"] = "stale"
        out.append(d)
    return out


def meter_audit(
    conn: sqlite3.Connection,
    meter_no: str,
    from_ts: str,
    to_ts: str,
) -> dict:
    """Meter conflict audit over committed data (see :mod:`app.audit`)."""
    return audit.meter_audit(conn, meter_no, from_ts, to_ts)
