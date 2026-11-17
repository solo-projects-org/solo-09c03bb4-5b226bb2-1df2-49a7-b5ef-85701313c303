"""End-to-end tests for meter conflict audit (/meter-audit) and gating.

Covers:
* same-meter occupancy is computed from adjacent readings;
* cross-meter segments are only split by a unique matching handover;
* undeterminable attribution is reported as pending, never ``clear``;
* current-supply eligibility of every conflict interval is revoked;
* corrections change the conclusion immediately, even when the rebuild fails
  or the service restarts in between;
* unrelated devices keep supplying current data throughout;
* revision corrections that move a reading to another device are honoured.
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402

WINDOW = {"from_ts": "2026-01-01T00:00:00Z", "to_ts": "2026-01-02T00:00:00Z"}


@pytest.fixture()
def ctx():
    tmp = tempfile.TemporaryDirectory()
    db_path = os.path.join(tmp.name, "test.db")
    app = create_app(Settings(db_path=db_path, fail_device=None))
    with TestClient(app) as client:
        yield client, db_path, tmp
    tmp.cleanup()


def reading(source, rev, device, ts, meter, kwh):
    return {
        "source_id": source, "revision": rev, "device_id": device,
        "ts": ts, "meter_no": meter, "cumulative_kwh": kwh,
    }


def handover(source, rev, device, ts, old_m, old_v, new_m, new_v):
    return {
        "source_id": source, "revision": rev, "device_id": device, "ts": ts,
        "old_meter_no": old_m, "old_final_kwh": old_v,
        "new_meter_no": new_m, "new_initial_kwh": new_v,
    }


def audit(client, meter="M1", **window):
    params = {"meter_no": meter, **WINDOW, **window}
    return client.get("/meter-audit", params=params).json()


def current(client, device=None):
    params = {"status": "current"}
    if device:
        params["device_id"] = device
    return client.get("/intervals", params=params).json()


def read_ok(client, payload, **kwargs):
    r = client.post("/readings", json=payload, **kwargs)
    assert r.status_code == 201, r.text
    return r.json()


def handover_ok(client, payload, **kwargs):
    r = client.post("/handovers", json=payload, **kwargs)
    assert r.status_code == 201, r.text
    return r.json()


# ---------------------------------------------------------------------------
# 1. Detection of a duplicated meter registration
# ---------------------------------------------------------------------------

def test_same_meter_on_two_devices_reports_overlap_period(ctx):
    client, _, _ = ctx
    read_ok(client, reading("D1-1", 1, "D1", "2026-01-01T00:00:00Z", "M1", 100.0))
    read_ok(client, reading("D1-2", 1, "D1", "2026-01-01T02:00:00Z", "M1", 130.0))
    read_ok(client, reading("D2-1", 1, "D2", "2026-01-01T01:00:00Z", "M1", 10.0))
    read_ok(client, reading("D2-2", 1, "D2", "2026-01-01T03:00:00Z", "M1", 40.0))

    body = audit(client)
    assert body["status"] == "conflict"
    assert body["conflict"] is True and body["has_conflict"] is True
    assert body["conflict_count"] == 1
    period = body["conflicts"][0]
    assert period["meter_no"] == "M1"
    assert period["start_ts"] == "2026-01-01T01:00:00.000Z"
    assert period["end_ts"] == "2026-01-01T02:00:00.000Z"
    assert period["devices"] == ["D1", "D2"]
    # Bounding readings of both devices are returned with full provenance.
    assert {(r["source_id"], r["device_id"]) for r in period["readings"]} == {
        ("D1-1", "D1"), ("D1-2", "D1"), ("D2-1", "D2"), ("D2-2", "D2"),
    }
    assert [(o["device_id"], o["start_ts"], o["end_ts"]) for o in period["occupancies"]] == [
        ("D1", "2026-01-01T00:00:00.000Z", "2026-01-01T02:00:00.000Z"),
        ("D2", "2026-01-01T01:00:00.000Z", "2026-01-01T03:00:00.000Z"),
    ]
    assert period["occupancies"][0]["start_ref"]["source_id"] == "D1-1"


def test_non_overlapping_devices_are_clear(ctx):
    client, _, _ = ctx
    read_ok(client, reading("A1", 1, "D1", "2026-01-01T00:00:00Z", "M1", 100.0))
    read_ok(client, reading("A2", 1, "D1", "2026-01-01T01:00:00Z", "M1", 110.0))
    # D2 starts exactly when D1 stops -> touching boundary is not an overlap.
    read_ok(client, reading("B1", 1, "D2", "2026-01-01T01:00:00Z", "M1", 110.0))
    read_ok(client, reading("B2", 1, "D2", "2026-01-01T02:00:00Z", "M1", 120.0))

    body = audit(client)
    assert body["status"] == "clear"
    assert body["conflicts"] == [] and body["conflict_count"] == 0
    assert body["pending"] == []


def test_same_device_same_meter_is_not_a_conflict(ctx):
    client, _, _ = ctx
    read_ok(client, reading("A1", 1, "D1", "2026-01-01T00:00:00Z", "M1", 100.0))
    read_ok(client, reading("A2", 1, "D1", "2026-01-01T02:00:00Z", "M1", 130.0))
    assert audit(client)["status"] == "clear"


def test_contiguous_overlaps_are_merged_into_one_period(ctx):
    client, _, _ = ctx
    read_ok(client, reading("D1-1", 1, "D1", "2026-01-01T00:00:00Z", "M1", 100.0))
    read_ok(client, reading("D1-2", 1, "D1", "2026-01-01T02:00:00Z", "M1", 130.0))
    read_ok(client, reading("D1-3", 1, "D1", "2026-01-01T04:00:00Z", "M1", 160.0))
    read_ok(client, reading("D2-1", 1, "D2", "2026-01-01T01:00:00Z", "M1", 10.0))
    read_ok(client, reading("D2-2", 1, "D2", "2026-01-01T03:00:00Z", "M1", 40.0))

    body = audit(client)
    assert body["conflict_count"] == 1
    assert body["conflicts"][0]["start_ts"] == "2026-01-01T01:00:00.000Z"
    assert body["conflicts"][0]["end_ts"] == "2026-01-01T03:00:00.000Z"


def test_window_filter_excludes_period_outside_range(ctx):
    client, _, _ = ctx
    read_ok(client, reading("D1-1", 1, "D1", "2026-01-01T00:00:00Z", "M1", 100.0))
    read_ok(client, reading("D1-2", 1, "D1", "2026-01-01T02:00:00Z", "M1", 130.0))
    read_ok(client, reading("D2-1", 1, "D2", "2026-01-01T01:00:00Z", "M1", 10.0))
    read_ok(client, reading("D2-2", 1, "D2", "2026-01-01T03:00:00Z", "M1", 40.0))

    later = audit(client, from_ts="2026-01-01T05:00:00Z", to_ts="2026-01-01T06:00:00Z")
    assert later["status"] == "clear" and later["conflicts"] == []


# ---------------------------------------------------------------------------
# 2. Cross-meter segments need a unique matching handover
# ---------------------------------------------------------------------------

def test_cross_meter_transition_without_handover_is_pending_not_clear(ctx):
    client, _, _ = ctx
    read_ok(client, reading("R1", 1, "D1", "2026-01-01T00:00:00Z", "M1", 100.0))
    read_ok(client, reading("R2", 1, "D1", "2026-01-01T02:00:00Z", "M2", 5.0))
    # A reading of another device on M1 only, inside the unresolved transition.
    read_ok(client, reading("R3", 1, "D2", "2026-01-01T01:00:00Z", "M1", 10.0))
    read_ok(client, reading("R4", 1, "D2", "2026-01-01T03:00:00Z", "M1", 40.0))

    body = audit(client)
    assert body["status"] == "pending"
    assert body["conflicts"] == []
    reasons = [p["reason"] for p in body["pending"]]
    assert any(r.startswith("handover_missing") for r in reasons)
    assert any(p["kind"] == "cross_meter" for p in body["pending"])


def test_unique_handover_splits_the_transition_and_reveals_overlap(ctx):
    client, _, _ = ctx
    read_ok(client, reading("R1", 1, "D1", "2026-01-01T00:00:00Z", "M1", 100.0))
    read_ok(client, reading("R2", 1, "D1", "2026-01-01T02:00:00Z", "M2", 5.0))
    read_ok(client, reading("R3", 1, "D2", "2026-01-01T01:00:00Z", "M1", 10.0))
    read_ok(client, reading("R4", 1, "D2", "2026-01-01T03:00:00Z", "M1", 40.0))
    handover_ok(client, handover(
        "H1", 1, "D1", "2026-01-01T01:30:00Z", "M1", 120.0, "M2", 5.0))

    body = audit(client)
    assert body["status"] == "conflict"
    period = body["conflicts"][0]
    # D1 occupies M1 only until the handover at 01:30.
    assert period["start_ts"] == "2026-01-01T01:00:00.000Z"
    assert period["end_ts"] == "2026-01-01T01:30:00.000Z"
    # The handover itself is returned as a source.
    assert [h["source_id"] for h in period["handovers"]] == ["H1"]
    occ_d1 = [o for o in period["occupancies"] if o["device_id"] == "D1"][0]
    assert occ_d1["handover_source_id"] == "H1"
    assert occ_d1["end_ref"] == {
        "kind": "handover", "source_id": "H1", "revision": 1,
        "ts": "2026-01-01T01:30:00.000Z",
    }


def test_ambiguous_handover_is_pending(ctx):
    client, _, _ = ctx
    read_ok(client, reading("R1", 1, "D1", "2026-01-01T00:00:00Z", "M1", 100.0))
    read_ok(client, reading("R2", 1, "D1", "2026-01-01T04:00:00Z", "M2", 5.0))
    handover_ok(client, handover(
        "H1", 1, "D1", "2026-01-01T01:00:00Z", "M1", 110.0, "M2", 1.0))
    handover_ok(client, handover(
        "H2", 1, "D1", "2026-01-01T02:00:00Z", "M1", 120.0, "M2", 2.0))

    body = audit(client, meter="M1")
    assert body["status"] == "pending"
    assert any(p["reason"].startswith("ambiguous_handover") for p in body["pending"])


def test_handover_revision_with_wrong_meter_removes_split(ctx):
    client, _, _ = ctx
    read_ok(client, reading("R1", 1, "D1", "2026-01-01T00:00:00Z", "M1", 100.0))
    read_ok(client, reading("R2", 1, "D1", "2026-01-01T02:00:00Z", "M2", 5.0))
    handover_ok(client, handover(
        "H1", 1, "D1", "2026-01-01T01:00:00Z", "M1", 110.0, "M2", 5.0))
    # Correction no longer mentions the M1 -> M2 pair, so no split is possible.
    handover_ok(client, handover(
        "H1", 2, "D1", "2026-01-01T01:00:00Z", "M9", 110.0, "M2", 5.0))

    body = audit(client)
    assert body["status"] == "pending"
    assert body["conflicts"] == []


# ---------------------------------------------------------------------------
# 3. Conflict intervals must not supply current energy
# ---------------------------------------------------------------------------

def test_conflict_intervals_are_not_current_energy(ctx):
    client, _, _ = ctx
    read_ok(client, reading("D1-1", 1, "D1", "2026-01-01T00:00:00Z", "M1", 100.0))
    read_ok(client, reading("D1-2", 1, "D1", "2026-01-01T02:00:00Z", "M1", 130.0))
    assert current(client, "D1")[0]["energy_kwh"] == pytest.approx(30.0)

    read_ok(client, reading("D2-1", 1, "D2", "2026-01-01T01:00:00Z", "M1", 10.0))
    read_ok(client, reading("D2-2", 1, "D2", "2026-01-01T03:00:00Z", "M1", 40.0))

    # Neither device may supply the overlapping interval as current energy.
    assert current(client, "D1") == []
    assert current(client, "D2") == []
    all_d1 = client.get("/intervals", params={"device_id": "D1", "status": "all"}).json()
    # A stale audit row may still carry its historical value; only the
    # conflict-gated row describes the live conclusion.
    conflict_rows = [i for i in all_d1 if i["status"] == "conflict"]
    assert conflict_rows and conflict_rows[0]["energy_kwh"] is None
    assert conflict_rows[0]["status"] == "conflict"
    assert conflict_rows[0]["reason"].startswith("meter_conflict")
    assert "M1" in conflict_rows[0]["conflict_detail"]
    # Explicit conflict query only returns the gated rows.
    assert all(
        i["status"] == "conflict"
        for i in client.get("/intervals", params={"status": "conflict"}).json()
    )


def test_unrelated_device_keeps_supplying_current_energy(ctx):
    client, _, _ = ctx
    read_ok(client, reading("D1-1", 1, "D1", "2026-01-01T00:00:00Z", "M1", 100.0))
    read_ok(client, reading("D1-2", 1, "D1", "2026-01-01T02:00:00Z", "M1", 130.0))
    read_ok(client, reading("X1", 1, "D9", "2026-01-01T00:00:00Z", "M9", 5.0))
    read_ok(client, reading("X2", 1, "D9", "2026-01-01T02:00:00Z", "M9", 11.0))

    read_ok(client, reading("D2-1", 1, "D2", "2026-01-01T01:00:00Z", "M1", 10.0))
    read_ok(client, reading("D2-2", 1, "D2", "2026-01-01T03:00:00Z", "M1", 40.0))

    assert current(client, "D1") == []
    assert current(client, "D2") == []
    d9 = current(client, "D9")
    assert len(d9) == 1 and d9[0]["energy_kwh"] == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# 4. Corrections change the conclusion immediately, independent of rebuild
# ---------------------------------------------------------------------------

def test_correction_changes_audit_even_when_rebuild_fails(ctx):
    client, db_path, _ = ctx
    read_ok(client, reading("D1-1", 1, "D1", "2026-01-01T00:00:00Z", "M1", 100.0))
    read_ok(client, reading("D1-2", 1, "D1", "2026-01-01T02:00:00Z", "M1", 130.0))
    read_ok(client, reading("D2-1", 1, "D2", "2026-01-01T01:00:00Z", "M1", 10.0))
    read_ok(client, reading("D2-2", 1, "D2", "2026-01-01T03:00:00Z", "M1", 40.0))
    assert audit(client)["status"] == "conflict"

    # Correction moves both of D2's readings to another meter -> the overlap
    # disappears (moving only one would leave an undeterminable transition).
    body = read_ok(
        client,
        reading("D2-1", 2, "D2", "2026-01-01T01:00:00Z", "M2", 10.0),
        headers={"X-Fail-Rebuild": "true"},
    )
    assert body["rebuild"]["success"] is False
    body = read_ok(
        client,
        reading("D2-2", 2, "D2", "2026-01-01T03:00:00Z", "M2", 40.0),
        headers={"X-Fail-Rebuild": "true"},
    )
    assert body["rebuild"]["success"] is False

    # The audit conclusion follows committed data, not the failed rebuild.
    assert audit(client, meter="M1")["status"] == "clear"
    # ... but the affected intervals stay revoked until a successful rebuild.
    assert current(client, "D1") == []
    assert current(client, "D2") == []
    assert client.post("/rebuild", params={"device_id": "D1"}).json()["success"] is True
    assert current(client, "D1")[0]["energy_kwh"] == pytest.approx(30.0)


def test_restart_cannot_restore_old_conclusion(ctx):
    client, db_path, _ = ctx
    read_ok(client, reading("D1-1", 1, "D1", "2026-01-01T00:00:00Z", "M1", 100.0))
    read_ok(client, reading("D1-2", 1, "D1", "2026-01-01T02:00:00Z", "M1", 130.0))
    read_ok(client, reading("D2-1", 1, "D2", "2026-01-01T01:00:00Z", "M1", 10.0))
    read_ok(client, reading("D2-2", 1, "D2", "2026-01-01T03:00:00Z", "M1", 40.0))
    assert audit(client)["status"] == "conflict"

    # Correct away the conflict (both readings), then crash the rebuild.
    read_ok(
        client,
        reading("D2-1", 2, "D2", "2026-01-01T01:00:00Z", "M2", 10.0),
        headers={"X-Fail-Rebuild": "true"},
    )
    read_ok(
        client,
        reading("D2-2", 2, "D2", "2026-01-01T03:00:00Z", "M2", 40.0),
        headers={"X-Fail-Rebuild": "true"},
    )
    assert audit(client)["status"] == "clear"

    app2 = create_app(Settings(db_path=db_path, fail_device=None))
    with TestClient(app2) as client2:
        # A restart neither resurrects the conflict nor restores old energy.
        assert audit(client2)["status"] == "clear"
        assert current(client2, "D1") == []
        assert current(client2, "D2") == []
        assert client2.post("/rebuild", params={"device_id": "D1"}).json()["success"] is True
        assert current(client2, "D1")[0]["energy_kwh"] == pytest.approx(30.0)


def test_correction_that_introduces_conflict_revokes_other_device(ctx):
    """A correction on D2 must revoke D1's already-current interval too."""
    client, _, _ = ctx
    read_ok(client, reading("D1-1", 1, "D1", "2026-01-01T00:00:00Z", "M1", 100.0))
    read_ok(client, reading("D1-2", 1, "D1", "2026-01-01T02:00:00Z", "M1", 130.0))
    read_ok(client, reading("D2-1", 1, "D2", "2026-01-01T01:00:00Z", "M2", 10.0))
    read_ok(client, reading("D2-2", 1, "D2", "2026-01-01T03:00:00Z", "M2", 40.0))
    assert audit(client, meter="M1")["status"] == "clear"
    assert current(client, "D1")[0]["energy_kwh"] == pytest.approx(30.0)

    # Move both D2 readings onto M1 -> D1 is now part of a conflict.
    body = read_ok(client, reading("D2-1", 2, "D2", "2026-01-01T01:00:00Z", "M1", 10.0))
    # The first move only makes D2's transition undeterminable; D1's interval
    # is still current because no cross-device overlap has been established.
    assert current(client, "D1") != []
    body = read_ok(client, reading("D2-2", 2, "D2", "2026-01-01T03:00:00Z", "M1", 40.0))
    # This correction establishes an overlap and must revoke D1 as well,
    # even though only D2's reading changed.
    assert "D1" in body["affected_devices"]
    assert current(client, "D1") == []
    assert audit(client, meter="M1")["status"] == "conflict"


def test_rebuild_failure_keeps_conflict_revoked_until_success(ctx):
    client, _, _ = ctx
    read_ok(client, reading("D1-1", 1, "D1", "2026-01-01T00:00:00Z", "M1", 100.0))
    read_ok(client, reading("D1-2", 1, "D1", "2026-01-01T02:00:00Z", "M1", 130.0))
    read_ok(client, reading("D2-1", 1, "D2", "2026-01-01T01:00:00Z", "M1", 10.0))
    read_ok(client, reading("D2-2", 1, "D2", "2026-01-01T03:00:00Z", "M1", 40.0))

    # Resolve by moving both D2 readings to M2, but fail the rebuild.
    read_ok(
        client, reading("D2-1", 2, "D2", "2026-01-01T01:00:00Z", "M2", 10.0),
        headers={"X-Fail-Rebuild": "true"},
    )
    read_ok(
        client, reading("D2-2", 2, "D2", "2026-01-01T03:00:00Z", "M2", 40.0),
        headers={"X-Fail-Rebuild": "true"},
    )
    assert audit(client)["status"] == "clear"
    assert current(client, "D1") == []

    # Retry both devices: only the interval itself recovers.
    assert client.post("/rebuild", params={"device_id": "D1"}).json()["success"]
    assert client.post("/rebuild", params={"device_id": "D2"}).json()["success"]
    assert current(client, "D1")[0]["energy_kwh"] == pytest.approx(30.0)
    assert current(client, "D2")[0]["energy_kwh"] == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# 5. Revision handling for moved sources (current revision is global)
# ---------------------------------------------------------------------------

def test_corrected_reading_moved_to_another_device_leaves_old_device(ctx):
    client, _, _ = ctx
    read_ok(client, reading("R1", 1, "D1", "2026-01-01T00:00:00Z", "M1", 100.0))
    read_ok(client, reading("R2", 1, "D1", "2026-01-01T02:00:00Z", "M1", 130.0))
    # Same source, now attributed to D2.  The superseded revision must not be
    # used by D1 any more; D1 keeps only its single reading, so it has no
    # current interval, and the old device is reported as affected.
    body = read_ok(client, reading("R2", 2, "D2", "2026-01-01T02:00:00Z", "M1", 130.0))
    assert "D1" in body["affected_devices"]
    assert client.get("/intervals", params={"device_id": "D1", "status": "current"}).json() == []
    intervals = client.get("/intervals", params={"device_id": "D1", "status": "all"}).json()
    assert all(i["status"] != "current" for i in intervals)
    d2 = client.get("/intervals", params={"device_id": "D2", "status": "all"}).json()
    assert d2 == []  # a single reading cannot form an interval


# ---------------------------------------------------------------------------
# 6. Query validation and API surface
# ---------------------------------------------------------------------------

def test_unknown_meter_is_pending_not_clear(ctx):
    client, _, _ = ctx
    body = audit(client, meter="NOPE")
    assert body["status"] == "pending"
    assert body["meter_known"] is False
    assert body["pending"][0]["kind"] == "no_data"


def test_audit_rejects_naive_or_bad_windows(ctx):
    client, _, _ = ctx
    base = {"meter_no": "M1"}
    assert client.get("/meter-audit", params={**base, "from_ts": "2026-01-01T00:00:00",
                                              "to_ts": "2026-01-02T00:00:00Z"}).status_code == 422
    assert client.get("/meter-audit", params={**base, "from_ts": "nope",
                                              "to_ts": "2026-01-02T00:00:00Z"}).status_code == 422
    # Ordering is validated.
    assert client.get("/meter-audit", params={**base, "from_ts": "2026-01-02T00:00:00Z",
                                              "to_ts": "2026-01-01T00:00:00Z"}).status_code == 422
    # Missing parameters are 422, not a silent default window.
    assert client.get("/meter-audit", params={"meter_no": "M1"}).status_code == 422
    # Aliases are accepted.
    r = client.get("/meter-audit", params={"meter_no": "M1", "start_ts": "2026-01-01T00:00:00Z",
                                           "end_ts": "2026-01-02T00:00:00Z"})
    assert r.status_code == 200


def test_openapi_and_docs_expose_meter_audit(ctx):
    client, _, _ = ctx
    spec = client.get("/openapi.json").json()
    assert "/meter-audit" in spec["paths"]
    params = {p["name"] for p in spec["paths"]["/meter-audit"]["get"]["parameters"]}
    assert {"meter_no", "from_ts", "to_ts"} <= params
    assert client.get("/docs").status_code == 200


# ---------------------------------------------------------------------------
# 7. Concurrent audit / correction on one committed dataset
# ---------------------------------------------------------------------------

def test_concurrent_audit_and_correction_are_consistent(ctx):
    """Parallel audits must see complete committed states, never half a write."""
    import threading
    import traceback

    client, _, _ = ctx
    read_ok(client, reading("D1-1", 1, "D1", "2026-01-01T00:00:00Z", "M1", 100.0))
    read_ok(client, reading("D1-2", 1, "D1", "2026-01-01T04:00:00Z", "M1", 140.0))

    failures: list[str] = []

    def worker(i: int) -> None:
        for j in range(12):
            try:
                if (i + j) % 2 == 0:
                    r = client.post(
                        "/readings",
                        json=reading(
                            f"W{i}-{j}", 1, f"Z{i}",
                            f"2026-01-01T0{j % 5}:00:00Z", "MX", 1.0,
                        ),
                    )
                    assert r.status_code == 201, r.text
                else:
                    r = client.get("/meter-audit", params={
                        "meter_no": "M1", **WINDOW,
                    })
                    assert r.status_code == 200, r.text
                    body = r.json()
                    assert body["status"] in {"clear", "conflict", "pending"}
                    for period in body["conflicts"]:
                        assert len(period["devices"]) >= 2
            except Exception:  # pragma: no cover - failure path
                failures.append(traceback.format_exc())

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert failures == []


def test_audit_sees_correction_immediately_after_accept(ctx):
    """An accepted correction is visible to the next audit (committed data)."""
    client, _, _ = ctx
    read_ok(client, reading("D1-1", 1, "D1", "2026-01-01T00:00:00Z", "M1", 100.0))
    read_ok(client, reading("D1-2", 1, "D1", "2026-01-01T02:00:00Z", "M1", 130.0))
    read_ok(client, reading("D2-1", 1, "D2", "2026-01-01T01:00:00Z", "M1", 10.0))
    read_ok(client, reading("D2-2", 1, "D2", "2026-01-01T03:00:00Z", "M1", 40.0))
    assert audit(client)["status"] == "conflict"

    read_ok(client, reading("D2-1", 2, "D2", "2026-01-01T01:00:00Z", "M2", 10.0))
    read_ok(client, reading("D2-2", 2, "D2", "2026-01-01T03:00:00Z", "M2", 40.0))
    assert audit(client)["status"] == "clear"
    assert current(client, "D1")[0]["energy_kwh"] == pytest.approx(30.0)
