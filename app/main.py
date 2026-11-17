"""FastAPI application: meter readings, handovers, intervals, rebuild.

Swagger UI is served at ``/docs`` (ReDoc at ``/redoc``) and the OpenAPI
document at ``/openapi.json``.
"""
from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .config import Settings
from .database import ConnectionPool
from .errors import ConflictError, RebuildFailure
from .models import (
    AcceptOut,
    HandoverIn,
    IntervalOut,
    MeterAuditOut,
    MeterAuditQuery,
    ReadingIn,
    RebuildOut,
    to_utc_iso,
)
from . import service


def _parse_utc(value: str) -> datetime:
    """Parse a UTC instant, normalizing a trailing ``Z``.

    Naive timestamps and malformed values are rejected with 422 so a window
    can never silently be interpreted in local time.
    """
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"invalid UTC timestamp {value!r}: {exc}")
    if parsed.tzinfo is None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"timestamp {value!r} must include a timezone "
                "(use UTC, e.g. 2026-01-01T00:00:00Z)"
            ),
        )
    return parsed


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # SQLite connections are checked out per request (WAL + busy_timeout
        # guard concurrent writers); see ConnectionPool.
        app.state.pool = ConnectionPool(settings.db_path)
        app.state.pool.init_schema()
        # Deliberately NO automatic rebuild on startup: a restart must never
        # resurrect intervals that lost their current-supply eligibility.
        yield
        app.state.pool.close()

    app = FastAPI(
        title="工厂能耗区间结算 API",
        description=(
            "提交按设备区分的累计电表读数与换表交接，按相邻读数生成带来源编号的"
            "区间电量。倒退读数或交接缺失记为无法判定（energy_kwh=null），绝不报零。"
            "``GET /meter-audit`` 按表号与 UTC 时段核查同一块表是否被重复登记到"
            "多台设备；冲突区间不作为现行电量供数。"
        ),
        version="1.1.0",
        lifespan=lifespan,
    )

    def db(request: Request) -> sqlite3.Connection:
        """One connection per request, returned when the request ends.

        Sync dependencies and the endpoint body may run on different
        threadpool threads; the connection is therefore attached to the
        request state instead of a thread.
        """
        conn = getattr(request.state, "conn", None)
        if conn is None:
            conn = request.app.state.pool.acquire()
            request.state.conn = conn
        return conn

    @app.middleware("http")
    async def release_connection(request: Request, call_next):
        try:
            return await call_next(request)
        finally:
            conn = getattr(request.state, "conn", None)
            if conn is not None:
                request.app.state.pool.release(conn)
                request.state.conn = None

    @app.exception_handler(ConflictError)
    async def conflict_handler(request: Request, exc: ConflictError):
        return JSONResponse(status_code=409, content={"detail": exc.detail})

    @app.get("/health", tags=["运维"])
    def health(conn: sqlite3.Connection = Depends(db)):
        row = conn.execute("SELECT COUNT(*) AS n FROM intervals WHERE status='current'").fetchone()
        stale = conn.execute("SELECT COUNT(*) AS n FROM intervals WHERE status='stale'").fetchone()
        conflict = conn.execute("SELECT COUNT(*) AS n FROM intervals WHERE status='conflict'").fetchone()
        return {
            "status": "ok",
            "current_intervals": row["n"],
            "stale_intervals": stale["n"],
            "conflict_intervals": conflict["n"],
        }

    @app.post("/readings", response_model=AcceptOut, status_code=201, tags=["读数"])
    def post_reading(
        payload: ReadingIn,
        request: Request,
        conn: sqlite3.Connection = Depends(db),
        x_fail_rebuild: bool = Header(default=False, include_in_schema=False),
    ):
        result = service.submit_reading(conn, payload)
        rebuild_info = _auto_rebuild(request, conn, result["affected_devices"], x_fail_rebuild)
        return {**result, "accepted": True, "rebuild": rebuild_info}

    @app.post("/handovers", response_model=AcceptOut, status_code=201, tags=["换表交接"])
    def post_handover(
        payload: HandoverIn,
        request: Request,
        conn: sqlite3.Connection = Depends(db),
        x_fail_rebuild: bool = Header(default=False, include_in_schema=False),
    ):
        result = service.submit_handover(conn, payload)
        rebuild_info = _auto_rebuild(request, conn, result["affected_devices"], x_fail_rebuild)
        return {**result, "accepted": True, "rebuild": rebuild_info}

    def _auto_rebuild(request: Request, conn: sqlite3.Connection, devices: list[str], fail: bool) -> dict:
        if not devices:
            return {"triggered": False}
        fail_set = set(devices) if fail else set()
        out = service.rebuild(
            conn,
            devices=devices,
            fail_device=request.app.state.settings.fail_device,
            fail_header_devices=fail_set,
        )
        out["triggered"] = True
        return out

    @app.post("/rebuild", response_model=RebuildOut, tags=["区间重建"])
    def post_rebuild(
        request: Request,
        device_id: str | None = Query(default=None, description="只重建指定设备"),
        all_devices: bool = Query(default=False, description="强制重建全部设备（默认只重建存在 stale/conflict 区间的设备）"),
        x_fail_rebuild: bool = Header(default=False, include_in_schema=False),
        conn: sqlite3.Connection = Depends(db),
    ):
        fail_set: set[str] = set()
        if x_fail_rebuild:
            if device_id:
                fail_set.add(device_id)
            else:
                fail_set.update(service.devices_with_stale_intervals(conn))
        try:
            return service.rebuild(
                conn, device_id=device_id, all_devices=all_devices,
                only_stale=not all_devices and device_id is None,
                fail_device=request.app.state.settings.fail_device,
                fail_header_devices=fail_set,
            )
        except RebuildFailure as exc:  # pragma: no cover - rebuild() records per-device failures
            return JSONResponse(status_code=500, content={"detail": str(exc)})

    @app.get("/intervals", response_model=list[IntervalOut], tags=["区间查询"])
    def get_intervals(
        device_id: str | None = Query(default=None),
        status: str = Query(
            default="current",
            pattern="^(current|stale|conflict|all)$",
            description="current=可现行供数；stale=已被后续提交取代；conflict=因串表冲突停供；all=全部",
        ),
        conn: sqlite3.Connection = Depends(db),
    ):
        """区间电量。

        ``current`` 只返回可继续供数的区间：与串表冲突重叠的区间绝不出现在
        该结果中（即使尚未重算）。``conflict`` 仅返回停供的冲突区间；``all``
        返回全部并标注每行是否冲突。
        """
        rows = service.list_intervals(conn, device_id, status)
        return [
            {
                "id": r["id"],
                "device_id": r["device_id"],
                "start_ts": r["start_ts"],
                "end_ts": r["end_ts"],
                "start_meter_no": r["start_meter_no"],
                "end_meter_no": r["end_meter_no"],
                "energy_kwh": r["energy_kwh"],
                "undeterminable": r["energy_kwh"] is None,
                "reason": r["reason"],
                "status": r["status"],
                "start_source_id": r["start_source_id"],
                "start_revision": r["start_revision"],
                "end_source_id": r["end_source_id"],
                "end_revision": r["end_revision"],
                "handover_source_id": r["handover_source_id"],
                "handover_revision": r["handover_revision"],
                "conflict": bool(r["conflict"]),
                "conflict_detail": r["conflict_detail"],
            }
            for r in rows
        ]

    @app.get("/meter-audit", response_model=MeterAuditOut, tags=["表号核查"])
    def get_meter_audit(
        request: Request,
        meter_no: str | None = Query(default=None, description="要核查的表号"),
        from_ts: str | None = Query(default=None, description="窗口起点（UTC，需带时区），含"),
        to_ts: str | None = Query(default=None, description="窗口终点（UTC，需带时区），不含"),
        meter_id: str | None = Query(default=None, include_in_schema=False),
        start_ts: str | None = Query(default=None, include_in_schema=False),
        end_ts: str | None = Query(default=None, include_in_schema=False),
        conn: sqlite3.Connection = Depends(db),
    ):
        """按表号与 UTC 时段核查跨设备重复登记（串表）时段。

        结论完全由已提交的读数/交接现行修订推导，与区间重建是否成功或
        服务是否重启无关。存在跨设备重叠即 ``status=conflict``；无法确定
        归属时 ``status=pending``（绝不报 clear/无冲突）。``from_ts`` 含、
        ``to_ts`` 不含；``start_ts``/``end_ts`` 是等价的别名。
        """
        meter = meter_no if meter_no is not None else meter_id
        start = from_ts if from_ts is not None else start_ts
        end = to_ts if to_ts is not None else end_ts
        missing = [
            name
            for name, value in (("meter_no", meter), ("from_ts", start), ("to_ts", end))
            if value is None
        ]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"missing required query parameter(s): {', '.join(missing)}",
            )
        try:
            query = MeterAuditQuery(
                meter_no=meter,
                from_ts=_parse_utc(start),
                to_ts=_parse_utc(end),
            )
        except ValidationError as exc:
            first = exc.errors(include_url=False, include_context=False, include_input=False)[0]
            location = ".".join(str(p) for p in first.get("loc", ()))
            raise HTTPException(
                status_code=422,
                detail=f"{location}: {first.get('msg', 'invalid value')}",
            )
        with request.app.state.pool.read_snapshot() as snap:
            result = service.meter_audit(
                snap,
                query.meter_no,
                to_utc_iso(query.from_ts),
                to_utc_iso(query.to_ts),
            )
        return result

    @app.get("/rebuild-runs", tags=["运维"])
    def get_rebuild_runs(conn: sqlite3.Connection = Depends(db)):
        return [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM rebuild_runs ORDER BY id DESC LIMIT 50"
            ).fetchall()
        ]

    app.state.settings = settings
    return app


app = create_app()
