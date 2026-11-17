"""Pydantic request/response models and shared helpers."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def to_utc_iso(dt: datetime) -> str:
    """Normalize any aware datetime to a UTC ISO-8601 string."""
    if dt.tzinfo is None:
        raise ValueError("timestamp must include a timezone (use UTC, e.g. 2026-01-01T00:00:00Z)")
    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class ReadingIn(BaseModel):
    source_id: str = Field(min_length=1, description="来源编号，同一读数来源的唯一标识")
    revision: int = Field(ge=1, description="递增修订号，首次为 1，再次提交必须 +1")
    device_id: str = Field(min_length=1)
    ts: datetime = Field(description="读数时刻（UTC，需带时区）")
    meter_no: str = Field(min_length=1)
    cumulative_kwh: float = Field(ge=0, description="累计电表读数 kWh（非负）")

    @field_validator("source_id", "device_id", "meter_no")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v


class HandoverIn(BaseModel):
    source_id: str = Field(min_length=1, description="来源编号")
    revision: int = Field(ge=1, description="递增修订号")
    device_id: str = Field(min_length=1)
    ts: datetime = Field(description="换表时刻（UTC，需带时区）")
    old_meter_no: str = Field(min_length=1)
    old_final_kwh: float = Field(ge=0, description="旧表末值（累计 kWh）")
    new_meter_no: str = Field(min_length=1)
    new_initial_kwh: float = Field(ge=0, description="新表初值（累计 kWh）")

    @field_validator("source_id", "device_id", "old_meter_no", "new_meter_no")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v

    @field_validator("new_meter_no")
    @classmethod
    def _distinct(cls, v: str, info):
        old = info.data.get("old_meter_no")
        if old is not None and v == old:
            raise ValueError("new_meter_no must differ from old_meter_no")
        return v


class AcceptOut(BaseModel):
    accepted: bool
    kind: str  # reading | handover
    source_id: str
    revision: int
    idempotent: bool = False
    affected_devices: list[str]
    rebuild: dict


class IntervalOut(BaseModel):
    id: int
    device_id: str
    start_ts: str
    end_ts: str
    start_meter_no: str
    end_meter_no: str
    energy_kwh: float | None
    undeterminable: bool
    reason: str | None
    status: str
    start_source_id: str
    start_revision: int
    end_source_id: str
    end_revision: int
    handover_source_id: str | None
    handover_revision: int | None
    conflict: bool
    conflict_detail: str | None


class RebuildOut(BaseModel):
    run_id: int
    success: bool
    rebuilt_devices: list[str]
    failed_devices: dict[str, str]
    detail: str | None = None


# ---------------------------------------------------------------------------
# Meter conflict audit (/meter-audit)
# ---------------------------------------------------------------------------

class ProvenanceOut(BaseModel):
    """A pointer to the authoritative reading/handover bounding a segment."""

    kind: Literal["reading", "handover"]
    source_id: str
    revision: int
    ts: str


class ReadingSourceOut(BaseModel):
    source_id: str
    revision: int
    device_id: str
    ts: str
    meter_no: str
    cumulative_kwh: float


class HandoverSourceOut(BaseModel):
    source_id: str
    revision: int
    device_id: str
    ts: str
    old_meter_no: str
    old_final_kwh: float
    new_meter_no: str
    new_initial_kwh: float


class ConflictOccupancyOut(BaseModel):
    """One device's occupancy segment that overlaps another device's."""

    device_id: str
    meter_no: str
    start_ts: str
    end_ts: str
    start_ref: ProvenanceOut
    end_ref: ProvenanceOut
    handover_source_id: str | None
    handover_revision: int | None


class ConflictOut(BaseModel):
    meter_no: str
    start_ts: str
    end_ts: str
    devices: list[str]
    device_ids: list[str]
    occupancies: list[ConflictOccupancyOut]
    readings: list[ReadingSourceOut]
    handovers: list[HandoverSourceOut]
    reading_sources: list[dict]
    handover_sources: list[dict]


class PendingOut(BaseModel):
    """A period whose meter attribution cannot be determined."""

    kind: Literal["cross_meter", "unresolved_reading", "no_data"]
    device_id: str | None
    meter_no: str | None
    start_ts: str | None
    end_ts: str | None
    old_meter_no: str | None
    new_meter_no: str | None
    reason: str
    readings: list[ReadingSourceOut]
    sources: list[dict]


class MeterAuditOut(BaseModel):
    meter_no: str
    from_ts: str
    to_ts: str
    status: Literal["clear", "conflict", "pending"]
    conflict: bool = Field(description="True when status == 'conflict'")
    has_conflict: bool
    conflict_count: int
    meter_known: bool
    conflicts: list[ConflictOut]
    conflict_periods: list[ConflictOut]
    pending: list[PendingOut]
    pending_reviews: list[PendingOut]
    readings: list[ReadingSourceOut]
    handovers: list[HandoverSourceOut]


class MeterAuditQuery(BaseModel):
    """Validated query parameters (UTC instants, half-open window)."""

    meter_no: str = Field(min_length=1)
    from_ts: datetime
    to_ts: datetime

    @field_validator("meter_no")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v

    @field_validator("from_ts", "to_ts")
    @classmethod
    def _aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError(
                "timestamp must include a timezone (use UTC, e.g. 2026-01-01T00:00:00Z)"
            )
        return v

    @model_validator(mode="after")
    def _ordered(self) -> "MeterAuditQuery":
        if self.from_ts >= self.to_ts:
            raise ValueError("from_ts must be strictly before to_ts")
        return self
