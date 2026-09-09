"""
Pydantic request/response schemas.

These decouple the database model from the wire format, allowing us to
hide internal fields (prev_hash, hash) from the public API while still
exposing them on demand.
"""
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


# ------------------------------------------------------------------ #
# Camera
# ------------------------------------------------------------------ #
class CameraBase(BaseModel):
    name: str
    url: str
    location: Optional[str] = None
    is_active: bool = True


class CameraCreate(CameraBase):
    pass


class CameraRead(CameraBase):
    id: int
    model_config = ConfigDict(from_attributes=True)


# ------------------------------------------------------------------ #
# Rule
# ------------------------------------------------------------------ #
class RuleBase(BaseModel):
    rule_type: str
    geometry: Any          # list of coordinate pairs / line points
    params: Optional[dict] = None
    is_active: bool = True


class RuleCreate(RuleBase):
    pass


class RuleRead(RuleBase):
    id: int
    camera_id: int
    model_config = ConfigDict(from_attributes=True)


# ------------------------------------------------------------------ #
# Alert
# ------------------------------------------------------------------ #
class AlertBase(BaseModel):
    alert_type: str
    object_class: Optional[str] = None
    track_id: Optional[int] = None
    confidence: Optional[float] = None
    timestamp: str
    snapshot_path: Optional[str] = None
    clip_path: Optional[str] = None


class AlertRead(AlertBase):
    id: int
    camera_id: int
    hash: str
    prev_hash: str
    model_config = ConfigDict(from_attributes=True)


# ------------------------------------------------------------------ #
# Stats
# ------------------------------------------------------------------ #
class StatsResponse(BaseModel):
    total_alerts: int
    by_type: dict[str, int]
    by_camera: dict[str, int]
    today: int
