"""
SQLAlchemy ORM models.

Five tables:

``cameras``        registered video sources (RTSP / HTTP / webcam / file)
``rules``          virtual fences, zones, loiter areas, direction rules
``alerts``         the tamper-evident event log (SHA-256 hash chain)
``checkpoints``    periodic Merkle checkpoints sealing a range of the chain
``watchlist_entries``  enrolled face embeddings
``analysis_sessions``  uploaded-MP4 analysis runs

Timestamps are stored as aware **UTC ISO-8601** strings (sortable, filterable)
alongside a denormalised **IST** display string, so the audit log reads
correctly to an Indian operator without any client-side conversion.
"""
from sqlalchemy import (
    Boolean, Column, Float, ForeignKey, Index, Integer, String, Text,
)
from sqlalchemy.orm import relationship

from core.database import Base

# Alert severities, ordered.
SEVERITY_INFO = "INFO"
SEVERITY_LOW = "LOW"
SEVERITY_MEDIUM = "MEDIUM"
SEVERITY_HIGH = "HIGH"
SEVERITY_CRITICAL = "CRITICAL"
SEVERITY_ORDER = {
    SEVERITY_INFO: 0, SEVERITY_LOW: 1, SEVERITY_MEDIUM: 2,
    SEVERITY_HIGH: 3, SEVERITY_CRITICAL: 4,
}


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True)
    name = Column(String(120), nullable=False)
    url = Column(String(500), nullable=False)
    location = Column(String(200), default="")
    is_active = Column(Boolean, default=True)
    is_online = Column(Boolean, default=False)
    #: "live"   — RTSP / HTTP / webcam index
    #: "file"   — an MP4 registered as a first-class camera source
    #: "upload" — the pseudo-source that owns offline analysis events
    source_kind = Column(String(20), default="live")
    created_at = Column(String(40), default="")

    #: Soft-delete marker. Removing a camera must not destroy its events: the
    #: alert log is a SHA-256 hash chain, and deleting rows from the middle of
    #: it invalidates every subsequent row, so integrity verification would fail
    #: forever afterwards. A removed camera that still owns events is therefore
    #: archived — hidden from every listing, never auto-started, its stream gone
    #: — while its sealed evidence stays verifiable. A camera with no events is
    #: deleted outright. See ``retire_camera``.
    is_deleted = Column(Boolean, default=False, nullable=False)
    deleted_at = Column(String(40), default="")

    #: Rules are configuration and are removed with the camera.
    rules = relationship("Rule", back_populates="camera", cascade="all, delete-orphan")
    #: Alerts are evidence. ``passive_deletes`` keeps SQLAlchemy from issuing a
    #: cascade that would silently shred the audit chain; the delete path checks
    #: for dependants and archives instead.
    alerts = relationship("Alert", back_populates="camera", passive_deletes=True)
    sessions = relationship("AnalysisSession", back_populates="camera",
                            passive_deletes=True)

    def __repr__(self) -> str:
        return f"<Camera {self.id} {self.name}>"

    @property
    def is_file_source(self) -> bool:
        return (self.source_kind or "live") == "file"


class Rule(Base):
    __tablename__ = "rules"

    id = Column(Integer, primary_key=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    #: 'line' | 'zone' | 'loiter' | 'direction'
    rule_type = Column(String(50), nullable=False)
    geometry = Column(Text)      # JSON coordinate list, in frame pixel space
    params = Column(Text)        # JSON, e.g. {"dwell_seconds": 30}
    is_active = Column(Boolean, default=True)
    name = Column(String(200))
    created_at = Column(String(40), default="")

    camera = relationship("Camera", back_populates="rules")

    def __repr__(self) -> str:
        return f"<Rule {self.id} name={self.name} type={self.rule_type}>"


class Alert(Base):
    """
    One sealed event in the audit log.

    ``prev_hash``/``hash`` chain every row to its predecessor: altering any
    hashed column of any row invalidates that row and everything after it.
    """

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    alert_type = Column(String(100), nullable=False)
    severity = Column(String(20), default=SEVERITY_MEDIUM)
    object_class = Column(String(50), default="")
    track_id = Column(Integer, default=0)
    confidence = Column(Float, default=0.0)

    #: Aware UTC ISO-8601 — canonical, sortable, hashed.
    timestamp = Column(String(50), nullable=False)
    #: Denormalised IST display string — what the operator actually reads.
    timestamp_ist = Column(String(64), default="")

    #: Which rule produced this event, and of what kind.
    rule_name = Column(String(200), default="")
    rule_type = Column(String(50), default="")
    #: 'ai_detection' for model output, 'rule_engine' for analytics rules.
    detector = Column(String(40), default="rule_engine")
    #: 'live' or 'upload'
    source_type = Column(String(20), default="live")
    #: Upload analysis session id, when source_type == 'upload'.
    session_id = Column(String(64), default="")
    #: Free-form JSON: plate text, watchlist name, dwell seconds, geometry…
    details_json = Column(Text, default="{}")
    description = Column(Text, default="")

    snapshot_path = Column(String(500), default="")
    clip_path = Column(String(500), default="")

    prev_hash = Column(String(64), nullable=False)
    hash = Column(String(64), nullable=False)

    camera = relationship("Camera", back_populates="alerts")

    __table_args__ = (
        Index("ix_alerts_timestamp", "timestamp"),
        Index("ix_alerts_camera_type", "camera_id", "alert_type"),
        Index("ix_alerts_session", "session_id"),
    )

    def __repr__(self) -> str:
        return f"<Alert {self.id} type={self.alert_type} cam={self.camera_id}>"


class Checkpoint(Base):
    """
    A periodic Merkle checkpoint over a contiguous range of the alert chain.

    This is **not** a public blockchain anchor — nothing is broadcast to any
    network.  It is a locally sealed checkpoint: a Merkle root over the alert
    hashes in ``[first_alert_id, last_alert_id]`` plus the chain tip at seal
    time.  Exported checkpoints let a third party verify a range of the log
    without being handed the whole database.
    """

    __tablename__ = "checkpoints"

    id = Column(Integer, primary_key=True)
    checkpoint_uid = Column(String(64), nullable=False, unique=True)
    first_alert_id = Column(Integer, default=0)
    last_alert_id = Column(Integer, default=0)
    alert_count = Column(Integer, default=0)
    merkle_root = Column(String(64), nullable=False)
    chain_tip_hash = Column(String(64), nullable=False)
    timestamp = Column(String(50), nullable=False)     # aware UTC ISO-8601
    timestamp_ist = Column(String(64), default="")

    def __repr__(self) -> str:
        return f"<Checkpoint {self.id} root={self.merkle_root[:12]}…>"


class WatchlistEntry(Base):
    """Persistent face watchlist entry (normalised ArcFace embedding)."""

    __tablename__ = "watchlist_entries"

    id = Column(Integer, primary_key=True)
    name = Column(String(120), nullable=False)
    embedding_json = Column(Text, nullable=False)
    metadata_json = Column(Text, default="{}")
    created_at = Column(String(64), default="")

    def __repr__(self) -> str:
        return f"<WatchlistEntry {self.id} name={self.name}>"


class AnalysisSession(Base):
    """An uploaded-MP4 analysis run, persisted so results survive a restart."""

    __tablename__ = "analysis_sessions"

    id = Column(Integer, primary_key=True)
    session_uid = Column(String(64), nullable=False, unique=True)
    filename = Column(String(300), nullable=False)
    stored_path = Column(String(500), nullable=False)
    output_path = Column(String(500), default="")
    #: The camera whose rules this run inherited, or the upload pseudo-source.
    #: This foreign key is what made "Remove Camera" fail with
    #: ``FOREIGN KEY constraint failed``: the ORM had no relationship for it, so
    #: no cascade applied and SQLite rejected the parent DELETE outright.
    camera_id = Column(Integer, ForeignKey("cameras.id"))

    #: queued | running | completed | failed | cancelled
    status = Column(String(20), default="queued")
    error = Column(Text, default="")

    total_frames = Column(Integer, default=0)
    processed_frames = Column(Integer, default=0)
    analysed_frames = Column(Integer, default=0)
    duration_seconds = Column(Float, default=0.0)
    source_fps = Column(Float, default=0.0)
    width = Column(Integer, default=0)
    height = Column(Integer, default=0)
    size_bytes = Column(Integer, default=0)

    detections_total = Column(Integer, default=0)
    persons_seen = Column(Integer, default=0)
    vehicles_seen = Column(Integer, default=0)
    alerts_generated = Column(Integer, default=0)
    processing_fps = Column(Float, default=0.0)

    created_at = Column(String(50), default="")
    created_at_ist = Column(String(64), default="")
    completed_at = Column(String(50), default="")
    completed_at_ist = Column(String(64), default="")

    camera = relationship("Camera", back_populates="sessions")

    __table_args__ = (Index("ix_sessions_created", "created_at"),)

    def __repr__(self) -> str:
        return f"<AnalysisSession {self.session_uid} {self.status}>"
