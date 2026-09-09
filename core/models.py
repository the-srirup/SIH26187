"""
SQLAlchemy ORM models.

Three tables — cameras, rules, alerts — match the schema recommended
in the hackathon build guide.  Each alert includes a *prev_hash* column
so the full alert history forms a tamper-evident hash chain.
"""
from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from core.database import Base


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True)
    name = Column(String(120), nullable=False)
    url = Column(String(500), nullable=False)
    location = Column(String(200))
    is_active = Column(Boolean, default=True)

    rules = relationship("Rule", back_populates="camera", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="camera", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Camera {self.id} {self.name}>"


class Rule(Base):
    __tablename__ = "rules"

    id = Column(Integer, primary_key=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    rule_type = Column(String(50), nullable=False)  # 'line', 'zone', 'loiter', 'wrong_direction'
    geometry = Column(Text)  # JSON coordinates
    params = Column(Text)  # JSON parameters, e.g. {"dwell_seconds": 60}
    is_active = Column(Boolean, default=True)
    name = Column(String(200))

    camera = relationship("Camera", back_populates="rules")

    def __repr__(self) -> str:
        return f"<Rule {self.id} name={self.name} type={self.rule_type}>"


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    alert_type = Column(String(100), nullable=False)
    object_class = Column(String(50))
    track_id = Column(Integer)
    confidence = Column(Float)
    timestamp = Column(String(50), nullable=False)  # ISO-8601 string
    snapshot_path = Column(String(500))
    clip_path = Column(String(500))
    prev_hash = Column(String(64), nullable=False)
    hash = Column(String(64), nullable=False)

    camera = relationship("Camera", back_populates="alerts")

    def __repr__(self) -> str:
        return f"<Alert {self.id} type={self.alert_type} cam={self.camera_id}>"
