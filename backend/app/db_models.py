"""Database models for persistent disaster hazards."""

from __future__ import annotations

from sqlalchemy import Boolean, Float, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db import Base


class HazardRecord(Base):
    __tablename__ = "hazards"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    radius_m: Mapped[float] = mapped_column(Float, nullable=False)
    severity: Mapped[float] = mapped_column(Float, nullable=False)
    hazard_type: Mapped[str] = mapped_column(String(64), nullable=False)
    hard_constraint: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False, default="manual")
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
