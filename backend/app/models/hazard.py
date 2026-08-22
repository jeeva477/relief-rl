from __future__ import annotations
from sqlalchemy import Boolean, Float, String
from sqlalchemy.orm import Mapped, mapped_column
from geoalchemy2 import Geography
from backend.app.db import Base

class HazardRecord(Base):
    __tablename__ = "hazards"
    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    radius_m: Mapped[float] = mapped_column(Float, nullable=False)
    severity: Mapped[float] = mapped_column(Float, nullable=False)
    hazard_type: Mapped[str] = mapped_column(String(64), nullable=False)
    hard_constraint: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    source: Mapped[str] = mapped_column(String(64), default="manual", nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    location = mapped_column(Geography(geometry_type="POINT", srid=4326), nullable=True)
