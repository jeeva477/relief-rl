"""SQLAlchemy-backed persistent hazard store.

The same HazardStore interface is used by the API in demo mode and when
DATABASE_URL is configured, so the frontend does not need to change.
"""
from __future__ import annotations
from sqlalchemy import select
from backend.app.db_models import HazardRecord
from backend.app.schemas.hazard import HazardOut, HazardType
from backend.app.schemas.location import LatLng
from backend.app.services.hazard_service import HazardStore

class SQLAlchemyHazardStore(HazardStore):
    def __init__(self, session_factory): self.session_factory = session_factory

    @staticmethod
    def _to_schema(row: HazardRecord) -> HazardOut:
        return HazardOut(id=row.id, location=LatLng(latitude=row.latitude, longitude=row.longitude), radius_m=row.radius_m, severity=row.severity, hazard_type=HazardType(row.hazard_type), hard_constraint=row.hard_constraint, source=row.source, active=row.active)

    def list_active(self) -> list[HazardOut]:
        with self.session_factory() as session:
            rows = session.scalars(select(HazardRecord).where(HazardRecord.active.is_(True))).all()
            return [self._to_schema(row) for row in rows]

    def list_all(self) -> list[HazardOut]:
        with self.session_factory() as session:
            return [self._to_schema(row) for row in session.scalars(select(HazardRecord)).all()]

    def add(self, hazard: HazardOut) -> HazardOut:
        with self.session_factory() as session:
            row = HazardRecord(id=hazard.id, latitude=hazard.location.latitude, longitude=hazard.location.longitude, radius_m=hazard.radius_m, severity=hazard.severity, hazard_type=hazard.hazard_type.value, hard_constraint=hazard.hard_constraint, source=hazard.source, active=hazard.active)
            session.add(row); session.commit(); session.refresh(row); return self._to_schema(row)

    def update(self, hazard_id: str, hazard: HazardOut) -> HazardOut | None:
        with self.session_factory() as session:
            row = session.get(HazardRecord, hazard_id)
            if row is None: return None
            row.latitude=hazard.location.latitude; row.longitude=hazard.location.longitude; row.radius_m=hazard.radius_m; row.severity=hazard.severity; row.hazard_type=hazard.hazard_type.value; row.hard_constraint=hazard.hard_constraint; row.source=hazard.source; row.active=hazard.active
            session.commit(); session.refresh(row); return self._to_schema(row)

    def deactivate(self, hazard_id: str) -> bool:
        with self.session_factory() as session:
            row=session.get(HazardRecord, hazard_id)
            if row is None: return False
            row.active=False; session.commit(); return True

    def delete(self, hazard_id: str) -> bool:
        with self.session_factory() as session:
            row=session.get(HazardRecord, hazard_id)
            if row is None: return False
            session.delete(row); session.commit(); return True
