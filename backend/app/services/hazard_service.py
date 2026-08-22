"""Hazard persistence interface and development store."""

from __future__ import annotations
import abc
from backend.app.schemas.hazard import HazardOut

class HazardStore(abc.ABC):
    @abc.abstractmethod
    def list_active(self) -> list[HazardOut]: ...
    @abc.abstractmethod
    def list_all(self) -> list[HazardOut]: ...
    @abc.abstractmethod
    def add(self, hazard: HazardOut) -> HazardOut: ...
    @abc.abstractmethod
    def update(self, hazard_id: str, hazard: HazardOut) -> HazardOut | None: ...
    @abc.abstractmethod
    def deactivate(self, hazard_id: str) -> bool: ...
    @abc.abstractmethod
    def delete(self, hazard_id: str) -> bool: ...

class InMemoryHazardStore(HazardStore):
    def __init__(self):
        self._hazards: dict[str, HazardOut] = {}
    def list_active(self) -> list[HazardOut]:
        return [h for h in self._hazards.values() if h.active]
    def list_all(self) -> list[HazardOut]:
        return list(self._hazards.values())
    def add(self, hazard: HazardOut) -> HazardOut:
        self._hazards[hazard.id] = hazard
        return hazard
    def update(self, hazard_id: str, hazard: HazardOut) -> HazardOut | None:
        if hazard_id not in self._hazards: return None
        self._hazards[hazard_id] = hazard
        return hazard
    def deactivate(self, hazard_id: str) -> bool:
        if hazard_id in self._hazards:
            self._hazards[hazard_id].active = False
            return True
        return False
    def delete(self, hazard_id: str) -> bool:
        return self._hazards.pop(hazard_id, None) is not None

_default_store = InMemoryHazardStore()

def get_hazard_store() -> HazardStore:
    return _default_store
