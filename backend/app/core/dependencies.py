"""Shared, process-wide dependencies."""

from __future__ import annotations

import os
from functools import lru_cache

import torch

from backend.app.config import Settings, get_settings
from backend.app.db import Base, create_engine_and_session
from backend.app.services.google_maps import GoogleMapsProvider, get_provider
from backend.app.services.hazard_service import HazardStore, get_hazard_store
from backend.app.services.sql_hazard_service import SQLAlchemyHazardStore
from rl.envs.evacuation_env import N_ACTIONS, OBS_DIM
from rl.models.actor_critic import ActorCritic
from rl.models.qrdqn_net import QuantileNetwork


class ModelHandle:
    def __init__(self, model: ActorCritic | None, model_name: str = "SafeRoute-ActorCritic",
                 model_version: str = "1.0", incompatible_reason: str | None = None):
        self.model = model
        self.model_name = model_name
        self.model_version = model_version
        self.incompatible_reason = incompatible_reason

    @property
    def available(self) -> bool:
        return self.model is not None

    @property
    def compatible(self) -> bool:
        return self.model is not None and self.incompatible_reason is None


@lru_cache(maxsize=1)
def get_model_handle() -> ModelHandle:
    settings = get_settings()
    if not os.path.exists(settings.model_path):
        return ModelHandle(model=None, incompatible_reason="No checkpoint file found")
    try:
        ckpt = torch.load(settings.model_path, map_location="cpu", weights_only=False)
        ckpt_obs_dim = int(ckpt.get("obs_dim", 0))
        ckpt_n_actions = int(ckpt.get("n_actions", 0))
        if ckpt_obs_dim != OBS_DIM or ckpt_n_actions != N_ACTIONS:
            return ModelHandle(
                model=None,
                incompatible_reason=(
                    f"Checkpoint schema mismatch: obs_dim={ckpt_obs_dim}/n_actions={ckpt_n_actions} "
                    f"(expected obs_dim={OBS_DIM}/n_actions={N_ACTIONS}). Retrain with the current env."
                ),
            )
        model = ActorCritic(
            obs_dim=ckpt_obs_dim,
            n_actions=ckpt_n_actions,
            hidden_dim=ckpt.get("hidden_dim", 128),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return ModelHandle(
            model=model,
            model_name=ckpt.get("model_name", "SafeRoute-ActorCritic"),
            model_version=ckpt.get("model_version", "1.0"),
        )
    except Exception:
        return ModelHandle(model=None, incompatible_reason="Checkpoint could not be loaded (corrupt or incompatible)")


def reload_model() -> ModelHandle:
    """Re-read the model file and clear the cached handle (used after training/loading)."""
    get_model_handle.cache_clear()
    return get_model_handle()


class QRDQNModelHandle:
    """Mirrors ModelHandle for the QR-DQN checkpoint so the two agents can be
    loaded, reported on, and compared with the same honesty guarantees
    (never fabricate a model; report the real incompatibility reason)."""

    def __init__(self, model: QuantileNetwork | None, model_name: str = "ReliefRL-QRDQN",
                 model_version: str = "1.0", incompatible_reason: str | None = None):
        self.model = model
        self.model_name = model_name
        self.model_version = model_version
        self.incompatible_reason = incompatible_reason

    @property
    def available(self) -> bool:
        return self.model is not None

    @property
    def compatible(self) -> bool:
        return self.model is not None and self.incompatible_reason is None


@lru_cache(maxsize=1)
def get_qrdqn_model_handle() -> QRDQNModelHandle:
    settings = get_settings()
    path = settings.qrdqn_model_path
    if not os.path.exists(path):
        return QRDQNModelHandle(model=None, incompatible_reason="No QR-DQN checkpoint file found")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        ckpt_obs_dim = int(ckpt.get("obs_dim", 0))
        ckpt_n_actions = int(ckpt.get("n_actions", 0))
        if ckpt_obs_dim != OBS_DIM or ckpt_n_actions != N_ACTIONS:
            return QRDQNModelHandle(
                model=None,
                incompatible_reason=(
                    f"Checkpoint schema mismatch: obs_dim={ckpt_obs_dim}/n_actions={ckpt_n_actions} "
                    f"(expected obs_dim={OBS_DIM}/n_actions={N_ACTIONS}). Retrain with the current env."
                ),
            )
        model = QuantileNetwork(
            obs_dim=ckpt_obs_dim,
            n_actions=ckpt_n_actions,
            n_quantiles=ckpt.get("n_quantiles", 51),
            hidden_dim=ckpt.get("hidden_dim", 128),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return QRDQNModelHandle(
            model=model,
            model_name=ckpt.get("model_name", "ReliefRL-QRDQN"),
            model_version=ckpt.get("model_version", "1.0"),
        )
    except Exception:
        return QRDQNModelHandle(model=None, incompatible_reason="Checkpoint could not be loaded (corrupt or incompatible)")


def reload_qrdqn_model() -> QRDQNModelHandle:
    """Re-read the QR-DQN checkpoint and clear the cached handle."""
    get_qrdqn_model_handle.cache_clear()
    return get_qrdqn_model_handle()


def get_maps_provider(settings: Settings | None = None) -> GoogleMapsProvider:
    settings = settings or get_settings()
    return get_provider(settings.google_maps_api_key, settings.demo_mode)


@lru_cache(maxsize=1)
def get_hazards() -> HazardStore:
    """Select persistent storage when DATABASE_URL is configured.

    With no DATABASE_URL, DEMO_MODE keeps the original in-memory behavior.
    """
    settings = get_settings()
    if settings.database_url:
        engine, factory = create_engine_and_session(settings.database_url)
        Base.metadata.create_all(bind=engine)
        return SQLAlchemyHazardStore(factory)
    return get_hazard_store()


def clear_model_cache() -> None:
    get_model_handle.cache_clear()
    get_qrdqn_model_handle.cache_clear()


def clear_hazard_cache() -> None:
    get_hazards.cache_clear()
