from __future__ import annotations

import os

from pydantic import BaseModel, Field


class Settings(BaseModel):
    app_env: str = Field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    google_maps_api_key: str = Field(default_factory=lambda: os.getenv("GOOGLE_MAPS_API_KEY", ""))
    database_url: str = Field(default_factory=lambda: os.getenv("DATABASE_URL", ""))
    model_path: str = Field(default_factory=lambda: os.getenv("MODEL_PATH", "rl/checkpoints/best_model.pt"))
    qrdqn_model_path: str = Field(default_factory=lambda: os.getenv("QRDQN_MODEL_PATH", "rl/checkpoints/qrdqn_best_model.pt"))
    demo_mode: bool = Field(default_factory=lambda: os.getenv("DEMO_MODE", "true").lower() == "true")
    cors_origins: str = Field(default_factory=lambda: os.getenv("CORS_ORIGINS", "*"))
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # Proximity alert thresholds (Section 21) -- configurable, not hard-coded
    alert_distance_high_m: float = Field(default_factory=lambda: float(os.getenv("ALERT_DISTANCE_HIGH_M", "500")))
    alert_distance_moderate_m: float = Field(default_factory=lambda: float(os.getenv("ALERT_DISTANCE_MODERATE_M", "1500")))
    alert_cooldown_s: float = Field(default_factory=lambda: float(os.getenv("ALERT_COOLDOWN_S", "60")))

    # Hugging Face explanation layer (Section 18) -- explanation-only, never
    # controls PPO/QR-DQN/routes/vehicles/rewards. Disabled by default; the
    # rule-based explanation in sim_service.explain_episode() is always the
    # fallback and is itself never fabricated (built only from real frames).
    hf_enabled: bool = Field(default_factory=lambda: os.getenv("HF_ENABLED", "false").lower() == "true")
    hf_api_token: str = Field(default_factory=lambda: os.getenv("HF_API_TOKEN", ""))
    hf_model: str = Field(default_factory=lambda: os.getenv("HF_MODEL", "Qwen/Qwen2.5-3B-Instruct"))
    hf_timeout_seconds: float = Field(default_factory=lambda: float(os.getenv("HF_TIMEOUT_SECONDS", "12")))
    hf_max_new_tokens: int = Field(default_factory=lambda: int(os.getenv("HF_MAX_NEW_TOKENS", "400")))


def get_settings() -> Settings:
    # Re-read on every call so tests can monkeypatch os.environ freely.
    return Settings()
