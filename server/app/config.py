"""Settings, read from environment variables (or server/.env)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_ADMIN_TOKEN = "dev-admin-token"
SERVER_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=SERVER_DIR / ".env", extra="ignore")

    ai_name: str = "LISA"

    # --- Supabase (unset = dev mode: in-memory database + local files) ---
    supabase_url: str | None = None
    supabase_service_role_key: SecretStr | None = None

    # --- Security ---
    admin_token: SecretStr = SecretStr(DEFAULT_ADMIN_TOKEN)
    ota_signing_key_pem: SecretStr | None = None      # PEM text, or ...
    ota_signing_key_path: Path | None = None           # ... a file with it

    # --- Local storage (dev mode blobs, generated keys) ---
    data_dir: Path = SERVER_DIR / ".data"
    public_base_url: str = "http://localhost:8000"     # used for local signed download URLs

    # --- Model / tokenizer ---
    tokenizer_path: Path = SERVER_DIR.parent / "tokenizer.bin"
    bootstrap_image_path: Path | None = None           # used when no model version exists yet

    # --- Evolution ---
    trainer: Literal["mock", "torch", "remote"] = "mock"
    remote_trainer_url: str | None = None
    remote_trainer_token: SecretStr | None = None
    teacher: Literal["rule", "claude"] = "rule"
    teacher_model: str = "claude-opus-5-5"
    teacher_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    teacher_max_calls_per_run: int = 50
    run_evolution_inline: bool | None = None           # None: inline in dev mode, worker otherwise
    evolution_min_new_logs: int = 200                  # worker: start a run after this many new interactions
    evolution_interval_s: int = 6 * 3600               # worker: or after this long, if anything is new
    evolution_min_gap_s: int = 600                     # worker: never start runs closer together than this
    worker_poll_s: int = 30
    curate_batch: int = 5000
    synth_per_device: int = 400                        # ~400 spread over the range: 0.00 -> 0.85 (tests/test_model.py)
    min_train_examples: int = 32
    # Measured on the real sim-thermo dataset (374 train / 35 val), 1200 steps from v1:
    #   1e-3 cosine -> 0.97 exact match;  3e-3 cosine -> 0.57-0.66 (memorizes);  3e-3 constant -> 0.00-0.34
    train_steps: int = 1200
    train_batch_size: int = 16
    train_lr: float = 1e-3
    train_schedule: Literal["constant", "cosine"] = "cosine"
    train_warmup_steps: int = 60
    train_method: Literal["full", "lora"] = "full"
    lora_rank: int = 8
    freeze_embeddings: bool = True                     # keeps OTA deltas ~2 MiB smaller
    gate_min_improvement: float = 0.0                  # exact-match gain needed to publish
    gate_min_val_examples: int = 20                    # fewer can't separate learning from "always the majority answer"
    auto_activate: bool = True
    delta_max_ratio: float = 0.7                       # publish a delta only if it's this much smaller

    @property
    def dev_mode(self) -> bool:
        return not self.supabase_url

    @property
    def evolution_inline(self) -> bool:
        return self.dev_mode if self.run_evolution_inline is None else self.run_evolution_inline

    @model_validator(mode="after")
    def _production_guards(self) -> "Settings":
        if not self.dev_mode:
            if self.supabase_service_role_key is None:
                raise ValueError("SUPABASE_SERVICE_ROLE_KEY is required when SUPABASE_URL is set")
            if self.admin_token.get_secret_value() == DEFAULT_ADMIN_TOKEN:
                raise ValueError("Set ADMIN_TOKEN to a strong secret before connecting to Supabase")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
