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
    tokenizer_path: Path = SERVER_DIR.parent / "lisa_tokenizer.bin"        # new models (tools/build_lisa_tokenizer.py)
    legacy_tokenizer_path: Path = SERVER_DIR.parent / "tokenizer.bin"      # models without IMAGE_FLAG_TOK_LISA
    bootstrap_image_path: Path | None = None           # used when no model version exists yet

    # --- Evolution ---
    trainer: Literal["mock", "torch", "remote"] = "mock"
    remote_trainer_url: str | None = None
    remote_trainer_token: SecretStr | None = None
    teacher: Literal["rule", "claude", "groq"] = "rule"
    teacher_model: str | None = None                   # default: claude-opus-5-5 / openai/gpt-oss-120b (groq)
    groq_api_key: SecretStr | None = None              # TEACHER=groq (free plan at console.groq.com)
    teacher_base_url: str = "https://api.groq.com/openai/v1"   # TEACHER=groq: any OpenAI-compatible API
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
    #   Continuing from the active model (the normal run), 1500 steps at 5e-4 learned new topics
    #   without forgetting (chat 0.516 -> 0.539); 2000 at 1e-3 lost 3/128 old answers twice.
    #   Retraining from scratch (worker --base-version 1 --unfreeze-embeddings): --steps 3000 --lr 1e-3.
    train_steps: int = 1500
    train_batch_size: int = 16
    train_lr: float = 5e-4
    train_schedule: Literal["constant", "cosine"] = "cosine"
    train_warmup_steps: int = 60
    train_method: Literal["full", "lora"] = "full"
    lora_rank: int = 8
    freeze_embeddings: bool = True                     # keeps OTA deltas ~2 MiB smaller
    chat_enabled: bool = True                          # train the conversation corpus (chat_corpus.py) too
    teacher_lessons_per_run: int = 5                   # new topics the Claude teacher writes per run (1 call)
    gate_max_task_regression: float = 0.05             # no task (actions, chat) may lose more exact match than this
    gate_eval_per_task: int = 128                      # validation examples decoded per task for exact match
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
        if self.teacher == "groq" and not (self.groq_api_key and self.groq_api_key.get_secret_value()):
            raise ValueError("TEACHER=groq needs GROQ_API_KEY (create one at console.groq.com -> API Keys)")
        if not self.dev_mode:
            if self.supabase_service_role_key is None:
                raise ValueError("SUPABASE_SERVICE_ROLE_KEY is required when SUPABASE_URL is set")
            if self.admin_token.get_secret_value() == DEFAULT_ADMIN_TOKEN:
                raise ValueError("Set ADMIN_TOKEN to a strong secret before connecting to Supabase")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
