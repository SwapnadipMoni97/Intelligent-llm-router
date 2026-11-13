from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Intelligent LLM Router"
    routing_strategy: str = "intelligent"
    model_registry_path: str = "config/models.yaml"
    request_timeout_seconds: float = 45.0
    max_retries: int = 2
    retry_base_seconds: float = 0.25
    rate_limit_requests_per_minute: int = 60
    structured_logs: bool = True
    circuit_failure_threshold: int = 3
    circuit_recovery_seconds: float = 30.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
