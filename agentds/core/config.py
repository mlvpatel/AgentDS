"""
Personal Data Scientist Configuration Management.

This module provides centralized configuration management using Pydantic Settings.
Supports environment variables, .env files, and YAML configuration.

Author: Malav Patel
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """LLM-specific settings."""

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        extra="ignore",
    )

    # API Keys
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    mistral_api_key: str | None = Field(default=None, alias="MISTRAL_API_KEY")
    together_api_key: str | None = Field(default=None, alias="TOGETHERAI_API_KEY")
    deepseek_api_key: str | None = Field(default=None, alias="DEEPSEEK_API_KEY")
    xai_api_key: str | None = Field(default=None, alias="XAI_API_KEY")
    cohere_api_key: str | None = Field(default=None, alias="COHERE_API_KEY")

    # Azure OpenAI
    azure_api_key: str | None = Field(default=None, alias="AZURE_API_KEY")
    azure_api_base: str | None = Field(default=None, alias="AZURE_API_BASE")
    azure_api_version: str = Field(default="2024-02-01", alias="AZURE_API_VERSION")

    # Google Vertex AI
    vertexai_project: str | None = Field(default=None, alias="VERTEXAI_PROJECT")
    vertexai_location: str = Field(default="us-central1", alias="VERTEXAI_LOCATION")

    # AWS Bedrock
    aws_access_key_id: str | None = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str | None = Field(
        default=None, alias="AWS_SECRET_ACCESS_KEY"
    )
    aws_region_name: str = Field(default="us-east-1", alias="AWS_REGION_NAME")

    # Ollama (local)
    ollama_api_base: str = Field(
        default="http://localhost:11434", alias="OLLAMA_API_BASE"
    )

    # Default settings
    default_model: str = Field(default="openai/gpt-4o-mini")
    default_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_retries: int = Field(default=3, ge=0)
    request_timeout: int = Field(default=120, ge=10)

    def get_available_providers(self) -> list[str]:
        """Return list of configured providers with valid API keys."""
        providers = []
        if self.openai_api_key:
            providers.append("openai")
        if self.anthropic_api_key:
            providers.append("anthropic")
        if self.gemini_api_key:
            providers.append("gemini")
        if self.groq_api_key:
            providers.append("groq")
        if self.mistral_api_key:
            providers.append("mistral")
        if self.together_api_key:
            providers.append("together")
        if self.deepseek_api_key:
            providers.append("deepseek")
        if self.xai_api_key:
            providers.append("xai")
        if self.cohere_api_key:
            providers.append("cohere")
        if self.azure_api_key and self.azure_api_base:
            providers.append("azure")
        if self.vertexai_project:
            providers.append("vertex_ai")
        if self.aws_access_key_id and self.aws_secret_access_key:
            providers.append("bedrock")
        # Ollama is always available (local)
        providers.append("ollama")
        return providers


class RedisSettings(BaseSettings):
    """Redis-specific settings."""

    model_config = SettingsConfigDict(
        env_prefix="REDIS_",
        extra="ignore",
    )

    url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    max_connections: int = Field(default=10)
    socket_timeout: int = Field(default=5)
    retry_on_timeout: bool = Field(default=True)


class MLflowSettings(BaseSettings):
    """MLflow-specific settings."""

    model_config = SettingsConfigDict(
        env_prefix="MLFLOW_",
        extra="ignore",
    )

    tracking_uri: str = Field(
        default="http://localhost:5000", alias="MLFLOW_TRACKING_URI"
    )
    experiment_name: str = Field(default="agentds")
    artifact_location: str | None = Field(default=None)


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application
    app_name: str = Field(default="Personal Data Scientist")
    app_version: str = Field(default="1.0.0")
    debug: bool = Field(default=False, alias="DEBUG")
    environment: str = Field(default="development", alias="ENVIRONMENT")

    # Server
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=7860, alias="PORT")

    # Paths
    config_dir: Path = Field(default=Path("config"))
    output_dir: Path = Field(default=Path("outputs"))
    temp_dir: Path = Field(default=Path("temp"))
    log_dir: Path = Field(default=Path("logs"))
    checkpoint_dir: Path = Field(default=Path("checkpoints"))

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="json", alias="LOG_FORMAT")

    # Security
    api_key_header: str = Field(default="X-API-Key")
    api_keys: list[str] = Field(default_factory=list)

    # Rate limiting
    rate_limit_per_minute: int = Field(default=60)

    # Human-in-the-loop
    human_in_loop: bool = Field(default=True, alias="HUMAN_IN_LOOP")
    auto_approve_low_risk: bool = Field(default=False)

    # Nested settings
    llm: LLMSettings = Field(default_factory=LLMSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)

    # Configuration files
    llm_config_path: Path = Field(default=Path("config/llm_config.yaml"))
    pipeline_config_path: Path = Field(default=Path("config/pipeline_config.yaml"))
    feature_flags_path: Path = Field(default=Path("config/feature_flags.yaml"))

    @field_validator("api_keys", mode="before")
    @classmethod
    def parse_api_keys(cls, v: Any) -> list[str]:
        """Parse API keys from comma-separated string."""
        if isinstance(v, str):
            return [k.strip() for k in v.split(",") if k.strip()]
        return v or []

    @model_validator(mode="after")
    def create_directories(self) -> Settings:
        """Create necessary directories if they don't exist."""
        for dir_path in [
            self.output_dir,
            self.temp_dir,
            self.log_dir,
            self.checkpoint_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
        return self

    def load_yaml_config(self, path: Path) -> dict[str, Any]:
        """Load configuration from YAML file."""
        if not path.exists():
            return {}
        with open(path) as f:
            return yaml.safe_load(f) or {}

    def get_llm_config(self) -> dict[str, Any]:
        """Load LLM configuration from YAML."""
        return self.load_yaml_config(self.llm_config_path)

    def get_pipeline_config(self) -> dict[str, Any]:
        """Load pipeline configuration from YAML."""
        return self.load_yaml_config(self.pipeline_config_path)

    def get_feature_flags(self) -> dict[str, Any]:
        """Load feature flags from YAML."""
        config = self.load_yaml_config(self.feature_flags_path)
        # Apply environment-specific overrides
        env_overrides = config.get("environments", {}).get(self.environment, {})
        features = config.get("features", {})
        for key, value in env_overrides.items():
            if key in features:
                features[key]["enabled"] = value
        return features

    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature flag is enabled."""
        features = self.get_feature_flags()
        feature = features.get(feature_name, {})
        return feature.get("enabled", False)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def reload_settings() -> Settings:
    """Reload settings (clears cache)."""
    get_settings.cache_clear()
    return get_settings()
