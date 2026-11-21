from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    app_name: str = "HaorWatch AI API"
    app_version: str = "0.1.0"
    api_v1_prefix: str = "/api/v1"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/haorwatch"
    cors_origins: str = "http://localhost:5173"
    data_root: str = "./data"
    model_root: str = "./models"
    default_study_area_slug: str = "hakaluki-haor"
    operator_api_key: str = "replace-with-long-random-secret"
    stale_after_days: int = 45

    # JRC permanent-water occurrence threshold (percent, 0-100). Configurable so
    # the permanent-water mask rule is explicit and adjustable per deployment.
    permanent_water_occurrence_threshold: float = 95.0
    # ESA WorldCover class IDs treated as mapped (non-crop-specific) cropland.
    agricultural_class_ids: str = "40"

    google_earth_engine_project: str = ""
    copernicus_client_id: str = ""
    copernicus_client_secret: str = ""

    llm_provider: str = "none"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def agricultural_class_list(self) -> list[int]:
        return [int(v.strip()) for v in self.agricultural_class_ids.split(",") if v.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
