from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    DATABASE_URL: str = "postgresql://postgres:admin123@localhost:5432/soli_db"
    STORAGE_BASE_PATH: str = "./storage"

    # NVIDIA NIM — used by research agent for query expansion
    NVIDIA_API_KEY: str = ""
    NVIDIA_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
    NVIDIA_LLM_MODEL: str = "meta/llama-3.2-11b-vision-instruct"

    # Perplexica / SearXNG Search Engine URLs
    PERPLEXICA_API_URL: str = "http://localhost:3001"
    SEARXNG_API_URL: str = "http://localhost:8080"


settings = Settings()
