import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config.yaml"

def load_config() -> dict[str, Any]:
    """
    Load application confirguation for config.yml.
    """

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Confirguation file not found at : {CONFIG_PATH}"
        )

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not config:
        raise ValueError(f"Confirguation file is empty at : {CONFIG_PATH}")
    return config

def validate_config(config_data: dict[str, Any]) -> None:
        """
        Validate the minimum required gateway configuration.
        """
        required_sections = [
            "models",
            "caching",
            "routing_rules",
            "fallbacks",
        ]

        for section in required_sections:
            if section not in config_data:
                raise ValueError(
                    f"Missing required configuration section: {section}"
                )

        required_models = [
            "high_tier",
            "mid_tier",
            "low_tier",
            "embedding",
        ]

        for model in required_models:
            if model not in config_data["models"]:
                raise ValueError(
                    f"Missing required model configuration: {model}"
                )

        if "similarity_threshold" not in config_data["caching"]:
            raise ValueError(
                "Missing caching.similarity_threshold"
            )

        if not isinstance(
            config_data["caching"]["similarity_threshold"],
            (int, float),
        ):
            raise ValueError(
                "caching.similarity_threshold must be a number"
            )

        if not isinstance(config_data["routing_rules"], list):
            raise ValueError(
                "routing_rules must be a list"
            )

        if not isinstance(config_data["fallbacks"], list):
            raise ValueError(
                "fallbacks must be a list"
            )

config = load_config()
validate_config(config)

def get_env(name: str, default:str | None = None) -> str | None:
    """
    Get environment variable value.
    """
    value = os.getenv(name, default)
    if value is None:
        raise ValueError(f"Environment variable '{name}' is not set.")
    return value

# Provider API keys
OPENAI_API_KEY = get_env("OPENAI_API_KEY")
ANTHROPIC_API_KEY = get_env("ANTHROPIC_API_KEY")
GROQ_API_KEY = get_env("GROQ_API_KEY")


# Infrastructure configuration
POSTGRES_HOST = get_env("POSTGRES_HOST", "localhost")
POSTGRES_PORT = get_env("POSTGRES_PORT", "5432")
POSTGRES_USER = get_env("POSTGRES_USER", "gateway")
POSTGRES_PASSWORD = get_env("POSTGRES_PASSWORD", "gateway_password")
POSTGRES_DB = get_env("POSTGRES_DB", "llm_gateway")

REDIS_HOST = get_env("REDIS_HOST", "localhost")
REDIS_PORT = get_env("REDIS_PORT", "6379")