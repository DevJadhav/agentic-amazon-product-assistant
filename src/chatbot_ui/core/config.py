"""
Configuration management for the AI-Powered Amazon Product Assistant.
Implements singleton pattern with validation and environment-specific settings.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator, SecretStr
from typing import Optional, Dict, Any
from functools import lru_cache
import os
from enum import Enum


class Environment(str, Enum):
    """Supported environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    DOCKER = "docker"


class Config(BaseSettings):
    """Application configuration with validation and type safety."""
    
    # Environment
    ENVIRONMENT: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Application environment"
    )
    
    # API Keys - Using SecretStr for security
    OPENAI_API_KEY: SecretStr = Field(
        ...,
        description="OpenAI API key for GPT models"
    )
    GROQ_API_KEY: SecretStr = Field(
        ...,
        description="Groq API key for Llama models"
    )
    GOOGLE_API_KEY: SecretStr = Field(
        ...,
        description="Google API key for Gemini models"
    )
    LANGSMITH_API_KEY: Optional[SecretStr] = Field(
        default=None,
        description="LangSmith API key for tracing"
    )
    
    # Service URLs
    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
        description="Ollama service base URL"
    )
    
    # Vector Database Configuration
    WEAVIATE_HOST: Optional[str] = Field(
        default=None,
        description="Weaviate host for vector database"
    )
    WEAVIATE_PORT: int = Field(
        default=8080,
        description="Weaviate HTTP port"
    )
    WEAVIATE_GRPC_PORT: int = Field(
        default=50051,
        description="Weaviate gRPC port"
    )
    
    # Performance Configuration
    MAX_TOKENS_DEFAULT: int = Field(
        default=512,
        ge=1,
        le=4096,
        description="Default max tokens for LLM responses"
    )
    TEMPERATURE_DEFAULT: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Default temperature for LLM responses"
    )
    
    # Feature Flags
    ENABLE_CACHING: bool = Field(
        default=True,
        description="Enable response caching"
    )
    ENABLE_RATE_LIMITING: bool = Field(
        default=True,
        description="Enable API rate limiting"
    )
    
    # Logging Configuration
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="forbid"  # Fail on unknown fields
    )
    
    @field_validator("ENVIRONMENT", mode="before")
    @classmethod
    def detect_docker_environment(cls, v: str) -> str:
        """Automatically detect Docker environment."""
        if os.getenv("WEAVIATE_HOST") is not None:
            return Environment.DOCKER
        return v
    
    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is valid."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == Environment.PRODUCTION
    
    @property
    def is_docker(self) -> bool:
        """Check if running in Docker environment."""
        return self.ENVIRONMENT == Environment.DOCKER or self.WEAVIATE_HOST is not None
    
    @property
    def api_keys_dict(self) -> Dict[str, Optional[str]]:
        """Get all API keys as a dictionary (careful with security)."""
        return {
            "openai": self.OPENAI_API_KEY.get_secret_value(),
            "groq": self.GROQ_API_KEY.get_secret_value(),
            "google": self.GOOGLE_API_KEY.get_secret_value(),
            "langsmith": self.LANGSMITH_API_KEY.get_secret_value() if self.LANGSMITH_API_KEY else None
        }
    
    def validate_required_keys(self) -> None:
        """Validate that at least one LLM API key is provided."""
        if not any([
            self.OPENAI_API_KEY,
            self.GROQ_API_KEY,
            self.GOOGLE_API_KEY,
            self.OLLAMA_BASE_URL
        ]):
            raise ValueError("At least one LLM provider API key must be configured")
    
    def get_weaviate_url(self) -> str:
        """Get the Weaviate connection URL."""
        if self.WEAVIATE_HOST:
            return f"http://{self.WEAVIATE_HOST}:{self.WEAVIATE_PORT}"
        return ""
    
    def to_dict(self, include_secrets: bool = False) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        data = {
            "environment": self.ENVIRONMENT,
            "is_docker": self.is_docker,
            "weaviate_url": self.get_weaviate_url(),
            "max_tokens_default": self.MAX_TOKENS_DEFAULT,
            "temperature_default": self.TEMPERATURE_DEFAULT,
            "enable_caching": self.ENABLE_CACHING,
            "enable_rate_limiting": self.ENABLE_RATE_LIMITING,
            "log_level": self.LOG_LEVEL
        }
        
        if include_secrets:
            data["api_keys"] = self.api_keys_dict
            
        return data


@lru_cache()
def get_config() -> Config:
    """
    Get singleton configuration instance.
    Uses LRU cache to ensure single instance.
    """
    config = Config()
    config.validate_required_keys()
    return config


# Global config instance
config = get_config()