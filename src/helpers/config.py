from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # OpenAI API key
    OPENAI_API_KEY: Optional[str] = None
    LLM_MODEL: Optional[str] = None  # Provide a default to avoid required field error
    MODEL_ID2: Optional[str] = None
    OLLAMA_API_GENERATION: str = "http://localhost:11434/api/generate"  # Default value
    OLLAMA_API_CHAT: str = "http://localhost:11434/api/chat"  # Default value

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Allow extra fields for debugging
        case_sensitive = False  # Handle case variations

def get_settings() -> Settings:
    """
    Get the settings from the environment variables.
    """
    return Settings()