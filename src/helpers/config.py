from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # OpenAI API key
    OPENAI_API_KEY: Optional[str] = None 
    LLM_MODEL: Optional[str] = "gpt-4o-mini"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

def get_settings() -> Settings:
    """
    Get the settings from the environment variables.
    """
    return Settings()


