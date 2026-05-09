from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional, List
from dotenv import load_dotenv
import os

load_dotenv()

class AgentConfig(BaseSettings):
    openai_api_key: Optional[str] = Field(default=os.getenv("OPENAI_API_KEY"))
    dashscope_api_key: Optional[str] = Field(default=os.getenv("DASHSCOPE_API_KEY"))
    anthropic_api_key: Optional[str] = Field(default=os.getenv("ANTHROPIC_API_KEY"))
    google_api_key: Optional[str] = Field(default=os.getenv("GOOGLE_API_KEY"))
    deepseek_api_key: Optional[str] = Field(default=os.getenv("DEEPSEEK_API_KEY"))
    
    llm_provider: str = Field(default="openai", description="LLM provider: openai, dashscope, anthropic, google, local, deepseek")
    default_model: str = Field(default="gpt-4", description="Default LLM model")
    llm_temperature: float = Field(default=0.7, description="LLM temperature parameter")
    llm_max_tokens: int = Field(default=4096, description="LLM max tokens")
    llm_base_url: Optional[str] = Field(default=os.getenv("LLM_BASE_URL"), description="Custom LLM API base URL")
    
    max_workers: int = Field(default=10, description="Maximum number of parallel workers")
    timeout: int = Field(default=300, description="Task timeout in seconds")
    log_level: str = Field(default="INFO", description="Logging level")
    expert_mode_threshold: float = Field(default=0.7, description="Threshold for entering expert mode")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    
    cache_backend: str = Field(default="memory", description="Cache backend: memory or redis")
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")
    cache_max_size: int = Field(default=1000, description="Maximum cache entries")
    redis_url: Optional[str] = Field(default=os.getenv("REDIS_URL"), description="Redis connection URL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"