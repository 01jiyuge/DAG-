from typing import Optional, Any, Dict, Union
from enum import Enum
from pydantic import BaseModel, Field
from hpd_agent.exceptions import ConfigurationError, LLMNotInitializedError
from hpd_agent.utils.logger import get_logger

class LLMProvider(str, Enum):
    OPENAI = "openai"
    DASHSCOPE = "dashscope"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"
    DEEPSEEK = "deepseek"

class LLMConfig(BaseModel):
    provider: LLMProvider = LLMProvider.OPENAI
    api_key: Optional[str] = None
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 4096
    base_url: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True

class LLMFactory:
    _instances: Dict[str, Any] = {}
    
    @classmethod
    async def create(cls, config: LLMConfig) -> Any:
        cache_key = f"{config.provider}:{config.model}:{config.api_key[:10] if config.api_key else 'no-key'}"
        
        if cache_key in cls._instances:
            return cls._instances[cache_key]
        
        logger = get_logger(__name__)
        llm = None
        
        if config.provider == LLMProvider.OPENAI:
            llm = await cls._create_openai_llm(config)
        
        elif config.provider == LLMProvider.DASHSCOPE:
            llm = await cls._create_dashscope_llm(config)
        
        elif config.provider == LLMProvider.ANTHROPIC:
            llm = await cls._create_anthropic_llm(config)
        
        elif config.provider == LLMProvider.GOOGLE:
            llm = await cls._create_google_llm(config)
        
        elif config.provider == LLMProvider.LOCAL:
            llm = await cls._create_local_llm(config)
        
        elif config.provider == LLMProvider.DEEPSEEK:
            llm = await cls._create_deepseek_llm(config)
        
        if llm is None:
            raise ConfigurationError(f"Unsupported LLM provider: {config.provider}")
        
        cls._instances[cache_key] = llm
        logger.info(f"Created {config.provider.value} LLM instance with model: {config.model}")
        
        return llm
    
    @classmethod
    async def _create_openai_llm(cls, config: LLMConfig) -> Any:
        if not config.api_key:
            raise ConfigurationError("OpenAI API key is required")
        
        try:
            try:
                from langchain_openai import ChatOpenAI
                params = {
                    "model": config.model,
                    "api_key": config.api_key,
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens
                }
                if config.base_url:
                    params["base_url"] = config.base_url
            except ImportError:
                from langchain.chat_models import ChatOpenAI
                params = {
                    "model_name": config.model,
                    "openai_api_key": config.api_key,
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens
                }
                if config.base_url:
                    params["openai_api_base"] = config.base_url
            
            return ChatOpenAI(**params)
        
        except ImportError as e:
            raise ConfigurationError(f"Failed to import OpenAI module: {e}")
    
    @classmethod
    async def _create_dashscope_llm(cls, config: LLMConfig) -> Any:
        if not config.api_key:
            raise ConfigurationError("DashScope API key is required")
        
        try:
            from langchain.chat_models import ChatDashScope
            
            return ChatDashScope(
                dashscope_api_key=config.api_key,
                model_name=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
        
        except ImportError as e:
            raise ConfigurationError(f"Failed to import DashScope module: {e}")
    
    @classmethod
    async def _create_anthropic_llm(cls, config: LLMConfig) -> Any:
        if not config.api_key:
            raise ConfigurationError("Anthropic API key is required")
        
        try:
            from langchain.chat_models import ChatAnthropic
            
            return ChatAnthropic(
                anthropic_api_key=config.api_key,
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
        
        except ImportError as e:
            raise ConfigurationError(f"Failed to import Anthropic module: {e}")
    
    @classmethod
    async def _create_google_llm(cls, config: LLMConfig) -> Any:
        try:
            from langchain.chat_models import ChatGooglePalm
            
            params = {
                "temperature": config.temperature,
                "max_tokens": config.max_tokens
            }
            
            if config.api_key:
                params["google_api_key"] = config.api_key
            
            return ChatGooglePalm(**params)
        
        except ImportError as e:
            raise ConfigurationError(f"Failed to import Google module: {e}")
    
    @classmethod
    async def _create_local_llm(cls, config: LLMConfig) -> Any:
        try:
            from langchain.chat_models import ChatOpenAI
            
            params = {
                "model_name": config.model,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "openai_api_base": config.base_url or "http://localhost:8000/v1"
            }
            
            return ChatOpenAI(**params)
        
        except ImportError as e:
            raise ConfigurationError(f"Failed to import OpenAI module for local LLM: {e}")
    
    @classmethod
    async def _create_deepseek_llm(cls, config: LLMConfig) -> Any:
        if not config.api_key:
            raise ConfigurationError("DeepSeek API key is required")
        
        try:
            try:
                from langchain_openai import ChatOpenAI
                params = {
                    "model": config.model,
                    "api_key": config.api_key,
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                    "base_url": config.base_url or "https://api.deepseek.com/v1"
                }
            except ImportError:
                from langchain.chat_models import ChatOpenAI
                params = {
                    "model_name": config.model,
                    "openai_api_key": config.api_key,
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                    "openai_api_base": config.base_url or "https://api.deepseek.com/v1"
                }
            
            return ChatOpenAI(**params)
        
        except ImportError as e:
            raise ConfigurationError(f"Failed to import OpenAI module for DeepSeek: {e}")
    
    @classmethod
    def clear_cache(cls):
        cls._instances.clear()

class AsyncLLMWrapper:
    def __init__(self, llm):
        self.llm = llm
    
    async def generate(self, prompts: Union[str, list]) -> str:
        if isinstance(prompts, str):
            prompts = [prompts]
        
        if hasattr(self.llm, 'agenerate'):
            from langchain_core.messages import HumanMessage, SystemMessage
            
            messages = [[SystemMessage(content="你是一个专业的AI助手"), HumanMessage(content=p)] for p in prompts]
            response = await self.llm.agenerate(messages)
            return response.generations[0][0].text
        
        elif hasattr(self.llm, 'generate'):
            from langchain_core.messages import HumanMessage, SystemMessage
            
            messages = [[SystemMessage(content="你是一个专业的AI助手"), HumanMessage(content=prompts[0])]]
            response = self.llm.generate(messages)
            return response.generations[0][0].text
        
        else:
            raise LLMNotInitializedError("LLM does not support generate method")