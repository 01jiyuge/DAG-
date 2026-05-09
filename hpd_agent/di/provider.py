from abc import ABC, abstractmethod
from typing import Any, Dict
from .container import DIContainer

class ServiceProvider(ABC):
    @abstractmethod
    def register(self, container: DIContainer) -> None:
        pass

class DefaultServiceProvider(ServiceProvider):
    def __init__(self, config: Any = None):
        self.config = config
    
    def register(self, container: DIContainer) -> None:
        if self.config:
            container.register("config", self.config)
        
        container.register_singleton("logger", lambda: self._create_logger())
    
    def _create_logger(self):
        from hpd_agent.utils.logger import get_logger
        return get_logger("default")

class CacheServiceProvider(ServiceProvider):
    def register(self, container: DIContainer) -> None:
        container.register_singleton("cache", self._create_cache)
    
    async def _create_cache(self):
        from hpd_agent.cache import CacheManager, CacheConfig
        
        container = DIContainer()
        config = container.get("config")
        
        cache_config = CacheConfig()
        if config:
            cache_config.backend = config.cache_backend
            cache_config.ttl_seconds = config.cache_ttl_seconds
            cache_config.max_size = config.cache_max_size
            cache_config.redis_url = config.redis_url
        
        return CacheManager(cache_config)

class SchedulerServiceProvider(ServiceProvider):
    def register(self, container: DIContainer) -> None:
        container.register_singleton("scheduler", self._create_scheduler)
    
    async def _create_scheduler(self):
        from hpd_agent.dag import DAGScheduler, SchedulerConfig
        
        container = DIContainer()
        config = container.get("config")
        
        scheduler_config = SchedulerConfig()
        if config:
            scheduler_config.max_workers = config.max_workers
            scheduler_config.default_timeout = config.timeout
            scheduler_config.retry_attempts = config.max_retries
        
        return DAGScheduler(scheduler_config)

class RouterServiceProvider(ServiceProvider):
    def register(self, container: DIContainer) -> None:
        container.register_singleton("router", self._create_router)
    
    def _create_router(self):
        from hpd_agent.routing import HierarchicalRouter
        return HierarchicalRouter()

class ExpertServiceProvider(ServiceProvider):
    def register(self, container: DIContainer) -> None:
        container.register_singleton("expert_executor", self._create_expert_executor)
    
    def _create_expert_executor(self):
        from hpd_agent.expert import ExpertExecutor, ExpertConfig
        return ExpertExecutor(ExpertConfig())

class MultimodalServiceProvider(ServiceProvider):
    def register(self, container: DIContainer) -> None:
        container.register_singleton("multimodal_processor", self._create_multimodal_processor)
    
    def _create_multimodal_processor(self):
        from hpd_agent.multimodal import MultimodalProcessor
        return MultimodalProcessor()

class LLMServiceProvider(ServiceProvider):
    def register(self, container: DIContainer) -> None:
        container.register_singleton("llm", self._create_llm)
    
    async def _create_llm(self):
        from hpd_agent.llm import LLMFactory, LLMConfig, LLMProvider
        
        container = DIContainer()
        config = container.get("config")
        
        if not config:
            return None
        
        api_key = None
        if config.llm_provider == "openai":
            api_key = config.openai_api_key
        elif config.llm_provider == "dashscope":
            api_key = config.dashscope_api_key
        elif config.llm_provider == "anthropic":
            api_key = config.anthropic_api_key
        elif config.llm_provider == "google":
            api_key = config.google_api_key
        elif config.llm_provider == "deepseek":
            api_key = config.deepseek_api_key
        
        if not api_key and config.llm_provider != "local":
            return None
        
        llm_config = LLMConfig(
            provider=LLMProvider(config.llm_provider),
            api_key=api_key,
            model=config.default_model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
            base_url=config.llm_base_url
        )
        
        return await LLMFactory.create(llm_config)

def register_default_providers(config: Any = None):
    container = DIContainer()
    
    providers = [
        DefaultServiceProvider(config),
        CacheServiceProvider(),
        SchedulerServiceProvider(),
        RouterServiceProvider(),
        ExpertServiceProvider(),
        MultimodalServiceProvider(),
        LLMServiceProvider()
    ]
    
    for provider in providers:
        provider.register(container)