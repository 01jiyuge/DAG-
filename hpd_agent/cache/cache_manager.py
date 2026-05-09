import asyncio
from typing import Any, Optional, Dict, Callable
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import hashlib
from hpd_agent.utils.logger import get_logger

class CacheBackend(str, Enum):
    MEMORY = "memory"
    REDIS = "redis"

class CacheConfig(BaseModel):
    backend: CacheBackend = CacheBackend.MEMORY
    ttl_seconds: int = 3600
    max_size: int = 1000
    redis_url: Optional[str] = None
    redis_prefix: str = "hpd_agent:"

class CacheManager:
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.logger = get_logger(__name__)
        self._backend = self._create_backend()
    
    def _create_backend(self):
        if self.config.backend == CacheBackend.REDIS:
            try:
                from .redis_cache import RedisCache
                return RedisCache(self.config)
            except ImportError:
                self.logger.warning("Redis not available, falling back to memory cache")
        
        from .memory_cache import MemoryCache
        return MemoryCache(self.config)
    
    def _generate_key(self, namespace: str, key: str) -> str:
        combined = f"{namespace}:{key}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    async def get(self, namespace: str, key: str) -> Optional[Any]:
        cache_key = self._generate_key(namespace, key)
        return await self._backend.get(cache_key)
    
    async def set(self, namespace: str, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        cache_key = self._generate_key(namespace, key)
        ttl = ttl_seconds or self.config.ttl_seconds
        await self._backend.set(cache_key, value, ttl)
    
    async def delete(self, namespace: str, key: str) -> None:
        cache_key = self._generate_key(namespace, key)
        await self._backend.delete(cache_key)
    
    async def clear(self, namespace: Optional[str] = None) -> None:
        await self._backend.clear(namespace)
    
    async def exists(self, namespace: str, key: str) -> bool:
        cache_key = self._generate_key(namespace, key)
        return await self._backend.exists(cache_key)
    
    async def get_or_set(self, namespace: str, key: str, func: Callable[[], Any], 
                        ttl_seconds: Optional[int] = None) -> Any:
        cached = await self.get(namespace, key)
        if cached is not None:
            self.logger.debug(f"Cache hit for {namespace}:{key}")
            return cached
        
        self.logger.debug(f"Cache miss for {namespace}:{key}")
        if asyncio.iscoroutinefunction(func):
            value = await func()
        else:
            value = func()
        await self.set(namespace, key, value, ttl_seconds)
        return value
    
    async def get_stats(self) -> Dict[str, Any]:
        return await self._backend.get_stats()