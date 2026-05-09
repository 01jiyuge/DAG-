from typing import Any, Optional, Dict
import json
import asyncio

class RedisCache:
    def __init__(self, config):
        self.config = config
        self._client = None
        self._lock = asyncio.Lock()
    
    async def _get_client(self):
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    try:
                        import redis.asyncio as aioredis
                        self._client = await aioredis.from_url(self.config.redis_url or "redis://localhost")
                    except ImportError:
                        raise ImportError("redis package is required for Redis cache backend")
        return self._client
    
    async def get(self, key: str) -> Optional[Any]:
        client = await self._get_client()
        value = await client.get(f"{self.config.redis_prefix}{key}")
        if value is not None:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value.decode('utf-8')
        return None
    
    async def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        client = await self._get_client()
        serialized = json.dumps(value)
        await client.set(f"{self.config.redis_prefix}{key}", serialized, ex=ttl_seconds)
    
    async def delete(self, key: str) -> None:
        client = await self._get_client()
        await client.delete(f"{self.config.redis_prefix}{key}")
    
    async def clear(self, namespace: Optional[str] = None) -> None:
        client = await self._get_client()
        pattern = f"{self.config.redis_prefix}*"
        keys = await client.keys(pattern)
        if keys:
            await client.delete(*keys)
    
    async def exists(self, key: str) -> bool:
        client = await self._get_client()
        return await client.exists(f"{self.config.redis_prefix}{key}") > 0
    
    async def get_stats(self) -> Dict[str, Any]:
        client = await self._get_client()
        info = await client.info()
        return {
            "size": info.get("used_memory_human", "N/A"),
            "keys": info.get("db0", {}).get("keys", "N/A"),
            "hits": info.get("keyspace_hits", 0),
            "misses": info.get("keyspace_misses", 0)
        }