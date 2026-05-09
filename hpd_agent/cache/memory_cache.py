from typing import Any, Optional, Dict, Tuple
import asyncio
from datetime import datetime, timedelta
import time

class MemoryCache:
    def __init__(self, config):
        self.config = config
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key in self._cache:
                value, expire_at = self._cache[key]
                if time.time() < expire_at:
                    self._hits += 1
                    return value
                else:
                    del self._cache[key]
        
        self._misses += 1
        return None
    
    async def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        async with self._lock:
            if len(self._cache) >= self.config.max_size:
                self._evict_oldest()
            
            expire_at = time.time() + ttl_seconds
            self._cache[key] = (value, expire_at)
    
    async def delete(self, key: str) -> None:
        async with self._lock:
            self._cache.pop(key, None)
    
    async def clear(self, namespace: Optional[str] = None) -> None:
        async with self._lock:
            self._cache.clear()
    
    async def exists(self, key: str) -> bool:
        async with self._lock:
            if key in self._cache:
                _, expire_at = self._cache[key]
                if time.time() < expire_at:
                    return True
                else:
                    del self._cache[key]
        return False
    
    def _evict_oldest(self):
        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
        del self._cache[oldest_key]
        self._evictions += 1
    
    async def get_stats(self) -> Dict[str, Any]:
        async with self._lock:
            return {
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0.0
            }