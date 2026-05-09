import pytest
import asyncio
from hpd_agent.cache import CacheManager, CacheConfig, CacheBackend

@pytest.mark.asyncio
async def test_memory_cache():
    config = CacheConfig(backend=CacheBackend.MEMORY, ttl_seconds=60, max_size=100)
    cache = CacheManager(config)
    
    await cache.set("test", "key1", "value1")
    result = await cache.get("test", "key1")
    assert result == "value1"
    
    await cache.set("test", "key2", {"nested": "data"})
    result = await cache.get("test", "key2")
    assert result == {"nested": "data"}
    
    await cache.delete("test", "key1")
    result = await cache.get("test", "key1")
    assert result is None

@pytest.mark.asyncio
async def test_cache_ttl():
    config = CacheConfig(backend=CacheBackend.MEMORY, ttl_seconds=1, max_size=100)
    cache = CacheManager(config)
    
    await cache.set("test", "ttl_key", "value")
    result = await cache.get("test", "ttl_key")
    assert result == "value"
    
    await asyncio.sleep(1.5)
    result = await cache.get("test", "ttl_key")
    assert result is None

@pytest.mark.asyncio
async def test_cache_get_or_set():
    config = CacheConfig(backend=CacheBackend.MEMORY)
    cache = CacheManager(config)
    
    call_count = [0]
    
    async def getter():
        call_count[0] += 1
        return "computed_value"
    
    result = await cache.get_or_set("test", "compute_key", getter)
    assert result == "computed_value"
    assert call_count[0] == 1
    
    result = await cache.get_or_set("test", "compute_key", getter)
    assert result == "computed_value"
    assert call_count[0] == 1

@pytest.mark.asyncio
async def test_cache_stats():
    config = CacheConfig(backend=CacheBackend.MEMORY)
    cache = CacheManager(config)
    
    await cache.set("stats", "key1", "value1")
    await cache.get("stats", "key1")
    await cache.get("stats", "key_nonexistent")
    
    stats = await cache.get_stats()
    assert "size" in stats
    assert "hits" in stats
    assert "misses" in stats
    assert stats["hits"] >= 1
    assert stats["misses"] >= 1