import pytest
import asyncio
from hpd_agent.di import DIContainer, inject
from hpd_agent.di.provider import register_default_providers
from hpd_agent.config import AgentConfig

@pytest.mark.asyncio
async def test_di_container():
    container = DIContainer()
    container.clear()
    
    container.register("service1", "value1")
    assert container.resolve("service1") == "value1"
    
    counter = [0]
    def factory_func():
        counter[0] += 1
        return {"value": counter[0]}
    container.register_factory("factory1", factory_func)
    result1 = container.resolve("factory1")
    result2 = container.resolve("factory1")
    assert result1["value"] == 1
    assert result2["value"] == 2
    assert result1 is not result2

@pytest.mark.asyncio
async def test_di_singleton():
    container = DIContainer()
    container.clear()
    
    counter = [0]
    container.register_singleton("singleton1", lambda: counter[0] + 1)
    
    result1 = container.resolve("singleton1")
    result2 = container.resolve("singleton1")
    assert result1 == result2

@pytest.mark.asyncio
async def test_inject_decorator():
    container = DIContainer()
    container.clear()
    
    container.register("dep1", "dependency_value")
    
    @inject
    async def test_func(dep1: str):
        return dep1
    
    result = await test_func()
    assert result == "dependency_value"

@pytest.mark.asyncio
async def test_register_default_providers():
    container = DIContainer()
    container.clear()
    
    config = AgentConfig()
    register_default_providers(config)
    
    assert container.get("config") is not None
    assert container.get("router") is not None
    assert container.get("expert_executor") is not None
    assert container.get("multimodal_processor") is not None