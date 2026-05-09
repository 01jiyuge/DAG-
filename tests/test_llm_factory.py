import pytest
import asyncio
from hpd_agent.llm import LLMFactory, LLMConfig, LLMProvider

@pytest.mark.asyncio
async def test_llm_config():
    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        api_key="test_key",
        model="gpt-4",
        temperature=0.7,
        max_tokens=4096
    )
    
    assert config.provider == LLMProvider.OPENAI
    assert config.api_key == "test_key"
    assert config.model == "gpt-4"

@pytest.mark.asyncio
async def test_llm_factory_cache():
    LLMFactory.clear_cache()
    
    config1 = LLMConfig(
        provider=LLMProvider.OPENAI,
        api_key="test_key",
        model="gpt-4"
    )
    
    config2 = LLMConfig(
        provider=LLMProvider.OPENAI,
        api_key="test_key",
        model="gpt-4"
    )
    
    try:
        llm1 = await LLMFactory.create(config1)
        llm2 = await LLMFactory.create(config2)
    except Exception:
        pytest.skip("LLM API not available")
    
    assert llm1 is llm2

@pytest.mark.asyncio
async def test_llm_factory_different_configs():
    LLMFactory.clear_cache()
    
    config1 = LLMConfig(
        provider=LLMProvider.OPENAI,
        api_key="test_key1",
        model="gpt-4"
    )
    
    config2 = LLMConfig(
        provider=LLMProvider.OPENAI,
        api_key="test_key2",
        model="gpt-4"
    )
    
    try:
        llm1 = await LLMFactory.create(config1)
        llm2 = await LLMFactory.create(config2)
    except Exception:
        pytest.skip("LLM API not available")
    
    assert llm1 is not llm2