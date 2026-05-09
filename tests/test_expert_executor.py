import pytest
import asyncio
from hpd_agent.expert import ExpertExecutor, ExpertConfig, GenerationResult

@pytest.mark.asyncio
async def test_rewrite_prompt():
    executor = ExpertExecutor(ExpertConfig())
    original = "Explain AI"
    rewritten = executor.rewrite_prompt(original)
    
    assert len(rewritten) > len(original)
    assert "详细" in rewritten or "专业" in rewritten

@pytest.mark.asyncio
async def test_evaluate_weights():
    executor = ExpertExecutor(ExpertConfig())
    prompt = "Analyze and evaluate this complex problem with precise calculations"
    weights = executor.evaluate_weights(prompt)
    
    assert "analysis" in weights
    assert "precision" in weights
    assert sum(weights.values()) == pytest.approx(1.0)

@pytest.mark.asyncio
async def test_multi_dimensional_scoring():
    executor = ExpertExecutor(ExpertConfig())
    
    results = [
        GenerationResult(content="首先，我们需要分析问题。其次，进行深入研究。最后，总结结论。", score=0.0),
        GenerationResult(content="简单回答", score=0.0)
    ]
    
    scored = executor.multi_dimensional_scoring(results)
    
    assert len(scored) == 2
    assert scored[0].score > scored[1].score
    assert scored[0].score > 0.3

@pytest.mark.asyncio
async def test_generate_multi_path():
    executor = ExpertExecutor(ExpertConfig(max_parallel_generations=2))
    
    async def mock_llm(prompt):
        await asyncio.sleep(0.1)
        return f"Response to: {prompt[:20]}..."
    
    results = await executor.generate_multi_path("Test prompt", mock_llm, num_paths=2)
    
    assert len(results) == 2
    assert all(isinstance(r, GenerationResult) for r in results)