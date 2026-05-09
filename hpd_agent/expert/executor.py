import asyncio
from typing import List, Dict, Any, Optional, Callable
from pydantic import BaseModel, Field
from enum import Enum
from hpd_agent.utils.logger import get_logger

class ExpertMode(str, Enum):
    NORMAL = "normal"
    EXPERT = "expert"
    AUTO = "auto"

class GenerationResult(BaseModel):
    content: str
    score: float = 0.0
    latency: float = 0.0
    token_count: int = 0

class ExpertConfig(BaseModel):
    enabled: bool = True
    max_parallel_generations: int = 3
    temperature: float = 0.3
    max_tokens: int = 8192
    quality_threshold: float = 0.7

class ExpertExecutor:
    def __init__(self, config: ExpertConfig = None):
        self.config = config or ExpertConfig()
        self.logger = get_logger(__name__)
    
    def rewrite_prompt(self, original_prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        enhancements = [
            "提供详细、结构化的分析",
            "考虑多种可能性和边缘情况",
            "使用专业术语和精确表达",
            "输出格式清晰、易于理解",
            "包含必要的背景和解释"
        ]
        
        enhanced_prompt = f"""
你是一位专业领域专家，请基于以下要求完成任务：

要求：
{chr(10).join(f"- {e}" for e in enhancements)}

上下文信息：
{context.get("history", "") if context else ""}

原始任务：
{original_prompt}

请提供专业、全面的回答。
"""
        return enhanced_prompt.strip()
    
    def evaluate_weights(self, prompt: str) -> Dict[str, float]:
        keywords = {
            "analysis": ["analyze", "analysis", "evaluate", "assess"],
            "creativity": ["create", "design", "invent", "develop"],
            "precision": ["precise", "accurate", "exact", "correct"],
            "depth": ["deep", "detailed", "comprehensive", "thorough"]
        }
        
        weights = {"analysis": 0.25, "creativity": 0.25, "precision": 0.25, "depth": 0.25}
        
        for category, words in keywords.items():
            count = sum(1 for word in words if word.lower() in prompt.lower())
            weights[category] = min(0.5, 0.25 + count * 0.1)
        
        total = sum(weights.values())
        for key in weights:
            weights[key] /= total
        
        return weights
    
    async def generate_multi_path(self, prompt: str, llm_callable: Callable[[str], Any], num_paths: int = 3) -> List[GenerationResult]:
        tasks = []
        for i in range(num_paths):
            modified_prompt = f"{prompt}\n\n生成方案 {i + 1}/{num_paths}"
            tasks.append(self._generate_with_retry(modified_prompt, llm_callable))
        
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]
    
    async def _generate_with_retry(self, prompt: str, llm_callable: Callable[[str], Any], max_retries: int = 3) -> Optional[GenerationResult]:
        for attempt in range(max_retries):
            try:
                import time
                start_time = time.time()
                response = await llm_callable(prompt)
                latency = time.time() - start_time
                
                return GenerationResult(
                    content=response,
                    score=0.0,
                    latency=latency,
                    token_count=len(response) * 4
                )
            except Exception as e:
                self.logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
        
        return None
    
    def multi_dimensional_scoring(self, results: List[GenerationResult]) -> List[GenerationResult]:
        scored_results = []
        
        for result in results:
            scores = {}
            
            scores["relevance"] = self._score_relevance(result.content)
            scores["completeness"] = self._score_completeness(result.content)
            scores["coherence"] = self._score_coherence(result.content)
            scores["depth"] = self._score_depth(result.content)
            
            weights = {"relevance": 0.3, "completeness": 0.25, "coherence": 0.25, "depth": 0.2}
            weighted_score = sum(scores[key] * weights[key] for key in scores)
            
            result.score = weighted_score
            scored_results.append(result)
        
        scored_results.sort(key=lambda x: x.score, reverse=True)
        return scored_results
    
    def _score_relevance(self, content: str) -> float:
        if len(content) < 50:
            return 0.3
        return min(len(content) / 500, 1.0)
    
    def _score_completeness(self, content: str) -> float:
        indicators = ["首先", "其次", "最后", "总结", "综上所述", "因此"]
        count = sum(1 for indicator in indicators if indicator in content)
        return min(count / 3, 1.0)
    
    def _score_coherence(self, content: str) -> float:
        sentences = content.count("。") + content.count("？") + content.count("！")
        if sentences < 2:
            return 0.5
        avg_length = len(content) / max(sentences, 1)
        return min(max(avg_length / 50, 0.3), 1.0)
    
    def _score_depth(self, content: str) -> float:
        depth_words = ["分析", "论证", "推导", "证明", "解释", "说明", "对比", "比较"]
        count = sum(1 for word in depth_words if word in content)
        return min(count / 4, 1.0)
    
    async def execute(self, prompt: str, llm_callable: Callable[[str], Any], context: Optional[Dict[str, Any]] = None) -> GenerationResult:
        if not self.config.enabled:
            result = await self._generate_with_retry(prompt, llm_callable)
            return result
        
        self.logger.info("Entering expert mode")
        enhanced_prompt = self.rewrite_prompt(prompt, context)
        weights = self.evaluate_weights(enhanced_prompt)
        self.logger.info(f"Evaluated weights: {weights}")
        
        results = await self.generate_multi_path(enhanced_prompt, llm_callable, self.config.max_parallel_generations)
        
        if not results:
            self.logger.error("All generation attempts failed")
            return GenerationResult(content="", score=0.0)
        
        scored_results = self.multi_dimensional_scoring(results)
        best_result = scored_results[0]
        
        if best_result.score >= self.config.quality_threshold:
            self.logger.info(f"Expert execution successful with score: {best_result.score:.2f}")
        else:
            self.logger.warning(f"Expert execution completed but score below threshold: {best_result.score:.2f}")
        
        return best_result