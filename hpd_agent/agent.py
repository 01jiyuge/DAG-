import asyncio
from typing import List, Dict, Any, Optional, Union, Callable
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from .config import AgentConfig
from .dag import DAGScheduler, SchedulerConfig, Task, TaskType, TaskResult
from .routing import HierarchicalRouter, RouteDecision
from .expert import ExpertExecutor, ExpertConfig, GenerationResult
from .multimodal import MultimodalProcessor, MultimodalInput, InputType
from .cache import CacheManager, CacheConfig, CacheBackend
from .llm import LLMFactory, LLMConfig, LLMProvider
from .exceptions import (
    LLMNotInitializedError,
    TaskTimeoutError,
    TaskRetryExhaustedError,
    HPDAgentError,
    ConfigurationError
)
from .utils.logger import get_logger

class AgentState(BaseModel):
    query: str
    context: Dict[str, Any] = Field(default_factory=dict)
    route_decision: Optional[RouteDecision] = None
    task_results: List[TaskResult] = Field(default_factory=list)
    final_result: Optional[str] = None
    error: Optional[str] = None

class HPDLLMAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = get_logger(__name__, config.log_level)
        self.router = HierarchicalRouter()
        scheduler_config = SchedulerConfig(
            max_workers=config.max_workers,
            default_timeout=config.timeout,
            retry_attempts=config.max_retries
        )
        self.scheduler = DAGScheduler(scheduler_config)
        self.expert_executor = ExpertExecutor(ExpertConfig())
        self.multimodal_processor = MultimodalProcessor()
        
        cache_config = CacheConfig(
            backend=CacheBackend(config.cache_backend),
            ttl_seconds=config.cache_ttl_seconds,
            max_size=config.cache_max_size,
            redis_url=config.redis_url
        )
        self.cache = CacheManager(cache_config)
        self.llm = None
        self.llm_config = None
    
    async def initialize_llm(self) -> None:
        if self.llm is not None:
            return
        
        config = self.config
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
            self.logger.warning(f"No API key provided for {config.llm_provider}")
            return
        
        self.llm_config = LLMConfig(
            provider=LLMProvider(config.llm_provider),
            api_key=api_key,
            model=config.default_model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
            base_url=config.llm_base_url
        )
        
        self.llm = await LLMFactory.create(self.llm_config)
        self.logger.info(f"LLM initialized: {config.llm_provider} - {config.default_model}")
    
    async def _call_llm(self, prompt: str, model: str = "gpt-3.5-turbo") -> str:
        if not self.llm:
            raise LLMNotInitializedError("LLM not initialized")
        
        if not isinstance(prompt, str) or len(prompt.strip()) == 0:
            raise ValueError("Prompt must be a non-empty string")
        
        cache_key = f"{model}:{prompt[:50]}"
        cached_response = await self.cache.get("llm", cache_key)
        if cached_response is not None:
            self.logger.debug(f"LLM cache hit for prompt starting with: {prompt[:30]}...")
            return cached_response
        
        messages = [
            SystemMessage(content="你是一个专业的AI助手，擅长分析和解决各种问题。"),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = await self.llm.agenerate([messages])
            result = response.generations[0][0].text
            await self.cache.set("llm", cache_key, result, ttl_seconds=3600)
            return result
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            raise
    
    async def _create_tasks_from_query(self, query: str, decision: RouteDecision) -> List[Task]:
        tasks = []
        
        if decision.level.value == "level_1":
            tasks.append(Task(
                name="simple_query_processing",
                type=TaskType.SIMPLE,
                difficulty=0.3,
                payload={"query": query}
            ))
        
        elif decision.level.value == "level_2":
            tasks.extend([
                Task(
                    id="task_analyze",
                    name="analyze_query",
                    type=TaskType.COMPLEX,
                    difficulty=0.6,
                    payload={"query": query}
                ),
                Task(
                    id="task_research",
                    name="research_context",
                    type=TaskType.COMPLEX,
                    difficulty=0.5,
                    dependencies=["task_analyze"],
                    payload={"query": query}
                ),
                Task(
                    id="task_synthesize",
                    name="synthesize_result",
                    type=TaskType.COMPLEX,
                    difficulty=0.7,
                    dependencies=["task_research"],
                    payload={"query": query}
                )
            ])
        
        elif decision.level.value == "level_3":
            tasks.extend([
                Task(
                    id="task_decompose",
                    name="decompose_problem",
                    type=TaskType.EXPERT,
                    difficulty=0.8,
                    payload={"query": query}
                ),
                Task(
                    id="task_subtask1",
                    name="subtask_analysis",
                    type=TaskType.EXPERT,
                    difficulty=0.75,
                    dependencies=["task_decompose"],
                    payload={"query": query}
                ),
                Task(
                    id="task_subtask2",
                    name="subtask_research",
                    type=TaskType.EXPERT,
                    difficulty=0.85,
                    dependencies=["task_decompose"],
                    payload={"query": query}
                ),
                Task(
                    id="task_subtask3",
                    name="subtask_validation",
                    type=TaskType.EXPERT,
                    difficulty=0.7,
                    dependencies=["task_subtask1", "task_subtask2"],
                    payload={"query": query}
                ),
                Task(
                    id="task_finalize",
                    name="finalize_result",
                    type=TaskType.EXPERT,
                    difficulty=0.9,
                    dependencies=["task_subtask3"],
                    payload={"query": query}
                )
            ])
        
        self.logger.info(f"Created {len(tasks)} tasks for query")
        return tasks
    
    async def _execute_task(self, task: Task) -> TaskResult:
        try:
            self.logger.info(f"Executing task: {task.name}")
            
            cache_key = f"{task.name}:{task.id}"
            cached_result = await self.cache.get("task", cache_key)
            if cached_result is not None:
                self.logger.debug(f"Task cache hit for {task.name}")
                return TaskResult(**cached_result)
            
            if task.type == TaskType.EXPERT:
                result = await self.expert_executor.execute(
                    task.payload.get("query", ""),
                    lambda p: self._call_llm(p)
                )
                task_result = TaskResult(
                    task_id=task.id,
                    success=True,
                    result={"content": result.content, "score": result.score}
                )
                
            else:
                prompt = task.payload.get("query", "")
                response = await self._call_llm(prompt)
                task_result = TaskResult(
                    task_id=task.id,
                    success=True,
                    result={"content": response}
                )
            
            await self.cache.set("task", cache_key, task_result.dict(), ttl_seconds=1800)
            return task_result
                
        except Exception as e:
            self.logger.error(f"Task {task.name} failed: {e}")
            return TaskResult(
                task_id=task.id,
                success=False,
                error=str(e)
            )
    
    async def _build_langgraph(self):
        async def route_node(state: AgentState):
            decision = self.router.route(state.query, state.context)
            state.route_decision = decision
            self.logger.info(f"Routing decision: {decision.node_name}")
            return state
        
        async def task_scheduler_node(state: AgentState):
            tasks = await self._create_tasks_from_query(state.query, state.route_decision)
            results = await self.scheduler.schedule(tasks, self._execute_task)
            state.task_results = results
            
            failed = [r for r in results if not r.success]
            if failed:
                state.error = f"{len(failed)} tasks failed"
                return "error"
            return "summarize"
        
        async def summarize_node(state: AgentState):
            results_text = "\n\n".join(
                f"Task {r.task_id}: {r.result.get('content', '')}"
                for r in state.task_results if r.success
            )
            
            summary_prompt = f"""
请总结以下任务执行结果：

{results_text}

原始查询：{state.query}

请提供一个清晰、简洁的总结。
"""
            summary = await self._call_llm(summary_prompt)
            state.final_result = summary
            return state
        
        workflow = StateGraph(AgentState)
        workflow.add_node("route", route_node)
        workflow.add_node("schedule", task_scheduler_node)
        workflow.add_node("summarize", summarize_node)
        
        workflow.set_entry_point("route")
        workflow.add_edge("route", "schedule")
        workflow.add_edge("schedule", "summarize")
        workflow.add_edge("summarize", END)
        
        return workflow.compile()
    
    async def run(self, query: Union[str, List[MultimodalInput]], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            self.logger.info(f"Starting agent with query: {query[:100]}..." if isinstance(query, str) else "Starting agent with multimodal input")
            
            await self.initialize_llm()
            
            if isinstance(query, list) and len(query) > 0:
                processed_text = await self.multimodal_processor.process_and_combine(query)
                text_query = processed_text
            else:
                text_query = str(query)
            
            decision = self.router.route(text_query, context)
            self.logger.info(f"Route decision: {decision.node_name} (confidence: {decision.confidence:.2f})")
            
            tasks = await self._create_tasks_from_query(text_query, decision)
            results = await self.scheduler.schedule(tasks, self._execute_task)
            
            success_count = sum(1 for r in results if r.success)
            self.logger.info(f"Task execution completed: {success_count}/{len(results)} succeeded")
            
            if success_count == 0:
                return {
                    "success": False,
                    "error": "All tasks failed",
                    "details": [r.error for r in results if not r.success]
                }
            
            results_text = "\n\n".join(
                f"【{i+1}】{r.result.get('content', '')}"
                for i, r in enumerate(results) if r.success
            )
            
            final_prompt = f"""
原始查询：{text_query}

任务执行结果：
{results_text}

请基于以上信息，提供一个完整、详细的最终回答。
"""
            
            if decision.level.value == "level_3":
                expert_result = await self.expert_executor.execute(
                    final_prompt,
                    lambda p: self._call_llm(p)
                )
                final_answer = expert_result.content
                score = expert_result.score
            else:
                final_answer = await self._call_llm(final_prompt)
                score = 0.0
            
            return {
                "success": True,
                "answer": final_answer,
                "score": score,
                "route_level": decision.level.value,
                "node": decision.node_name,
                "task_count": len(results),
                "success_count": success_count
            }
        
        except Exception as e:
            self.logger.error(f"Agent execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def run_with_langgraph(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            app = await self._build_langgraph()
            
            initial_state = AgentState(
                query=query,
                context=context or {}
            )
            
            final_state = await app.ainvoke(initial_state)
            
            if final_state.error:
                return {
                    "success": False,
                    "error": final_state.error
                }
            
            return {
                "success": True,
                "answer": final_state.final_result,
                "route_level": final_state.route_decision.level.value if final_state.route_decision else None,
                "task_count": len(final_state.task_results)
            }
        
        except Exception as e:
            self.logger.error(f"LangGraph execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }