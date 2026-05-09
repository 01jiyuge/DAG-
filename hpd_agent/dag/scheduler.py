import asyncio
from typing import List, Dict, Optional, Callable, Awaitable, Any
from collections import defaultdict
import uuid
from datetime import datetime, timedelta
from .task import Task, TaskStatus, TaskType, TaskResult
from hpd_agent.exceptions import DagCycleError, TaskTimeoutError, TaskRetryExhaustedError
from hpd_agent.utils.logger import get_logger

class SchedulerConfig:
    def __init__(
        self,
        max_workers: int = 10,
        default_timeout: int = 300,
        retry_attempts: int = 3,
        retry_delay: float = 1.0
    ):
        self.max_workers = max_workers
        self.default_timeout = default_timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

class DAGScheduler:
    def __init__(self, config: Optional[SchedulerConfig] = None, max_workers: Optional[int] = None):
        if max_workers is not None and config is None:
            self.config = SchedulerConfig(max_workers=max_workers)
        else:
            self.config = config or SchedulerConfig()
        self.logger = get_logger(__name__)
        self._running_tasks: Dict[str, asyncio.Task] = {}
    
    async def execute_task_with_timeout(
        self,
        task: Task,
        executor: Callable[[Task], Awaitable[TaskResult]],
        timeout: Optional[int] = None
    ) -> TaskResult:
        effective_timeout = timeout or self.config.default_timeout
        
        async def wrapped_executor():
            for attempt in range(self.config.retry_attempts):
                try:
                    self.logger.info(f"Executing task: {task.name} (id: {task.id}, attempt: {attempt + 1})")
                    task.status = TaskStatus.RUNNING
                    task.start_time = datetime.now()
                    
                    result = await executor(task)
                    
                    task.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
                    task.result = result.result
                    task.error = result.error
                    task.end_time = datetime.now()
                    
                    if result.success:
                        return result
                    else:
                        self.logger.warning(f"Task {task.name} failed, attempt {attempt + 1}")
                        
                except Exception as e:
                    self.logger.error(f"Task {task.name} failed with exception: {e}, attempt {attempt + 1}")
                    
                    if attempt < self.config.retry_attempts - 1:
                        await asyncio.sleep(self.config.retry_delay * (attempt + 1))
            
            task.status = TaskStatus.FAILED
            task.error = "All retry attempts exhausted"
            task.end_time = datetime.now()
            return TaskResult(task_id=task.id, success=False, error=task.error)
        
        try:
            return await asyncio.wait_for(wrapped_executor(), timeout=effective_timeout)
        except asyncio.TimeoutError:
            self.logger.error(f"Task {task.name} timed out after {effective_timeout} seconds")
            task.status = TaskStatus.FAILED
            task.error = f"Timeout after {effective_timeout} seconds"
            task.end_time = datetime.now()
            return TaskResult(task_id=task.id, success=False, error=task.error)
    
    async def kahn_topological_sort(self, tasks: List[Task]) -> List[List[str]]:
        in_degree: Dict[str, int] = {task.id: len(task.dependencies) for task in tasks}
        adjacency: Dict[str, List[str]] = defaultdict(list)
        task_map: Dict[str, Task] = {task.id: task for task in tasks}
        
        for task in tasks:
            for dep_id in task.dependencies:
                if dep_id in task_map:
                    adjacency[dep_id].append(task.id)
        
        queue = [task.id for task in tasks if in_degree[task.id] == 0]
        result = []
        
        while queue:
            current_level = []
            next_queue = []
            
            for task_id in queue:
                current_level.append(task_id)
                for neighbor in adjacency[task_id]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_queue.append(neighbor)
            
            if current_level:
                result.append(current_level)
            queue = next_queue
        
        remaining = sum(1 for v in in_degree.values() if v > 0)
        if remaining > 0:
            raise DagCycleError(
                f"DAG contains a cycle, {remaining} tasks have unresolved dependencies",
                details={"remaining_tasks": remaining}
            )
        
        return result
    
    async def schedule(
        self,
        tasks: List[Task],
        executor: Callable[[Task], Awaitable[TaskResult]],
        timeout: Optional[int] = None
    ) -> List[TaskResult]:
        if not tasks:
            return []
        
        self.logger.info(f"Starting DAG scheduling with {len(tasks)} tasks")
        start_time = datetime.now()
        
        try:
            levels = await self.kahn_topological_sort(tasks)
            self.logger.info(f"Topological sort completed, {len(levels)} levels")
            
            all_results = []
            semaphore = asyncio.Semaphore(self.config.max_workers)
            
            async def execute_with_limit(task: Task) -> TaskResult:
                async with semaphore:
                    return await self.execute_task_with_timeout(task, executor, timeout)
            
            for level_idx, level in enumerate(levels):
                self.logger.info(f"Processing level {level_idx + 1}/{len(levels)} with {len(level)} tasks")
                
                level_tasks = [task for task in tasks if task.id in level]
                level_tasks.sort(key=lambda t: (-t.difficulty, t.priority if hasattr(t, 'priority') else 0))
                
                coroutines = [execute_with_limit(task) for task in level_tasks]
                level_results = await asyncio.gather(*coroutines)
                all_results.extend(level_results)
                
                failed_tasks = [r for r in level_results if not r.success]
                if failed_tasks:
                    self.logger.warning(f"Level {level_idx + 1} completed with {len(failed_tasks)} failures")
            
            elapsed = datetime.now() - start_time
            self.logger.info(f"DAG scheduling completed in {elapsed.total_seconds():.2f} seconds")
            return all_results
        
        except (DagCycleError, ValueError) as e:
            self.logger.error(f"DAG scheduling failed: {e}")
            return [TaskResult(task_id="", success=False, error=str(e))]
    
    def build_dag_from_tasks(self, task_definitions: List[Dict[str, Any]]) -> List[Task]:
        tasks = []
        for definition in task_definitions:
            task = Task(
                id=definition.get("id", str(uuid.uuid4())),
                name=definition["name"],
                type=TaskType(definition.get("type", "simple")),
                difficulty=definition.get("difficulty", 0.5),
                dependencies=definition.get("dependencies", []),
                payload=definition.get("payload", {}),
                priority=definition.get("priority", 0)
            )
            tasks.append(task)
        return tasks
    
    def cancel_running_tasks(self):
        for task_id, async_task in self._running_tasks.items():
            if not async_task.done():
                async_task.cancel()
                self.logger.info(f"Cancelled running task: {task_id}")
        self._running_tasks.clear()