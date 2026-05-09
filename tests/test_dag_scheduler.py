import pytest
import asyncio
from hpd_agent.dag import DAGScheduler, Task, TaskType, TaskResult

@pytest.mark.asyncio
async def test_kahn_topological_sort():
    tasks = [
        Task(id="task1", name="Task 1", type=TaskType.SIMPLE, difficulty=0.3),
        Task(id="task2", name="Task 2", type=TaskType.SIMPLE, difficulty=0.4, dependencies=["task1"]),
        Task(id="task3", name="Task 3", type=TaskType.SIMPLE, difficulty=0.5, dependencies=["task1"]),
        Task(id="task4", name="Task 4", type=TaskType.SIMPLE, difficulty=0.6, dependencies=["task2", "task3"])
    ]
    
    scheduler = DAGScheduler()
    levels = await scheduler.kahn_topological_sort(tasks)
    
    assert len(levels) == 3
    assert "task1" in levels[0]
    assert "task4" in levels[-1]

@pytest.mark.asyncio
async def test_schedule_tasks():
    async def mock_executor(task):
        await asyncio.sleep(0.1)
        return TaskResult(task_id=task.id, success=True, result={"content": f"Result for {task.name}"})
    
    tasks = [
        Task(id="t1", name="Task 1", type=TaskType.SIMPLE, difficulty=0.3),
        Task(id="t2", name="Task 2", type=TaskType.SIMPLE, difficulty=0.4, dependencies=["t1"])
    ]
    
    scheduler = DAGScheduler(max_workers=2)
    results = await scheduler.schedule(tasks, mock_executor)
    
    assert len(results) == 2
    assert all(r.success for r in results)

@pytest.mark.asyncio
async def test_schedule_with_cycle():
    tasks = [
        Task(id="t1", name="Task 1", type=TaskType.SIMPLE, dependencies=["t2"]),
        Task(id="t2", name="Task 2", type=TaskType.SIMPLE, dependencies=["t1"])
    ]
    
    async def mock_executor(task):
        return TaskResult(task_id=task.id, success=True)
    
    scheduler = DAGScheduler()
    results = await scheduler.schedule(tasks, mock_executor)
    
    assert len(results) == 1
    assert not results[0].success
    assert "cycle" in results[0].error.lower()