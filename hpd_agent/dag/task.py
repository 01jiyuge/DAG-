from pydantic import BaseModel, Field
from typing import Optional, Any, Dict, List, Callable, Awaitable
from enum import Enum
from datetime import datetime
import uuid

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskType(str, Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"
    EXPERT = "expert"

class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: TaskType = TaskType.SIMPLE
    difficulty: float = Field(default=0.5, ge=0.0, le=1.0)
    priority: int = Field(default=0, ge=0, le=10)
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = Field(default_factory=list)
    payload: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    class Config:
        arbitrary_types_allowed = True

class TaskResult(BaseModel):
    task_id: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    
    def dict(self, **kwargs):
        return {
            "task_id": self.task_id,
            "success": self.success,
            "result": self.result,
            "error": self.error
        }