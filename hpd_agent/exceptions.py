from typing import Optional, Dict, Any

class HPDAgentError(Exception):
    base_message: str = "HPD Agent error occurred"
    
    def __init__(self, message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message or self.base_message)
        self.details = details or {}

class ConfigurationError(HPDAgentError):
    base_message = "Configuration error"

class LLMError(HPDAgentError):
    base_message = "LLM error"

class CacheError(HPDAgentError):
    base_message = "Cache error"

class SchedulerError(HPDAgentError):
    base_message = "Scheduler error"

class TaskError(HPDAgentError):
    base_message = "Task error"

class RouteError(HPDAgentError):
    base_message = "Routing error"

class MultimodalError(HPDAgentError):
    base_message = "Multimodal processing error"

class DagCycleError(SchedulerError):
    base_message = "DAG contains a cycle"

class TaskTimeoutError(TaskError):
    base_message = "Task timed out"

class TaskRetryExhaustedError(TaskError):
    base_message = "All retry attempts exhausted"

class LLMNotInitializedError(LLMError):
    base_message = "LLM not initialized"

class APIKeyMissingError(ConfigurationError):
    base_message = "API key missing"

class InvalidInputError(HPDAgentError):
    base_message = "Invalid input"

class ServiceUnavailableError(HPDAgentError):
    base_message = "Service unavailable"