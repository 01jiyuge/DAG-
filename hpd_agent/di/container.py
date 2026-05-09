from typing import Any, Dict, Optional, Type, Callable, get_type_hints
from functools import wraps
import asyncio
from hpd_agent.utils.logger import get_logger

class DIContainer:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DIContainer, cls).__new__(cls)
            cls._instance._services: Dict[str, Any] = {}
            cls._instance._factories: Dict[str, Callable] = {}
            cls._instance._singletons: Dict[str, Any] = {}
            cls._instance.logger = get_logger(__name__)
        return cls._instance
    
    def register(self, name: str, service: Any) -> None:
        self._services[name] = service
        self.logger.debug(f"Registered service: {name}")
    
    def register_factory(self, name: str, factory: Callable) -> None:
        self._factories[name] = factory
        self.logger.debug(f"Registered factory: {name}")
    
    def register_singleton(self, name: str, factory: Callable) -> None:
        self._singletons[name] = {"factory": factory, "instance": None}
        self.logger.debug(f"Registered singleton: {name}")
    
    def resolve(self, name: str) -> Any:
        if name in self._services:
            return self._services[name]
        
        if name in self._singletons:
            singleton = self._singletons[name]
            if singleton["instance"] is None:
                singleton["instance"] = singleton["factory"]()
            return singleton["instance"]
        
        if name in self._factories:
            return self._factories[name]()
        
        raise ValueError(f"Service not found: {name}")
    
    async def resolve_async(self, name: str) -> Any:
        if name in self._services:
            return self._services[name]
        
        if name in self._singletons:
            singleton = self._singletons[name]
            if singleton["instance"] is None:
                factory = singleton["factory"]
                singleton["instance"] = await factory() if asyncio.iscoroutinefunction(factory) else factory()
            return singleton["instance"]
        
        if name in self._factories:
            factory = self._factories[name]
            return await factory() if asyncio.iscoroutinefunction(factory) else factory()
        
        raise ValueError(f"Service not found: {name}")
    
    def get(self, name: str, default: Any = None) -> Any:
        try:
            return self.resolve(name)
        except ValueError:
            return default
    
    def clear(self):
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()
        self.logger.debug("Container cleared")

def inject(func):
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        hints = get_type_hints(func)
        container = DIContainer()
        
        for param_name, param_type in hints.items():
            if param_name not in kwargs:
                try:
                    kwargs[param_name] = await container.resolve_async(param_name)
                except ValueError:
                    pass
        
        result = func(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        hints = get_type_hints(func)
        container = DIContainer()
        
        for param_name, param_type in hints.items():
            if param_name not in kwargs:
                try:
                    kwargs[param_name] = container.resolve(param_name)
                except ValueError:
                    pass
        
        return func(*args, **kwargs)
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper