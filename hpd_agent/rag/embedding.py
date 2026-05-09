from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
import asyncio
from hpd_agent.utils.logger import get_logger

class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    DASHSCOPE = "dashscope"
    LOCAL = "local"

class EmbeddingConfig(BaseModel):
    provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    api_key: Optional[str] = None
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    base_url: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True

class EmbeddingManager:
    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()
        self.logger = get_logger(__name__)
        self._embedding = None
    
    async def initialize(self):
        if self._embedding is not None:
            return
        
        provider = self.config.provider
        
        if provider == EmbeddingProvider.OPENAI:
            await self._init_openai_embedding()
        elif provider == EmbeddingProvider.DASHSCOPE:
            await self._init_dashscope_embedding()
        elif provider == EmbeddingProvider.LOCAL:
            await self._init_local_embedding()
        
        self.logger.info(f"Initialized embedding provider: {provider.value}")
    
    async def _init_openai_embedding(self):
        from langchain_openai import OpenAIEmbeddings
        
        params = {
            "model": self.config.model,
            "dimensions": self.config.dimensions
        }
        
        if self.config.api_key:
            params["openai_api_key"] = self.config.api_key
        if self.config.base_url:
            params["openai_api_base"] = self.config.base_url
        
        self._embedding = OpenAIEmbeddings(**params)
    
    async def _init_dashscope_embedding(self):
        from langchain_community.embeddings import DashScopeEmbeddings
        
        self._embedding = DashScopeEmbeddings(
            dashscope_api_key=self.config.api_key,
            model=self.config.model
        )
    
    async def _init_local_embedding(self):
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        self._embedding = HuggingFaceEmbeddings(
            model_name=self.config.model or "all-MiniLM-L6-v2"
        )
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        if self._embedding is None:
            await self.initialize()
        
        if hasattr(self._embedding, 'embed_documents'):
            return await asyncio.to_thread(self._embedding.embed_documents, texts)
        else:
            return [await asyncio.to_thread(self._embedding.embed_query, text) for text in texts]
    
    async def embed_query(self, query: str) -> List[float]:
        if self._embedding is None:
            await self.initialize()
        
        if hasattr(self._embedding, 'embed_query'):
            return await asyncio.to_thread(self._embedding.embed_query, query)
        else:
            return (await self.embed([query]))[0]