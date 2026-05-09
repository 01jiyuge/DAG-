from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
from pydantic import BaseModel, Field
from hpd_agent.utils.logger import get_logger
from .document import Document
from .embedding import EmbeddingManager, EmbeddingConfig
from .vector_store import VectorStore, VectorStoreConfig

class RAGMode(str, Enum):
    RETRIEVE_ONLY = "retrieve_only"
    GENERATE_ONLY = "generate_only"
    HYBRID = "hybrid"

class RAGConfig(BaseModel):
    mode: RAGMode = RAGMode.HYBRID
    top_k: int = 4
    max_context_length: int = 4000
    embedding_config: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store_config: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    
    class Config:
        arbitrary_types_allowed = True

class RAGResult(BaseModel):
    documents: List[Document]
    scores: List[float]
    context: str
    context_length: int
    retrieval_time: float

class RAGRetriever:
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.logger = get_logger(__name__)
        self._embedding_manager = EmbeddingManager(self.config.embedding_config)
        self._vector_store = VectorStore(self.config.vector_store_config)
    
    async def initialize(self):
        await self._embedding_manager.initialize()
        await self._vector_store.initialize()
        self.logger.info("RAG retriever initialized")
    
    async def add_document(self, document: Document) -> str:
        ids = await self._vector_store.add_documents([document])
        await self._vector_store.persist()
        self.logger.debug(f"Added document: {document.id}")
        return ids[0] if ids else ""
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        ids = await self._vector_store.add_documents(documents)
        await self._vector_store.persist()
        self.logger.debug(f"Added {len(documents)} documents")
        return ids
    
    async def retrieve(self, query: str, k: Optional[int] = None) -> RAGResult:
        import time
        
        start_time = time.time()
        
        if k is None:
            k = self.config.top_k
        
        results = await self._vector_store.similarity_search(query, k=k)
        
        documents = []
        scores = []
        
        for doc, score in results:
            documents.append(doc)
            scores.append(score)
        
        context = self._build_context(documents)
        context_length = len(context)
        
        retrieval_time = time.time() - start_time
        
        self.logger.debug(f"Retrieved {len(documents)} documents in {retrieval_time:.2f}s")
        
        return RAGResult(
            documents=documents,
            scores=scores,
            context=context,
            context_length=context_length,
            retrieval_time=retrieval_time
        )
    
    def _build_context(self, documents: List[Document]) -> str:
        context_parts = []
        total_length = 0
        
        for doc in documents:
            content = doc.content
            source = doc.source or "unknown"
            
            part = f"【来源: {source}】\n{content}\n\n"
            part_length = len(part)
            
            if total_length + part_length <= self.config.max_context_length:
                context_parts.append(part)
                total_length += part_length
            else:
                remaining = self.config.max_context_length - total_length
                if remaining > 0:
                    context_parts.append(part[:remaining])
                break
        
        return "".join(context_parts).strip()
    
    async def generate_enhanced_prompt(self, query: str) -> str:
        if self.config.mode == RAGMode.GENERATE_ONLY:
            return f"请回答以下问题：\n{query}"
        
        rag_result = await self.retrieve(query)
        
        if self.config.mode == RAGMode.RETRIEVE_ONLY:
            return rag_result.context
        
        context = rag_result.context
        
        prompt = f"""
根据以下参考文档回答问题：

参考文档：
{context}

问题：
{query}

请根据参考文档内容回答问题。如果文档中没有相关信息，请说明"文档中未找到相关信息"。
"""
        
        return prompt.strip()
    
    async def get_stats(self) -> Dict[str, Any]:
        store_stats = await self._vector_store.get_stats()
        return {
            'vector_store': store_stats,
            'retrieval_mode': self.config.mode.value,
            'top_k': self.config.top_k,
            'max_context_length': self.config.max_context_length
        }