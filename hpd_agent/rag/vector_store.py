from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
from pydantic import BaseModel, Field
import asyncio
from hpd_agent.utils.logger import get_logger
from .document import Document, Chunk

class VectorStoreType(str, Enum):
    CHROMA = "chroma"
    FAISS = "faiss"
    PINECONE = "pinecone"

class VectorStoreConfig(BaseModel):
    type: VectorStoreType = VectorStoreType.CHROMA
    persist_directory: str = "./data/chroma"
    api_key: Optional[str] = None
    index_name: str = "hpd_agent_index"
    
    class Config:
        arbitrary_types_allowed = True

class VectorStore:
    def __init__(self, config: VectorStoreConfig = None):
        self.config = config or VectorStoreConfig()
        self.logger = get_logger(__name__)
        self._store = None
    
    async def initialize(self):
        if self._store is not None:
            return
        
        store_type = self.config.type
        
        if store_type == VectorStoreType.CHROMA:
            await self._init_chroma()
        elif store_type == VectorStoreType.FAISS:
            await self._init_faiss()
        elif store_type == VectorStoreType.PINECONE:
            await self._init_pinecone()
        
        self.logger.info(f"Initialized vector store: {store_type.value}")
    
    async def _init_chroma(self):
        from langchain_chroma import Chroma
        
        self._store = Chroma(
            persist_directory=self.config.persist_directory,
            collection_name=self.config.index_name
        )
    
    async def _init_faiss(self):
        from langchain_community.vectorstores import FAISS
        
        self._store = FAISS.from_texts(
            texts=[""],
            embedding=self._get_dummy_embedding()
        )
    
    async def _init_pinecone(self):
        from langchain_pinecone import Pinecone
        
        self._store = Pinecone.from_existing_index(
            index_name=self.config.index_name,
            embedding=self._get_dummy_embedding()
        )
    
    def _get_dummy_embedding(self):
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    async def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        if self._store is None:
            await self.initialize()
        
        return await asyncio.to_thread(
            self._store.add_texts,
            texts,
            metadatas=metadatas
        )
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return await self.add_texts(texts, metadatas)
    
    async def similarity_search(
        self, 
        query: str, 
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        if self._store is None:
            await self.initialize()
        
        results = await asyncio.to_thread(
            self._store.similarity_search_with_score,
            query,
            k=k,
            filter=filter
        )
        
        documents = []
        for doc, score in results:
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            documents.append((
                Document(
                    id=doc.id if hasattr(doc, 'id') else "",
                    content=doc.page_content if hasattr(doc, 'page_content') else str(doc),
                    metadata=metadata,
                    source=metadata.get('source')
                ),
                score
            ))
        
        return documents
    
    async def delete(self, ids: List[str]) -> None:
        if self._store is None:
            await self.initialize()
        
        await asyncio.to_thread(self._store.delete, ids=ids)
    
    async def persist(self):
        if hasattr(self._store, 'persist'):
            await asyncio.to_thread(self._store.persist)
            self.logger.info("Vector store persisted")
    
    async def get_stats(self) -> Dict[str, Any]:
        if hasattr(self._store, 'get_collection'):
            collection = await asyncio.to_thread(self._store.get_collection)
            return {
                'total_documents': collection.count(),
                'collection_name': collection.name
            }
        return {}