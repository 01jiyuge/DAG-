import pytest
from hpd_agent.rag import RAGRetriever, RAGConfig, Document
from hpd_agent.rag.embedding import EmbeddingManager, EmbeddingConfig, EmbeddingProvider
from hpd_agent.rag.vector_store import VectorStore, VectorStoreConfig, VectorStoreType

class TestEmbeddingManager:
    @pytest.mark.asyncio
    async def test_embedding_config_default(self):
        config = EmbeddingConfig()
        assert config.provider == EmbeddingProvider.OPENAI
        assert config.model == "text-embedding-3-small"
        assert config.dimensions == 1536
    
    @pytest.mark.asyncio
    async def test_embedding_initialization(self):
        config = EmbeddingConfig(provider=EmbeddingProvider.LOCAL)
        manager = EmbeddingManager(config)
        await manager.initialize()
        assert manager._embedding is not None
    
    @pytest.mark.asyncio
    async def test_embedding_embed_query(self):
        config = EmbeddingConfig(provider=EmbeddingProvider.LOCAL)
        manager = EmbeddingManager(config)
        await manager.initialize()
        embedding = await manager.embed_query("test query")
        assert isinstance(embedding, list)
        assert len(embedding) > 0

class TestVectorStore:
    @pytest.mark.asyncio
    async def test_vector_store_config(self):
        config = VectorStoreConfig()
        assert config.type == VectorStoreType.CHROMA
        assert config.persist_directory == "./data/chroma"
    
    @pytest.mark.asyncio
    async def test_vector_store_initialization(self):
        config = VectorStoreConfig(type=VectorStoreType.CHROMA)
        store = VectorStore(config)
        await store.initialize()
        assert store._store is not None
    
    @pytest.mark.asyncio
    async def test_vector_store_add_and_search(self):
        config = VectorStoreConfig(type=VectorStoreType.CHROMA, persist_directory="./data/test_chroma")
        store = VectorStore(config)
        await store.initialize()
        
        ids = await store.add_texts(["This is a test document", "Another test document"])
        assert len(ids) == 2
        
        results = await store.similarity_search("test")
        assert len(results) >= 1
        
        await store.persist()

class TestRAGRetriever:
    @pytest.mark.asyncio
    async def test_rag_config_default(self):
        config = RAGConfig()
        assert config.mode.value == "hybrid"
        assert config.top_k == 4
        assert config.max_context_length == 4000
    
    @pytest.mark.asyncio
    async def test_rag_retriever_initialization(self):
        config = RAGConfig(
            embedding_config=EmbeddingConfig(provider=EmbeddingProvider.LOCAL),
            vector_store_config=VectorStoreConfig(persist_directory="./data/test_rag")
        )
        retriever = RAGRetriever(config)
        await retriever.initialize()
        assert retriever._embedding_manager is not None
        assert retriever._vector_store is not None
    
    @pytest.mark.asyncio
    async def test_rag_add_document(self):
        config = RAGConfig(
            embedding_config=EmbeddingConfig(provider=EmbeddingProvider.LOCAL),
            vector_store_config=VectorStoreConfig(persist_directory="./data/test_rag_add")
        )
        retriever = RAGRetriever(config)
        await retriever.initialize()
        
        doc = Document(
            id="test_doc_1",
            content="Artificial Intelligence is the simulation of human intelligence processes by machines.",
            metadata={"category": "technology"},
            source="test_source"
        )
        
        doc_id = await retriever.add_document(doc)
        assert doc_id != ""
    
    @pytest.mark.asyncio
    async def test_rag_retrieve(self):
        config = RAGConfig(
            embedding_config=EmbeddingConfig(provider=EmbeddingProvider.LOCAL),
            vector_store_config=VectorStoreConfig(persist_directory="./data/test_rag_retrieve")
        )
        retriever = RAGRetriever(config)
        await retriever.initialize()
        
        doc = Document(
            id="retrieve_test",
            content="Machine learning is a subset of artificial intelligence.",
            metadata={"category": "AI"},
            source="test"
        )
        await retriever.add_document(doc)
        
        result = await retriever.retrieve("What is machine learning?")
        assert len(result.documents) > 0
        assert result.context_length > 0
    
    @pytest.mark.asyncio
    async def test_rag_generate_enhanced_prompt(self):
        config = RAGConfig(
            mode="hybrid",
            embedding_config=EmbeddingConfig(provider=EmbeddingProvider.LOCAL),
            vector_store_config=VectorStoreConfig(persist_directory="./data/test_rag_prompt")
        )
        retriever = RAGRetriever(config)
        await retriever.initialize()
        
        doc = Document(
            id="prompt_test",
            content="LangChain is a framework for developing applications powered by language models.",
            metadata={},
            source="langchain_docs"
        )
        await retriever.add_document(doc)
        
        prompt = await retriever.generate_enhanced_prompt("What is LangChain?")
        assert "参考文档" in prompt
        assert "LangChain" in prompt