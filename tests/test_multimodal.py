import pytest
import asyncio
from hpd_agent.multimodal import MultimodalProcessor, MultimodalInput, InputType

@pytest.mark.asyncio
async def test_process_text():
    processor = MultimodalProcessor()
    inputs = [MultimodalInput(type=InputType.TEXT, content="Hello, world!")]
    results = await processor.process(inputs)
    
    assert len(results) == 1
    assert results[0].source_type == InputType.TEXT
    assert results[0].content == "Hello, world!"
    assert results[0].confidence == 1.0

@pytest.mark.asyncio
async def test_process_image():
    processor = MultimodalProcessor()
    
    test_image_bytes = b"fake_image_data"
    inputs = [MultimodalInput(type=InputType.IMAGE, content=test_image_bytes)]
    results = await processor.process(inputs)
    
    assert len(results) == 1
    assert results[0].source_type == InputType.IMAGE
    assert results[0].confidence < 0.5

@pytest.mark.asyncio
async def test_combine_semantics():
    processor = MultimodalProcessor()
    
    from hpd_agent.multimodal import TextSemantic
    semantics = [
        TextSemantic(content="Text content", source_type=InputType.TEXT, confidence=1.0),
        TextSemantic(content="Image content", source_type=InputType.IMAGE, confidence=0.85)
    ]
    
    combined = processor.combine_semantics(semantics)
    
    assert "text" in combined.lower()
    assert "image" in combined.lower()

@pytest.mark.asyncio
async def test_process_and_combine():
    processor = MultimodalProcessor()
    inputs = [
        MultimodalInput(type=InputType.TEXT, content="Test query"),
        MultimodalInput(type=InputType.AUDIO, content=b"audio_data", metadata={"duration": 10})
    ]
    
    result = await processor.process_and_combine(inputs)
    
    assert "text" in result.lower()
    assert "audio" in result.lower()