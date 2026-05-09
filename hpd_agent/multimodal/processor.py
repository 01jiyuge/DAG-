from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum
import base64
from PIL import Image
import io
import asyncio
from hpd_agent.utils.logger import get_logger

class InputType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"

class MultimodalInput(BaseModel):
    type: InputType
    content: Union[str, bytes]
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class TextSemantic(BaseModel):
    content: str
    source_type: InputType
    confidence: float = 1.0

class MultimodalProcessor:
    def __init__(self):
        self.logger = get_logger(__name__)
    
    async def process(self, inputs: List[MultimodalInput]) -> List[TextSemantic]:
        tasks = [self._process_single(input) for input in inputs]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]
    
    async def _process_single(self, input_data: MultimodalInput) -> Optional[TextSemantic]:
        try:
            if input_data.type == InputType.TEXT:
                return TextSemantic(
                    content=str(input_data.content),
                    source_type=InputType.TEXT,
                    confidence=1.0
                )
            
            elif input_data.type == InputType.IMAGE:
                return await self._process_image(input_data)
            
            elif input_data.type == InputType.AUDIO:
                return await self._process_audio(input_data)
            
            elif input_data.type == InputType.VIDEO:
                return await self._process_video(input_data)
            
            elif input_data.type == InputType.FILE:
                return await self._process_file(input_data)
            
            else:
                self.logger.warning(f"Unsupported input type: {input_data.type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to process input: {e}")
            return None
    
    async def _process_image(self, input_data: MultimodalInput) -> TextSemantic:
        try:
            image_bytes = input_data.content if isinstance(input_data.content, bytes) else base64.b64decode(input_data.content)
            image = Image.open(io.BytesIO(image_bytes))
            
            width, height = image.size
            format = image.format
            mode = image.mode
            
            description = f"""
图片描述：
- 格式：{format}
- 尺寸：{width} x {height}
- 模式：{mode}
- 内容分析：需要图像理解模型进行深度分析
"""
            return TextSemantic(
                content=description.strip(),
                source_type=InputType.IMAGE,
                confidence=0.85
            )
        except Exception as e:
            self.logger.error(f"Image processing failed: {e}")
            return TextSemantic(
                content=f"图片处理失败: {str(e)}",
                source_type=InputType.IMAGE,
                confidence=0.3
            )
    
    async def _process_audio(self, input_data: MultimodalInput) -> TextSemantic:
        description = f"""
音频描述：
- 格式：{input_data.metadata.get('format', '未知')}
- 时长：{input_data.metadata.get('duration', '未知')}秒
- 采样率：{input_data.metadata.get('sample_rate', '未知')}Hz
- 内容：需要语音转文字模型进行转录
"""
        return TextSemantic(
            content=description.strip(),
            source_type=InputType.AUDIO,
            confidence=0.8
        )
    
    async def _process_video(self, input_data: MultimodalInput) -> TextSemantic:
        description = f"""
视频描述：
- 格式：{input_data.metadata.get('format', '未知')}
- 时长：{input_data.metadata.get('duration', '未知')}秒
- 分辨率：{input_data.metadata.get('resolution', '未知')}
- 帧率：{input_data.metadata.get('fps', '未知')}fps
- 内容：需要视频理解模型进行分析
"""
        return TextSemantic(
            content=description.strip(),
            source_type=InputType.VIDEO,
            confidence=0.75
        )
    
    async def _process_file(self, input_data: MultimodalInput) -> TextSemantic:
        file_name = input_data.metadata.get('filename', 'unknown')
        file_size = input_data.metadata.get('size', 'unknown')
        
        description = f"""
文件描述：
- 文件名：{file_name}
- 大小：{file_size}字节
- 类型：{input_data.metadata.get('content_type', '未知')}
- 内容：需要文件解析器进行提取
"""
        return TextSemantic(
            content=description.strip(),
            source_type=InputType.FILE,
            confidence=0.7
        )
    
    def combine_semantics(self, semantics: List[TextSemantic]) -> str:
        if not semantics:
            return ""
        
        parts = []
        for sem in semantics:
            parts.append(f"【{sem.source_type.value}输入】\n{sem.content}\n")
        
        return "\n".join(parts)
    
    async def process_and_combine(self, inputs: List[MultimodalInput]) -> str:
        semantics = await self.process(inputs)
        return self.combine_semantics(semantics)