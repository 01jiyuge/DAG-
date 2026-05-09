from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field
from hpd_agent.utils.logger import get_logger

class RouteLevel(str, Enum):
    LEVEL_1 = "level_1"
    LEVEL_2 = "level_2"
    LEVEL_3 = "level_3"

class RouteNode(BaseModel):
    id: str
    name: str
    level: RouteLevel
    model: str
    max_tokens: int = 4096
    temperature: float = 0.7
    capabilities: List[str] = Field(default_factory=list)

class RouteDecision(BaseModel):
    node_id: str
    node_name: str
    level: RouteLevel
    confidence: float

class HierarchicalRouter:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.nodes: Dict[str, RouteNode] = {}
        self._initialize_default_nodes()
    
    def _initialize_default_nodes(self):
        self.nodes = {
            "node_simple": RouteNode(
                id="node_simple",
                name="Simple Node",
                level=RouteLevel.LEVEL_1,
                model="gpt-3.5-turbo",
                max_tokens=4096,
                temperature=0.5,
                capabilities=["simple_query", "summarization", "classification"]
            ),
            "node_complex": RouteNode(
                id="node_complex",
                name="Complex Node",
                level=RouteLevel.LEVEL_2,
                model="gpt-4",
                max_tokens=8192,
                temperature=0.7,
                capabilities=["complex_reasoning", "analysis", "code_generation"]
            ),
            "node_expert": RouteNode(
                id="node_expert",
                name="Expert Node",
                level=RouteLevel.LEVEL_3,
                model="gpt-4-turbo",
                max_tokens=128000,
                temperature=0.3,
                capabilities=["deep_analysis", "research", "advanced_reasoning", "multi_step"]
            )
        }
    
    def add_node(self, node: RouteNode):
        self.nodes[node.id] = node
        self.logger.info(f"Added route node: {node.name}")
    
    def remove_node(self, node_id: str):
        if node_id in self.nodes:
            del self.nodes[node_id]
            self.logger.info(f"Removed route node: {node_id}")
    
    def analyze_difficulty(self, query: str, context: Optional[Dict[str, Any]] = None) -> float:
        length_score = min(len(query) / 500, 1.0)
        
        complexity_keywords = [
            "analyze", "explain", "solve", "design", "develop", "optimize", "debug", "research",
            "comprehensive", "complex", "principle", "mechanism", "algorithm", "pipeline",
            "implement", "architecture", "evaluate", "impact", "recommendation", "quantum",
            "machine learning", "deep learning", "neural network", "feature engineering"
        ]
        keyword_score = sum(1 for kw in complexity_keywords if kw.lower() in query.lower()) / len(complexity_keywords)
        
        sentence_count = len(query.split('.')) if query else 0
        sentence_score = min(sentence_count / 5, 1.0)
        
        context_score = len(context.get("history", [])) / 10 if context else 0
        
        difficulty = (length_score * 0.3 + keyword_score * 0.4 + sentence_score * 0.2 + context_score * 0.1)
        return min(max(difficulty, 0.1), 0.95)
    
    def route(self, query: str, context: Optional[Dict[str, Any]] = None) -> RouteDecision:
        difficulty = self.analyze_difficulty(query, context)
        self.logger.info(f"Analyzed difficulty: {difficulty:.2f}")
        
        if difficulty < 0.3:
            node = self.nodes["node_simple"]
            level = RouteLevel.LEVEL_1
        elif difficulty < 0.7:
            node = self.nodes["node_complex"]
            level = RouteLevel.LEVEL_2
        else:
            node = self.nodes["node_expert"]
            level = RouteLevel.LEVEL_3
        
        confidence = min(0.95, difficulty + 0.1)
        self.logger.info(f"Routing to {node.name} (level: {level}, confidence: {confidence:.2f})")
        
        return RouteDecision(
            node_id=node.id,
            node_name=node.name,
            level=level,
            confidence=confidence
        )
    
    def get_node(self, node_id: str) -> Optional[RouteNode]:
        return self.nodes.get(node_id)
    
    def get_nodes_by_level(self, level: RouteLevel) -> List[RouteNode]:
        return [node for node in self.nodes.values() if node.level == level]