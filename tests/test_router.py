import pytest
from hpd_agent.routing import HierarchicalRouter, RouteLevel

def test_difficulty_analysis():
    router = HierarchicalRouter()
    
    simple_query = "Hello, how are you?"
    difficulty = router.analyze_difficulty(simple_query)
    assert 0.1 <= difficulty < 0.3
    
    complex_query = "Analyze machine learning algorithms including neural network architectures, deep learning techniques, and feature engineering methods for predictive modeling"
    difficulty = router.analyze_difficulty(complex_query)
    assert 0.2 <= difficulty < 0.8
    
    expert_query = "Design a comprehensive machine learning pipeline for fraud detection with feature engineering, model selection, hyperparameter tuning, ensemble methods, and production deployment architecture"
    difficulty = router.analyze_difficulty(expert_query)
    assert difficulty >= 0.2

def test_route_simple():
    router = HierarchicalRouter()
    decision = router.route("What is the capital of France?")
    assert decision.level == RouteLevel.LEVEL_1
    assert decision.node_id == "node_simple"

def test_route_complex():
    router = HierarchicalRouter()
    decision = router.route("Analyze machine learning algorithms including neural network architectures and deep learning techniques for predictive modeling")
    assert decision.level in [RouteLevel.LEVEL_1, RouteLevel.LEVEL_2]

def test_route_expert():
    router = HierarchicalRouter()
    decision = router.route("Design a comprehensive machine learning pipeline for fraud detection with feature engineering, model selection, hyperparameter tuning, ensemble methods, and production deployment architecture")
    assert decision.level in [RouteLevel.LEVEL_1, RouteLevel.LEVEL_2, RouteLevel.LEVEL_3]