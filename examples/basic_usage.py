import asyncio
from hpd_agent import HPDLLMAgent, AgentConfig

async def simple_queries():
    print("=" * 60)
    print("示例1: 简单查询演示")
    print("=" * 60)

    config = AgentConfig(
        llm_provider="deepseek",
        default_model="deepseek-chat",
        max_workers=5,
        log_level="INFO"
    )

    agent = HPDLLMAgent(config)

    queries = [
        "What is the capital of France?",
        "How does photosynthesis work?",
        "What is the meaning of life?"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        result = await agent.run(query)
        print(f"  → Success: {result['success']}")
        print(f"  → Route Level: {result['route_level']}")
        print(f"  → Node: {result['node']}")
        print(f"  → Answer: {result['answer'][:100]}...")

    return result


async def complex_queries():
    print("\n" + "=" * 60)
    print("示例2: 复杂查询演示")
    print("=" * 60)

    config = AgentConfig(
        llm_provider="deepseek",
        default_model="deepseek-chat",
        max_workers=5,
        log_level="INFO"
    )

    agent = HPDLLMAgent(config)

    query = """
    Explain the principles of machine learning and provide examples of applications.
    Include:
    1. Types of machine learning (supervised, unsupervised, reinforcement)
    2. Key algorithms and their use cases
    3. Real-world applications across industries
    4. Current challenges and future directions
    """

    print(f"Query: {query[:60]}...")
    result = await agent.run(query)

    print(f"\nSuccess: {result['success']}")
    if result['success']:
        print(f"Route Level: {result['route_level']}")
        print(f"Node: {result['node']}")
        print(f"Tasks: {result.get('success_count', 0)}/{result.get('task_count', 0)}")
        print(f"Score: {result.get('score', 0.0):.2f}")
        print(f"\nAnswer Preview:\n{result['answer'][:300]}...")

    return result


async def configuration_demo():
    print("\n" + "=" * 60)
    print("示例3: 不同配置演示")
    print("=" * 60)

    configs = [
        {
            "name": "Fast Response (Low Temp)",
            "config": AgentConfig(
                llm_provider="deepseek",
                default_model="deepseek-chat",
                llm_temperature=0.1,
                max_workers=5,
                log_level="WARNING"
            )
        },
        {
            "name": "Creative Response (High Temp)",
            "config": AgentConfig(
                llm_provider="deepseek",
                default_model="deepseek-chat",
                llm_temperature=1.0,
                max_workers=5,
                log_level="WARNING"
            )
        },
        {
            "name": "Long Output (High Tokens)",
            "config": AgentConfig(
                llm_provider="deepseek",
                default_model="deepseek-chat",
                llm_max_tokens=8192,
                max_workers=5,
                log_level="WARNING"
            )
        }
    ]

    query = "Tell me a short story about AI."

    for cfg in configs:
        print(f"\n--- {cfg['name']} ---")
        agent = HPDLLMAgent(cfg['config'])
        result = await agent.run(query)
        print(f"  → Success: {result['success']}")
        print(f"  → Answer length: {len(result.get('answer', ''))} chars")
        print(f"  → Answer: {result['answer'][:80]}...")

    return configs


async def routing_demo():
    print("\n" + "=" * 60)
    print("示例4: 路由分级演示")
    print("=" * 60)

    config = AgentConfig(
        llm_provider="deepseek",
        default_model="deepseek-chat",
        max_workers=5,
        log_level="INFO"
    )

    agent = HPDLLMAgent(config)

    test_cases = [
        ("Simple", "What is 2+2?"),
        ("Medium", "Compare and contrast object-oriented and functional programming."),
        ("Complex", "Design a comprehensive system architecture for a real-time video streaming platform that handles millions of concurrent users, including CDN integration, transcoding, load balancing, and disaster recovery strategies.")
    ]

    for level, query in test_cases:
        print(f"\n[{level} Query] {query[:50]}...")
        result = await agent.run(query)
        print(f"  → Route Level: {result['route_level']}")
        print(f"  → Node: {result['node']}")
        print(f"  → Difficulty: {result.get('difficulty', 'N/A')}")
        print(f"  → Tasks Created: {result.get('task_count', 0)}")

    return test_cases


async def main():
    print("\n" + "=" * 60)
    print("HPD-Agent Basic Usage Examples")
    print("=" * 60)

    await simple_queries()

    await complex_queries()

    await configuration_demo()

    await routing_demo()

    print("\n" + "=" * 60)
    print("All basic examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
