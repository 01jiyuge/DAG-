import asyncio
from hpd_agent import HPDLLMAgent, AgentConfig

async def basic_expert_query():
    print("=" * 60)
    print("示例1: 基础专家模式查询")
    print("=" * 60)

    config = AgentConfig(
        llm_provider="deepseek",
        default_model="deepseek-chat",
        max_workers=10,
        log_level="INFO"
    )

    agent = HPDLLMAgent(config)

    expert_query = """
    Design a scalable microservices architecture for an e-commerce platform.
    Include:
    1. Service decomposition strategy
    2. Database design considerations
    3. API gateway pattern
    4. Event-driven architecture
    5. Security considerations
    """

    print(f"Expert Query: {expert_query[:80]}...")
    result = await agent.run(expert_query)

    print(f"\nSuccess: {result['success']}")
    if result['success']:
        print(f"Answer:\n{result['answer']}")
        print(f"\nScore: {result.get('score', 0.0):.2f}")
        print(f"Route Level: {result['route_level']}")
        print(f"Node: {result['node']}")
        print(f"Tasks: {result.get('success_count', 0)}/{result.get('task_count', 0)}")

    return result


async def multi_path_generation():
    print("\n" + "=" * 60)
    print("示例2: 多路径生成模式")
    print("=" * 60)

    config = AgentConfig(
        llm_provider="deepseek",
        default_model="deepseek-chat",
        max_workers=5,
        log_level="INFO"
    )

    agent = HPDLLMAgent(config)

    complex_query = """
    Analyze the impact of artificial intelligence on software development industry.
    Provide multiple perspectives including:
    - Current AI tools and their capabilities
    - Future trends and predictions
    - Best practices for AI-assisted development
    - Potential challenges and solutions
    """

    print(f"Complex Query: {complex_query[:80]}...")
    result = await agent.run(complex_query)

    print(f"\nSuccess: {result['success']}")
    if result['success']:
        print(f"Answer:\n{result['answer'][:500]}...")
        print(f"\n[Full answer length: {len(result.get('answer', ''))} characters]")
        print(f"Score: {result.get('score', 0.0):.2f}")
        print(f"Route Level: {result['route_level']}")

    return result


async def technical_deep_dive():
    print("\n" + "=" * 60)
    print("示例3: 技术深度分析模式")
    print("=" * 60)

    config = AgentConfig(
        llm_provider="deepseek",
        default_model="deepseek-chat",
        max_workers=10,
        timeout=600,
        log_level="INFO"
    )

    agent = HPDLLMAgent(config)

    technical_query = """
    Explain the internals of a distributed consensus algorithm (like Raft or Paxos).
    Cover:
    1. Leader election mechanism
    2. Log replication process
    3. Safety and consistency guarantees
    4. Membership changes
    5. Practical optimizations and limitations
    """

    print(f"Technical Query: {technical_query[:80]}...")
    result = await agent.run(technical_query)

    print(f"\nSuccess: {result['success']}")
    if result['success']:
        print(f"Answer:\n{result['answer'][:800]}...")
        print(f"\nScore: {result.get('score', 0.0):.2f}")
        print(f"Route Level: {result['route_level']}")
        print(f"Node: {result['node']}")

    return result


async def business_analysis():
    print("\n" + "=" * 60)
    print("示例4: 商业分析模式")
    print("=" * 60)

    config = AgentConfig(
        llm_provider="deepseek",
        default_model="deepseek-chat",
        max_workers=5,
        log_level="INFO"
    )

    agent = HPDLLMAgent(config)

    business_query = """
    Develop a comprehensive market entry strategy for a SaaS product.
    Include:
    1. Market analysis and target customer identification
    2. Competitive landscape assessment
    3. Pricing strategy and business model
    4. Go-to-market channels and tactics
    5. Success metrics and KPIs
    """

    print(f"Business Query: {business_query[:80]}...")
    result = await agent.run(business_query)

    print(f"\nSuccess: {result['success']}")
    if result['success']:
        print(f"Answer:\n{result['answer'][:600]}...")
        print(f"\nScore: {result.get('score', 0.0):.2f}")
        print(f"Route Level: {result['route_level']}")
        print(f"Tasks Completed: {result.get('success_count', 0)}/{result.get('task_count', 0)}")

    return result


async def simple_vs_expert_comparison():
    print("\n" + "=" * 60)
    print("示例5: 简单查询 vs 专家查询对比")
    print("=" * 60)

    config = AgentConfig(
        llm_provider="deepseek",
        default_model="deepseek-chat",
        max_workers=5,
        log_level="WARNING"
    )

    agent = HPDLLMAgent(config)

    simple_query = "What is Python?"
    complex_query = "Design a comprehensive Python monorepo architecture for a large-scale ML platform with support for distributed training, experiment tracking, model versioning, and A/B testing."

    print(f"\n[Simple Query] {simple_query}")
    simple_result = await agent.run(simple_query)
    print(f"  → Route Level: {simple_result['route_level']}")
    print(f"  → Node: {simple_result['node']}")
    print(f"  → Tasks: {simple_result.get('task_count', 0)}")

    print(f"\n[Complex Query] {complex_query[:60]}...")
    complex_result = await agent.run(complex_query)
    print(f"  → Route Level: {complex_result['route_level']}")
    print(f"  → Node: {complex_result['node']}")
    print(f"  → Tasks: {complex_result.get('task_count', 0)}")

    return simple_result, complex_result


async def batch_processing():
    print("\n" + "=" * 60)
    print("示例6: 批量查询处理")
    print("=" * 60)

    config = AgentConfig(
        llm_provider="deepseek",
        default_model="deepseek-chat",
        max_workers=10,
        log_level="WARNING"
    )

    agent = HPDLLMAgent(config)

    queries = [
        "What is machine learning?",
        "Explain neural network architectures",
        "Design patterns in software engineering",
        "Cloud computing best practices",
        "Data structures and algorithms overview"
    ]

    print(f"Processing {len(queries)} queries in batch...\n")

    results = []
    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] Query: {query[:40]}...")
        result = await agent.run(query)
        results.append(result)
        print(f"  → Success: {result['success']}, Level: {result['route_level']}")

    success_count = sum(1 for r in results if r['success'])
    print(f"\nBatch Processing Summary:")
    print(f"  Total: {len(queries)}")
    print(f"  Success: {success_count}")
    print(f"  Failed: {len(queries) - success_count}")

    return results


async def main():
    print("\n" + "=" * 60)
    print("HPD-Agent Expert Mode Examples")
    print("=" * 60)

    await basic_expert_query()

    await multi_path_generation()

    await technical_deep_dive()

    await business_analysis()

    await simple_vs_expert_comparison()

    await batch_processing()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
