import asyncio
from hpd_agent import HPDLLMAgent, AgentConfig

async def main():
    config = AgentConfig(
        max_workers=10,
        log_level="DEBUG"
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
    
    print(f"Expert Query: {expert_query[:100]}...")
    result = await agent.run(expert_query)
    
    print("\n=== Expert Mode Result ===")
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Answer:\n{result['answer']}")
        print(f"\nScore: {result.get('score', 0.0):.2f}")
        print(f"Route Level: {result['route_level']}")
        print(f"Node: {result['node']}")

if __name__ == "__main__":
    asyncio.run(main())