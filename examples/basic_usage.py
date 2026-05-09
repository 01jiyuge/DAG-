import asyncio
from hpd_agent import HPDLLMAgent, AgentConfig

async def main():
    config = AgentConfig(
        max_workers=5,
        log_level="INFO"
    )
    
    agent = HPDLLMAgent(config)
    
    simple_query = "What is the capital of France?"
    print(f"Simple Query: {simple_query}")
    result = await agent.run(simple_query)
    print(f"Result: {result}\n")
    
    complex_query = "Explain the principles of machine learning and provide examples of applications"
    print(f"Complex Query: {complex_query}")
    result = await agent.run(complex_query)
    print(f"Result: {result}\n")

if __name__ == "__main__":
    asyncio.run(main())