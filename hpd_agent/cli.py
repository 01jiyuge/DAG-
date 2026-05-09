import asyncio
import argparse
import sys
import json
from typing import Optional
from .config import AgentConfig
from .agent import HPDLLMAgent
from .utils.logger import get_logger, setup_logging
from .di import DIContainer, register_default_providers

logger = get_logger(__name__)

def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                     HPD-Agent v0.2.0                        ║
║        LLM Agent DAG Orchestration System                   ║
║    Hierarchical | Parallel | Dynamic                        ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_result(result: dict):
    if result["success"]:
        print("\n" + "="*60)
        print("RESULT")
        print("="*60)
        print(result["answer"])
        print("\n" + "-"*60)
        print("METRICS")
        print("-"*60)
        print(f"Score:          {result.get('score', 0.0):.4f}")
        print(f"Route Level:    {result.get('route_level', 'unknown')}")
        print(f"Node:           {result.get('node', 'unknown')}")
        print(f"Tasks:          {result.get('success_count', 0)}/{result.get('task_count', 0)} succeeded")
        print("-"*60)
    else:
        print("\n" + "="*60)
        print("ERROR")
        print("="*60)
        print(f"Error: {result.get('error', 'Unknown error')}")
        if 'details' in result:
            print("\nDetails:")
            for detail in result['details']:
                print(f"  - {detail}")
        print("="*60)

def run_agent(query: str, config: AgentConfig) -> dict:
    try:
        register_default_providers(config)
        agent = HPDLLMAgent(config)
        result = asyncio.run(agent.run(query))
        return result
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        return {"success": False, "error": str(e)}

def main():
    print_banner()
    
    parser = argparse.ArgumentParser(
        description="HPD-Agent: LLM Agent DAG Orchestration System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--query", "-q", type=str, required=True, help="Input query")
    
    parser.add_argument("--provider", "-p", type=str, default="openai", 
                        choices=["openai", "dashscope", "anthropic", "google", "local", "deepseek"],
                        help="LLM provider")
    parser.add_argument("--model", "-m", type=str, default="gpt-4", help="LLM model name")
    parser.add_argument("--temperature", "-t", type=float, default=0.7, help="LLM temperature")
    parser.add_argument("--max-tokens", type=int, default=4096, help="LLM max tokens")
    
    parser.add_argument("--max-workers", type=int, default=10, help="Max parallel workers")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retry attempts")
    
    parser.add_argument("--log-level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--log-file", type=str, help="Log file path")
    
    parser.add_argument("--cache-backend", type=str, default="memory", 
                        choices=["memory", "redis"], help="Cache backend")
    parser.add_argument("--cache-ttl", type=int, default=3600, help="Cache TTL in seconds")
    
    parser.add_argument("--output-json", "-j", action="store_true", 
                        help="Output result as JSON")
    
    args = parser.parse_args()
    
    setup_logging(log_level=args.log_level, log_file=args.log_file)
    
    try:
        config = AgentConfig(
            llm_provider=args.provider,
            default_model=args.model,
            llm_temperature=args.temperature,
            llm_max_tokens=args.max_tokens,
            max_workers=args.max_workers,
            timeout=args.timeout,
            max_retries=args.max_retries,
            log_level=args.log_level,
            cache_backend=args.cache_backend,
            cache_ttl_seconds=args.cache_ttl
        )
        
        logger.info(f"Starting HPD-Agent with provider: {args.provider}, model: {args.model}")
        
        result = run_agent(args.query, config)
        
        if args.output_json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print_result(result)
        
        if not result["success"]:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"CLI execution failed: {e}", exc_info=True)
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()