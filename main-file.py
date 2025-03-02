"""
Main Entry Point

This module provides the main entry point for the Agentic AI framework,
including high-level functions for creating and running agents.
"""

import logging
import os
from typing import Dict, List, Optional, Any, Union

from .agents import BaseAgent, ReasoningAgent, ToolUsingAgent
from .tools import BaseTool
from .memory import WorkingMemory, VectorStore
from .config import get_config, update_config
from .utils.logger import setup_logging, get_logger

# Get module logger
logger = get_logger("main")

def create_agent(
    agent_type: str = "reasoning",
    tools: Optional[List[BaseTool]] = None,
    memory_type: Optional[str] = "working",
    system_prompt: Optional[str] = None,
    agent_config: Optional[Dict[str, Any]] = None
) -> BaseAgent:
    """
    Create an agent with the specified configuration.
    
    This is a high-level factory function that simplifies agent creation
    by handling common configurations.
    
    Args:
        agent_type: Type of agent to create ("base", "reasoning", or "tool_using")
        tools: List of tools for the agent to use
        memory_type: Type of memory to use ("working", "vector", or None)
        system_prompt: Custom system prompt for the agent
        agent_config: Additional agent configuration parameters
        
    Returns:
        Configured agent instance
    """
    # Initialize tools list if not provided
    tools = tools or []
    
    # Initialize agent configuration if not provided
    agent_config = agent_config or {}
    
    # Initialize memory based on type
    memory = None
    if memory_type == "working":
        memory = WorkingMemory()
    elif memory_type == "vector":
        memory = VectorStore()
    
    # Get common configuration parameters
    config = get_config()
    model = agent_config.get("model", config["llm"]["model"])
    temperature = agent_config.get("temperature", config["llm"]["temperature"])
    max_tokens = agent_config.get("max_tokens", config["llm"]["max_tokens"])
    
    # Create agent based on type
    if agent_type == "base":
        agent = BaseAgent(
            tools=tools,
            memory=memory,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **agent_config
        )
    elif agent_type == "reasoning":
        reasoning_steps = agent_config.get("reasoning_steps", 3)
        agent = ReasoningAgent(
            tools=tools,
            memory=memory,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_steps=reasoning_steps,
            **agent_config
        )
    elif agent_type == "tool_using":
        max_tool_iterations = agent_config.get("max_tool_iterations", 5)
        agent = ToolUsingAgent(
            tools=tools,
            memory=memory,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_tool_iterations=max_tool_iterations,
            **agent_config
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    logger.info(f"Created {agent_type} agent with {len(tools)} tools and {memory_type} memory")
    return agent

def run_agent_session(
    agent: BaseAgent, 
    initial_prompt: Optional[str] = None,
    interactive: bool = True
) -> List[Dict[str, str]]:
    """
    Run an interactive or single-prompt agent session.
    
    Args:
        agent: The agent to run
        initial_prompt: Initial prompt to send to the agent
        interactive: Whether to run in interactive mode
        
    Returns:
        List of message dictionaries representing the conversation
    """
    conversation = []
    
    # Process initial prompt if provided
    if initial_prompt:
        logger.info(f"Processing initial prompt: {initial_prompt[:50]}...")
        response = agent.run(initial_prompt)
        
        conversation.append({"role": "user", "content": initial_prompt})
        conversation.append({"role": "assistant", "content": response})
        
        print(f"User: {initial_prompt}")
        print(f"Assistant: {response}")
    
    # Run interactive session if requested
    if interactive:
        print("\nEntering interactive mode. Type 'exit' to end the session.")
        
        while True:
            user_input = input("\nUser: ")
            
            if user_input.lower() in ["exit", "quit", "bye"]:
                break
            
            logger.info(f"Processing user input: {user_input[:50]}...")
            response = agent.run(user_input)
            
            conversation.append({"role": "user", "content": user_input})
            conversation.append({"role": "assistant", "content": response})
            
            print(f"Assistant: {response}")
    
    return conversation

def initialize_framework(
    log_level: str = "INFO",
    config_file: Optional[str] = None,
    api_key: Optional[str] = None
) -> None:
    """
    Initialize the framework with the specified configuration.
    
    Args:
        log_level: Logging level to use
        config_file: Path to configuration file
        api_key: API key for LLM provider
    """
    # Set up logging
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    setup_logging(level=numeric_level)
    
    # Load configuration if specified
    if config_file:
        from .config import load_config
        load_config(config_file=config_file)
    
    # Set API key if provided
    if api_key:
        update_config({"llm": {"api_key": api_key}})
    
    logger.info("Framework initialized successfully")

def main():
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Agentic AI Framework")
    parser.add_argument("--agent-type", choices=["base", "reasoning", "tool_using"], default="reasoning", help="Type of agent to create")
    parser.add_argument("--memory-type", choices=["working", "vector", "none"], default="working", help="Type of memory to use")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--config-file", help="Path to configuration file")
    parser.add_argument("--prompt", help="Initial prompt to send to the agent")
    parser.add_argument("--api-key", help="API key for LLM provider")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Initialize the framework
    initialize_framework(
        log_level=args.log_level,
        config_file=args.config_file,
        api_key=args.api_key
    )
    
    # Create an agent
    memory_type = None if args.memory_type == "none" else args.memory_type
    agent = create_agent(
        agent_type=args.agent_type,
        memory_type=memory_type
    )
    
    # Run agent session
    run_agent_session(
        agent=agent,
        initial_prompt=args.prompt,
        interactive=args.interactive
    )

if __name__ == "__main__":
    main()
