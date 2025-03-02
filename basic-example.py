"""
Basic Agent Example

This script demonstrates how to create and use a basic agent with the Agentic AI framework.
It covers agent initialization, tool configuration, and basic interaction patterns.
"""

import os
import sys
import logging

# Add parent directory to path to allow importing the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the framework
from agentic_ai import BaseAgent, ReasoningAgent, ToolUsingAgent
from agentic_ai import WebSearchTool, CodeExecutionTool
from agentic_ai import WorkingMemory
from agentic_ai.utils.logger import setup_logging

# Set up logging
setup_logging(level=logging.INFO, log_to_console=True)

def main():
    """Run the basic agent example."""
    print("Agentic AI Framework - Basic Agent Example")
    print("==========================================")
    
    # Initialize tools
    web_search = WebSearchTool()
    code_execution = CodeExecutionTool()
    
    # Initialize memory
    memory = WorkingMemory()
    
    # Create a basic agent
    agent = BaseAgent(
        tools=[web_search, code_execution],
        memory=memory,
        name="BasicExample"
    )
    
    print("\nCreated a basic agent with web search and code execution tools.")
    print("Let's ask it a question...\n")
    
    # Run the agent with a sample input
    query = "What is the formula for calculating compound interest? Can you show me how to implement it in Python?"
    print(f"Query: {query}\n")
    
    response = agent.run(query)
    print(f"Response:\n{response}\n")
    
    # Create a reasoning agent
    reasoning_agent = ReasoningAgent(
        tools=[web_search, code_execution],
        memory=memory,
        name="ReasoningExample",
        reasoning_steps=3
    )
    
    print("\nCreated a reasoning agent with explicit thinking steps.")
    print("Let's ask it a more complex question...\n")
    
    # Run the reasoning agent
    complex_query = "I need to analyze data on climate change trends. What approach would you recommend, and what tools should I use?"
    print(f"Query: {complex_query}\n")
    
    detailed_response = reasoning_agent.run(complex_query)
    print(f"Detailed response with reasoning:\n{detailed_response}\n")
    
    # Extract just the final answer
    concise_response = reasoning_agent.run(complex_query, extract_final_answer=True)
    print(f"Concise response (final answer only):\n{concise_response}\n")
    
    # Create a tool-using agent
    tool_agent = ToolUsingAgent(
        tools=[web_search, code_execution],
        memory=memory,
        name="ToolUsingExample",
        max_tool_iterations=3
    )
    
    print("\nCreated a tool-using agent specialized in effective tool use.")
    print("Let's give it a task that requires tools...\n")
    
    # Run the tool-using agent
    tool_query = "Find information about population growth rates and create a Python function to visualize it as a line chart."
    print(f"Query: {tool_query}\n")
    
    tool_response = tool_agent.run(tool_query)
    print(f"Tool-using agent response:\n{tool_response}\n")
    
    print("Example complete! You've seen how to create and use three types of agents:")
    print("1. BaseAgent - Simple agent with tool access")
    print("2. ReasoningAgent - Agent with explicit thinking steps")
    print("3. ToolUsingAgent - Agent specialized in effective tool use")

if __name__ == "__main__":
    main()
