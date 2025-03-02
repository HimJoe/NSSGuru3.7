"""
Agentic AI Framework

A framework for building and deploying autonomous AI agents with reasoning capabilities,
tool use, and persistent memory.

This package provides modular components for building intelligent AI agents that can:
- Reason about complex problems with a multi-step thinking process
- Use various tools to interact with the world
- Maintain persistent memory across interactions
- Collaborate in multi-agent systems
- Execute sophisticated workflows autonomously
"""

# Package metadata
__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import commonly used modules for easy access
from .agents.base_agent import BaseAgent
from .agents.reasoning_agent import ReasoningAgent
from .agents.tool_using_agent import ToolUsingAgent

from .memory.vector_store import VectorStore
from .memory.working_memory import WorkingMemory

from .tools.base_tool import BaseTool, tool
from .tools.web_search import WebSearchTool, web_search_simple
from .tools.code_execution import CodeExecutionTool

from .utils.logger import get_logger, setup_logging

# Set up default logging
setup_logging()

# Define what's available when using `from agentic_ai import *`
__all__ = [
    # Agents
    "BaseAgent",
    "ReasoningAgent",
    "ToolUsingAgent",
    
    # Memory
    "VectorStore",
    "WorkingMemory",
    
    # Tools
    "BaseTool",
    "tool",
    "WebSearchTool",
    "web_search_simple",
    "CodeExecutionTool",
    
    # Utils
    "get_logger",
    "setup_logging",
]

# Display package information when imported directly
if __name__ == "__main__":
    print(f"Agentic AI Framework v{__version__}")
    print("A framework for building autonomous AI agents with reasoning, tool use, and memory")
