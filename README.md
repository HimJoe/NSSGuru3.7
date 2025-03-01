# NSSGuru3.7
Multi agentic flow to work on the NSS framework 

# Agentic AI Solution

A framework for building and deploying autonomous AI agents with reasoning capabilities, tool use, and persistent memory for innovators to deploy and use NSS framework. 

## Overview

This framework provides a foundation for building intelligent AI agents that can:- 
- Reason about complex problems with a multi-step thinking process
- Use various tools to interact with the world (web search, code execution, etc.)
- Maintain persistent memory across interactions
- Collaborate in multi-agent systems
- Execute sophisticated workflows autonomously

## Key Features

- **Modular Agent Architecture**: Easily customize and extend agent capabilities
- **Powerful Memory System**: Both short-term working memory and long-term vector storage
- **Extensive Tool Library**: Ready-to-use tools for common tasks
- **Flexible Prompt Templates**: Create specialized agents for different domains
- **Multi-Agent Orchestration**: Coordinate multiple agents to solve complex problems

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/agentic-ai-solution.git
cd agentic-ai-solution

# Install the package
pip install -e .
```

## Quick Start

```python
from agentic_ai.agents import ReasoningAgent
from agentic_ai.tools import WebSearchTool
from agentic_ai.memory import VectorStore

# Initialize memory
memory = VectorStore()

# Initialize tools
web_search = WebSearchTool()

# Create an agent with reasoning capabilities
agent = ReasoningAgent(
    tools=[web_search],
    memory=memory,
    system_prompt="You are a helpful research assistant that finds information and answers questions."
)

# Run the agent
response = agent.run("Research the latest advances in reinforcement learning and summarize the key findings.")
print(response)
```

## Examples

Check out the [examples](src/examples/) directory for more detailed usage examples:

- [Basic Agent](src/examples/basic_agent.py): Simple agent setup and usage
- [Multi-Agent System](src/examples/multi_agent_system.py): Coordinating multiple agents
- [Custom Tool Example](src/examples/custom_tool_example.py): Creating your own tools

## Documentation

- [Architecture Overview](docs/architecture.md)
- [API Reference](docs/api_reference.md)
- [User Guide](docs/user_guide.md)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

This document outlines the complete folder structure and files needed for NSS Agentic AI solution. The organization of the repository, followed by the content of each important file.

## Repository Structure 
agentic-ai-solution/
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py
├── docs/
│   ├── architecture.md
│   ├── api_reference.md
│   └── user_guide.md
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_agents.py
│   ├── test_memory.py
│   └── test_tools.py
└── src/
    ├── agentic_ai/
    │   ├── __init__.py
    │   ├── config.py
    │   ├── main.py
    │   ├── agents/
    │   │   ├── __init__.py
    │   │   ├── base_agent.py
    │   │   ├── reasoning_agent.py
    │   │   └── tool_using_agent.py
    │   ├── memory/
    │   │   ├── __init__.py
    │   │   ├── vector_store.py
    │   │   └── working_memory.py
    │   ├── tools/
    │   │   ├── __init__.py
    │   │   ├── base_tool.py
    │   │   ├── web_search.py
    │   │   ├── code_execution.py
    │   │   └── file_operations.py
    │   ├── prompts/
    │   │   ├── __init__.py
    │   │   ├── system_prompts.py
    │   │   └── templates.py
    │   └── utils/
    │       ├── __init__.py
    │       ├── logger.py
    │       └── helpers.py
    └── examples/
        ├── basic_agent.py
        ├── multi_agent_system.py
        └── custom_tool_example.py
