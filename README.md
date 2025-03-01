# NSSGuru3.7
Multi agentic flow to work on the NSS framework 
# Agentic AI Solution - Repository Structure

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
