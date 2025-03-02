"""
Agentic AI Solution - Streamlit Application

This application provides a web interface for interacting with the Agentic AI framework.
It allows users to configure and use different types of agents with various tools and memory systems.
"""

import os
import sys
import streamlit as st
import json
import time
from typing import Dict, List, Any, Optional

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath('src'))

# Import the framework
from agentic_ai import BaseAgent, ReasoningAgent, ToolUsingAgent
from agentic_ai import WebSearchTool, CodeExecutionTool, FileOperationsTool
from agentic_ai import WorkingMemory, VectorStore
from agentic_ai.prompts import system_prompts
from agentic_ai.utils.logger import setup_logging, get_logger
from agentic_ai.config import load_config, update_config

# Set up logging
setup_logging(level="INFO", log_to_console=True)
logger = get_logger("streamlit_app")

# Page configuration
st.set_page_config(
    page_title="Agentic AI Solution",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
try:
    config = load_config()
    logger.info("Configuration loaded successfully")
except Exception as e:
    logger.error(f"Error loading configuration: {str(e)}")
    config = {}

# Helper functions
def get_agent_class(agent_type: str):
    """Get the agent class based on type."""
    if agent_type == "Base Agent":
        return BaseAgent
    elif agent_type == "Reasoning Agent":
        return ReasoningAgent
    elif agent_type == "Tool-Using Agent":
        return ToolUsingAgent
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

def get_memory_class(memory_type: str):
    """Get the memory class based on type."""
    if memory_type == "Working Memory":
        return WorkingMemory
    elif memory_type == "Vector Store":
        return VectorStore
    else:
        return None

def get_system_prompt(prompt_type: str, custom_prompt: str = None):
    """Get the system prompt based on type."""
    if prompt_type == "Default":
        return system_prompts.DEFAULT_SYSTEM_PROMPT
    elif prompt_type == "Reasoning":
        return system_prompts.REASONING_SYSTEM_PROMPT
    elif prompt_type == "Tool-Using":
        return system_prompts.TOOL_USING_SYSTEM_PROMPT
    elif prompt_type == "Research":
        return system_prompts.RESEARCH_AGENT_PROMPT
    elif prompt_type == "Creative":
        return system_prompts.CREATIVE_AGENT_PROMPT
    elif prompt_type == "Coding":
        return system_prompts.CODING_AGENT_PROMPT
    elif prompt_type == "Custom":
        return custom_prompt or ""
    else:
        return system_prompts.DEFAULT_SYSTEM_PROMPT

def create_agent_instance(
    agent_type: str,
    tools: List[Any],
    memory_type: str,
    system_prompt: str,
    name: str,
    agent_config: Dict[str, Any]
):
    """Create an agent instance based on configuration."""
    # Get agent class
    agent_class = get_agent_class(agent_type)
    
    # Create memory if specified
    memory = None
    if memory_type != "None":
        memory_class = get_memory_class(memory_type)
        memory = memory_class(namespace=name.lower().replace(" ", "_"))
    
    # Create agent
    if agent_type == "Tool-Using Agent" and not tools:
        # Tool-Using Agent requires at least one tool
        st.error("Tool-Using Agent requires at least one tool.")
        return None
    
    try:
        if agent_type == "Reasoning Agent":
            agent = agent_class(
                tools=tools,
                memory=memory,
                system_prompt=system_prompt,
                reasoning_steps=int(agent_config.get("reasoning_steps", 3)),
                model=agent_config.get("model", "claude-3-7-sonnet"),
                temperature=float(agent_config.get("temperature", 0.7)),
                max_tokens=int(agent_config.get("max_tokens", 4000)),
                name=name
            )
        elif agent_type == "Tool-Using Agent":
            agent = agent_class(
                tools=tools,
                memory=memory,
                system_prompt=system_prompt,
                max_tool_iterations=int(agent_config.get("max_tool_iterations", 5)),
                model=agent_config.get("model", "claude-3-7-sonnet"),
                temperature=float(agent_config.get("temperature", 0.7)),
                max_tokens=int(agent_config.get("max_tokens", 4000)),
                name=name
            )
        else:  # Base Agent
            agent = agent_class(
                tools=tools,
                memory=memory,
                system_prompt=system_prompt,
                model=agent_config.get("model", "claude-3-7-sonnet"),
                temperature=float(agent_config.get("temperature", 0.7)),
                max_tokens=int(agent_config.get("max_tokens", 4000)),
                name=name
            )
        
        return agent
    except Exception as e:
        st.error(f"Error creating agent: {str(e)}")
        logger.error(f"Error creating agent: {str(e)}")
        return None

def format_chat_message(message: Dict[str, str]):
    """Format a chat message for display."""
    role = message["role"]
    content = message["content"]
    
    if role == "user":
        return f"**You:** {content}"
    elif role == "assistant":
        return f"**Assistant:** {content}"
    elif role == "system":
        return f"**System:** {content}"
    else:
        return f"**{role.capitalize()}:** {content}"

# Create temp directory for file operations
os.makedirs("temp", exist_ok=True)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'tools' not in st.session_state:
    st.session_state.tools = {
        "web_search": {
            "enabled": False,
            "instance": None
        },
        "code_execution": {
            "enabled": False,
            "instance": None
        },
        "file_operations": {
            "enabled": False,
            "instance": None
        }
    }

# Define tabs
tab1, tab2, tab3 = st.tabs(["Chat", "Agent Configuration", "System Settings"])

# Tab 2: Agent Configuration
with tab2:
    st.header("Agent Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Agent configuration
        st.subheader("Agent Settings")
        
        agent_type = st.selectbox(
            "Agent Type",
            ["Base Agent", "Reasoning Agent", "Tool-Using Agent"],
            index=1  # Default to Reasoning Agent
        )
        
        agent_name = st.text_input("Agent Name", value="MyAgent")
        
        # Model settings
        model_options = ["claude-3-7-sonnet", "claude-3-opus", "claude-3-5-sonnet", "claude-3-5-haiku"]
        model = st.selectbox("Model", model_options, index=0)
        
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        
        max_tokens = st.number_input("Max Tokens", min_value=100, max_value=10000, value=4000, step=100)
        
        # Agent-specific settings
        if agent_type == "Reasoning Agent":
            reasoning_steps = st.number_input("Reasoning Steps", min_value=1, max_value=10, value=3, step=1)
            agent_config = {"reasoning_steps": reasoning_steps}
        elif agent_type == "Tool-Using Agent":
            max_tool_iterations = st.number_input("Max Tool Iterations", min_value=1, max_value=10, value=5, step=1)
            agent_config = {"max_tool_iterations": max_tool_iterations}
        else:
            agent_config = {}
        
        # Add common settings
        agent_config.update({
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens
        })
        
        # System prompt
        st.subheader("System Prompt")
        
        prompt_type = st.radio(
            "Prompt Type",
            ["Default", "Reasoning", "Tool-Using", "Research", "Creative", "Coding", "Custom"],
            index=1  # Default to Reasoning
        )
        
        if prompt_type == "Custom":
            custom_prompt = st.text_area("Custom System Prompt", height=200)
        else:
            custom_prompt = None
        
        system_prompt = get_system_prompt(prompt_type, custom_prompt)
        
        with st.expander("View System Prompt"):
            st.write(system_prompt)
    
    with col2:
        # Tool configuration
        st.subheader("Tools")
        
        # Web Search Tool
        web_search_enabled = st.checkbox("Web Search Tool", value=st.session_state.tools["web_search"]["enabled"])
        if web_search_enabled:
            with st.expander("Web Search Settings"):
                search_engine = st.selectbox("Search Engine", ["google", "bing"], index=0)
                max_results = st.slider("Max Results", min_value=1, max_value=10, value=5, step=1)
                
                if st.session_state.tools["web_search"]["instance"] is None or web_search_enabled != st.session_state.tools["web_search"]["enabled"]:
                    st.session_state.tools["web_search"]["instance"] = WebSearchTool(
                        search_engine=search_engine,
                        max_results=max_results
                    )
                    st.session_state.tools["web_search"]["enabled"] = web_search_enabled
        
        # Code Execution Tool
        code_execution_enabled = st.checkbox("Code Execution Tool", value=st.session_state.tools["code_execution"]["enabled"])
        if code_execution_enabled:
            with st.expander("Code Execution Settings"):
                timeout = st.slider("Timeout (seconds)", min_value=1.0, max_value=30.0, value=5.0, step=1.0)
                allow_plots = st.checkbox("Allow Plots", value=True)
                
                if st.session_state.tools["code_execution"]["instance"] is None or code_execution_enabled != st.session_state.tools["code_execution"]["enabled"]:
                    st.session_state.tools["code_execution"]["instance"] = CodeExecutionTool(
                        timeout=timeout,
                        allow_plots=allow_plots
                    )
                    st.session_state.tools["code_execution"]["enabled"] = code_execution_enabled
        
        # File Operations Tool
        file_operations_enabled = st.checkbox("File Operations Tool", value=st.session_state.tools["file_operations"]["enabled"])
        if file_operations_enabled:
            with st.expander("File Operations Settings"):
                default_directory = st.text_input("Default Directory", value="temp")
                
                if st.session_state.tools["file_operations"]["instance"] is None or file_operations_enabled != st.session_state.tools["file_operations"]["enabled"]:
                    st.session_state.tools["file_operations"]["instance"] = FileOperationsTool(
                        allowed_directories=[default_directory],
                        default_directory=default_directory
                    )
                    st.session_state.tools["file_operations"]["enabled"] = file_operations_enabled
        
        # Memory configuration
        st.subheader("Memory")
        
        memory_type = st.radio(
            "Memory Type",
            ["None", "Working Memory", "Vector Store"],
            index=1  # Default to Working Memory
        )
    
    # Create agent button
    if st.button("Create Agent"):
        # Collect tools
        tools = []
        if web_search_enabled and st.session_state.tools["web_search"]["instance"]:
            tools.append(st.session_state.tools["web_search"]["instance"])
        
        if code_execution_enabled and st.session_state.tools["code_execution"]["instance"]:
            tools.append(st.session_state.tools["code_execution"]["instance"])
        
        if file_operations_enabled and st.session_state.tools["file_operations"]["instance"]:
            tools.append(st.session_state.tools["file_operations"]["instance"])
        
        # Create agent
        agent = create_agent_instance(
            agent_type=agent_type,
            tools=tools,
            memory_type=memory_type,
            system_prompt=system_prompt,
            name=agent_name,
            agent_config=agent_config
        )
        
        if agent:
            st.session_state.agent = agent
            st.session_state.messages = []
            
            # Add system message
            st.session_state.messages.append({
                "role": "system",
                "content": f"Agent '{agent_name}' created successfully."
            })
            
            st.success(f"Agent '{agent_name}' created successfully.")
        else:
            st.error("Failed to create agent. Check configuration and try again.")

# Tab 3: System Settings
with tab3:
    st.header("System Settings")
    
    # API Key configuration
    st.subheader("API Keys")
    
    anthropic_api_key = st.text_input("Anthropic API Key", type="password")
    
    if st.button("Save API Keys"):
        if anthropic_api_key:
            # Update configuration
            update_config({
                "llm": {
                    "api_key": anthropic_api_key
                }
            })
            st.success("API keys saved successfully.")
    
    # Logging settings
    st.subheader("Logging")
    
    log_level = st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"], index=1)
    
    if st.button("Apply Logging Settings"):
        # Update logging
        setup_logging(level=log_level, log_to_console=True)
        st.success(f"Logging level set to {log_level}")
    
    # Export/Import configuration
    st.subheader("Export/Import Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Configuration"):
            # Get current configuration
            current_config = load_config()
            
            # Convert to JSON
            config_json = json.dumps(current_config, indent=2)
            
            # Offer download
            st.download_button(
                label="Download Configuration",
                data=config_json,
                file_name="agentic_ai_config.json",
                mime="application/json"
            )
    
    with col2:
        uploaded_file = st.file_uploader("Import Configuration", type=["json", "yaml", "yml"])
        
        if uploaded_file is not None:
            try:
                # Read file content
                content = uploaded_file.read()
                
                # Parse based on file type
                if uploaded_file.name.endswith(('.yaml', '.yml')):
                    import yaml
                    config_data = yaml.safe_load(content)
                else:
                    config_data = json.loads(content)
                
                # Update configuration
                update_config(config_data)
                
                st.success("Configuration imported successfully.")
            except Exception as e:
                st.error(f"Error importing configuration: {str(e)}")

# Tab 1: Chat Interface
with tab1:
    st.header("Chat with Agent")
    
    # Display agent information if available
    if st.session_state.agent:
        agent_info = f"Active Agent: **{st.session_state.agent.name}** ({st.session_state.agent.__class__.__name__})"
        
        # Add tool information
        tools_list = [tool.name for tool in st.session_state.agent.tools]
        if tools_list:
            agent_info += f" | Tools: {', '.join(tools_list)}"
        
        st.markdown(agent_info)
    else:
        st.warning("No agent is currently active. Please configure and create an agent in the 'Agent Configuration' tab.")
    
    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            st.markdown(format_chat_message(message))
    
    # Chat input
    if st.session_state.agent:
        user_input = st.chat_input("Type your message here...")
        
        if user_input:
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Display user message
            with chat_container:
                st.markdown(format_chat_message({"role": "user", "content": user_input}))
            
            # Process with agent
            try:
                with st.spinner("Thinking..."):
                    start_time = time.time()
                    
                    response = st.session_state.agent.run(user_input)
                    
                    elapsed_time = time.time() - start_time
                    logger.info(f"Agent response generated in {elapsed_time:.2f} seconds")
                
                # Add assistant message to history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Display assistant message
                with chat_container:
                    st.markdown(format_chat_message({"role": "assistant", "content": response}))
            
            except Exception as e:
                error_message = f"Error processing message: {str(e)}"
                logger.error(error_message)
                
                # Add error message to history
                st.session_state.messages.append({"role": "system", "content": error_message})
                
                # Display error message
                with chat_container:
                    st.markdown(format_chat_message({"role": "system", "content": error_message}))
    
    # Reset chat button
    if st.button("Reset Chat"):
        # Keep the agent but clear the chat history
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "system",
            "content": "Chat history has been reset."
        })
        
        # Rerun to update the UI
        st.rerun()
