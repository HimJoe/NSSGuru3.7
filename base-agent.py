"""
Base Agent Implementation

This module defines the BaseAgent class, which serves as the foundation
for all agent types in the framework. It handles core functionality like
tool integration, memory management, and interaction patterns.
"""

import logging
from typing import Dict, List, Optional, Any, Union

from ..memory import BaseMemory
from ..tools import BaseTool
from ..prompts import system_prompts
from ..utils.logger import get_logger

class BaseAgent:
    """
    Base class for all agents in the framework.
    
    Handles core functionality including tool management, memory integration,
    and basic interaction patterns.
    """
    
    def __init__(
        self,
        tools: Optional[List[BaseTool]] = None,
        memory: Optional[BaseMemory] = None,
        system_prompt: str = system_prompts.DEFAULT_SYSTEM_PROMPT,
        model: str = "claude-3-7-sonnet",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        name: str = "Agent",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a base agent with tools, memory, and configuration.
        
        Args:
            tools: List of tools the agent can use
            memory: Memory system for the agent
            system_prompt: System prompt that defines agent behavior
            model: LLM model to use for this agent
            temperature: Sampling temperature for the model
            max_tokens: Maximum tokens to generate in responses
            name: Agent name (used in logging and multi-agent systems)
            metadata: Additional agent metadata
        """
        self.tools = tools or []
        self.memory = memory
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.name = name
        self.metadata = metadata or {}
        self.logger = get_logger(f"agent.{name.lower()}")
        
        # Register tools with the agent
        self._register_tools()
        
        self.logger.info(f"Initialized {self.name} with {len(self.tools)} tools")
    
    def _register_tools(self) -> None:
        """Register and validate all tools with the agent."""
        for tool in self.tools:
            self.logger.debug(f"Registered tool: {tool.name}")
    
    def _format_tool_descriptions(self) -> str:
        """Format tool descriptions for inclusion in the prompt."""
        if not self.tools:
            return "No tools available."
        
        tool_descriptions = []
        for tool in self.tools:
            tool_descriptions.append(f"Tool: {tool.name}\nDescription: {tool.description}\nUsage: {tool.usage}")
        
        return "\n\n".join(tool_descriptions)
    
    def _build_prompt(self, user_input: str) -> str:
        """
        Build the complete prompt for the model.
        
        Args:
            user_input: The user's input message
            
        Returns:
            Complete formatted prompt for the model
        """
        # Get relevant context from memory if available
        context = ""
        if self.memory:
            context = self.memory.retrieve(user_input)
        
        # Format tool descriptions
        tool_descriptions = self._format_tool_descriptions()
        
        # Construct the full prompt
        prompt = f"{self.system_prompt}\n\n"
        
        if context:
            prompt += f"Relevant context from memory:\n{context}\n\n"
        
        if self.tools:
            prompt += f"Available tools:\n{tool_descriptions}\n\n"
        
        prompt += f"User: {user_input}\n\nAssistant:"
        
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call the underlying LLM with the given prompt.
        
        This is a placeholder method that should be implemented by the actual
        integration with your LLM API of choice (OpenAI, Anthropic, etc.)
        
        Args:
            prompt: The fully formatted prompt
            
        Returns:
            Model response as a string
        """
        # In a real implementation, this would make an API call to the LLM
        self.logger.info(f"Calling LLM with model {self.model}")
        
        # Placeholder for API call
        # response = llm_client.complete(
        #     model=self.model,
        #     prompt=prompt,
        #     temperature=self.temperature,
        #     max_tokens=self.max_tokens
        # )
        
        # For now, return a placeholder
        return "This is a placeholder response. Implement actual LLM API call here."
    
    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse tool calls from the model's response.
        
        Args:
            response: Raw model response
            
        Returns:
            List of parsed tool calls with their parameters
        """
        # This is a simplified placeholder
        # In a real implementation, you would parse JSON or special markers
        tool_calls = []
        
        # Example parsing logic (would be more sophisticated in practice)
        if "TOOL_CALL" in response:
            # Simple parsing logic - in reality, use regex or proper JSON parsing
            parts = response.split("TOOL_CALL:")
            for part in parts[1:]:
                tool_name = part.split("[")[1].split("]")[0]
                params_text = part.split("{")[1].split("}")[0]
                params = {}
                for param_pair in params_text.split(","):
                    if ":" in param_pair:
                        k, v = param_pair.split(":", 1)
                        params[k.strip()] = v.strip().strip('"\'')
                
                tool_calls.append({
                    "tool_name": tool_name,
                    "parameters": params
                })
        
        return tool_calls
    
    def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute a list of tool calls and return their results.
        
        Args:
            tool_calls: List of parsed tool calls with parameters
            
        Returns:
            List of tool execution results
        """
        results = []
        
        for call in tool_calls:
            tool_name = call["tool_name"]
            params = call["parameters"]
            
            # Find the matching tool
            tool = next((t for t in self.tools if t.name == tool_name), None)
            
            if not tool:
                error_msg = f"Tool '{tool_name}' not found"
                self.logger.warning(error_msg)
                results.append({
                    "tool_name": tool_name,
                    "error": error_msg,
                    "success": False
                })
                continue
            
            try:
                # Execute the tool
                self.logger.info(f"Executing tool: {tool_name}")
                result = tool.execute(**params)
                results.append({
                    "tool_name": tool_name,
                    "result": result,
                    "success": True
                })
            except Exception as e:
                self.logger.error(f"Error executing tool {tool_name}: {str(e)}")
                results.append({
                    "tool_name": tool_name,
                    "error": str(e),
                    "success": False
                })
        
        return results
    
    def run(self, user_input: str) -> str:
        """
        Process a user input and generate a response, potentially using tools.
        
        This is the main entry point for agent interaction.
        
        Args:
            user_input: User's input message
            
        Returns:
            Agent response
        """
        self.logger.info(f"Running agent with input: {user_input[:50]}...")
        
        # Store user input in memory if available
        if self.memory:
            self.memory.add({"role": "user", "content": user_input})
        
        # Build the prompt
        prompt = self._build_prompt(user_input)
        
        # Get initial response from LLM
        response = self._call_llm(prompt)
        
        # Parse any tool calls
        tool_calls = self._parse_tool_calls(response)
        
        # If no tool calls, return the response
        if not tool_calls:
            if self.memory:
                self.memory.add({"role": "assistant", "content": response})
            return response
        
        # Execute tool calls
        tool_results = self._execute_tool_calls(tool_calls)
        
        # Augment the prompt with tool results and get final response
        tool_results_text = "\n\n".join([
            f"Tool: {result['tool_name']}\n"
            f"Success: {result['success']}\n"
            f"Result: {result.get('result', '')}\n"
            f"Error: {result.get('error', '')}"
            for result in tool_results
        ])
        
        augmented_prompt = f"{prompt}\n\nTool Results:\n{tool_results_text}\n\nFinal Response:"
        final_response = self._call_llm(augmented_prompt)
        
        # Store final response in memory
        if self.memory:
            self.memory.add({"role": "assistant", "content": final_response})
        
        return final_response
    
    def reset(self) -> None:
        """Reset the agent's state, including working memory."""
        if self.memory:
            self.memory.clear()
        self.logger.info(f"Reset agent {self.name}")
