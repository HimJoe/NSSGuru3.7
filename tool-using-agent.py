"""
Tool-Using Agent Implementation

This module defines the ToolUsingAgent class, which specializes in effectively
utilizing a diverse set of tools to solve complex problems.
"""

from typing import Dict, List, Optional, Any, Union
import json
import re

from .base_agent import BaseAgent
from ..prompts import system_prompts
from ..memory import BaseMemory
from ..tools import BaseTool

class ToolUsingAgent(BaseAgent):
    """
    An agent specialized in effectively using tools to solve complex problems.
    
    This agent has enhanced capabilities for tool selection, parameter validation,
    and tool result interpretation.
    """
    
    def __init__(
        self,
        tools: List[BaseTool],  # Tools are required for this agent type
        memory: Optional[BaseMemory] = None,
        system_prompt: str = system_prompts.TOOL_USING_SYSTEM_PROMPT,
        model: str = "claude-3-7-sonnet",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        max_tool_iterations: int = 5,
        name: str = "ToolUsingAgent",
        metadata: Optional[Dict[str, Any]] = None,
        auto_validate_params: bool = True,
    ):
        """
        Initialize a tool-using agent.
        
        Args:
            tools: List of tools the agent can use (required)
            memory: Memory system for the agent
            system_prompt: System prompt that defines agent behavior
            model: LLM model to use for this agent
            temperature: Sampling temperature for the model
            max_tokens: Maximum tokens to generate in responses
            max_tool_iterations: Maximum number of tool calls in a single turn
            name: Agent name
            metadata: Additional agent metadata
            auto_validate_params: Whether to validate tool parameters before execution
        """
        if not tools:
            raise ValueError("ToolUsingAgent requires at least one tool")
            
        super().__init__(
            tools=tools,
            memory=memory,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            name=name,
            metadata=metadata
        )
        self.max_tool_iterations = max_tool_iterations
        self.auto_validate_params = auto_validate_params
        self.logger.info(f"Initialized {self.name} with {len(tools)} tools")
    
    def _format_tool_descriptions(self) -> str:
        """
        Format detailed tool descriptions with parameter information.
        
        Overrides the base method to include parameter details.
        """
        tool_descriptions = []
        
        for tool in self.tools:
            # Get parameter information
            params_info = []
            for param_name, param_info in tool.params_schema.items():
                required = "required" if param_name in tool.required_params else "optional"
                desc = param_info.get("description", "No description")
                param_type = param_info.get("type", "string")
                
                params_info.append(f"- {param_name} ({param_type}, {required}): {desc}")
            
            # Format the complete tool description
            tool_desc = (
                f"Tool: {tool.name}\n"
                f"Description: {tool.description}\n"
                f"Parameters:\n" + "\n".join(params_info) + "\n"
                f"Returns: {tool.returns_description}\n"
                f"Example usage: {tool.usage}"
            )
            
            tool_descriptions.append(tool_desc)
        
        return "\n\n".join(tool_descriptions)
    
    def _validate_tool_params(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, str]:
        """
        Validate parameters for a specific tool.
        
        Args:
            tool_name: Name of the tool
            params: Parameters to validate
            
        Returns:
            Dictionary of validation errors, empty if valid
        """
        # Find the tool
        tool = next((t for t in self.tools if t.name == tool_name), None)
        if not tool:
            return {"error": f"Tool '{tool_name}' not found"}
        
        errors = {}
        
        # Check for required parameters
        for param_name in tool.required_params:
            if param_name not in params or params[param_name] is None:
                errors[param_name] = f"Required parameter '{param_name}' is missing"
        
        # Validate parameter types
        for param_name, param_value in params.items():
            if param_name in tool.params_schema:
                schema = tool.params_schema[param_name]
                param_type = schema.get("type", "string")
                
                # Basic type validation
                if param_type == "string" and not isinstance(param_value, str):
                    errors[param_name] = f"Parameter '{param_name}' must be a string"
                elif param_type == "integer" and not (isinstance(param_value, int) or (isinstance(param_value, str) and param_value.isdigit())):
                    errors[param_name] = f"Parameter '{param_name}' must be an integer"
                elif param_type == "boolean" and not isinstance(param_value, bool) and param_value not in ["true", "false", "True", "False"]:
                    errors[param_name] = f"Parameter '{param_name}' must be a boolean"
                elif param_type == "array" and not isinstance(param_value, list):
                    errors[param_name] = f"Parameter '{param_name}' must be an array"
                
                # Check enumerated values if specified
                if "enum" in schema and param_value not in schema["enum"]:
                    errors[param_name] = f"Parameter '{param_name}' must be one of: {', '.join(schema['enum'])}"
        
        return errors
    
    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse tool calls using a more sophisticated JSON extraction method.
        
        This overrides the base method with a more robust implementation.
        
        Args:
            response: Raw model response
            
        Returns:
            List of parsed tool calls with their parameters
        """
        tool_calls = []
        
        # Look for JSON-formatted tool calls
        json_pattern = r'```json\s*(.*?)\s*```'
        tool_call_pattern = r'TOOL_CALL:\s*(\{.*?\})'
        
        # Try to find JSON blocks first
        json_matches = re.findall(json_pattern, response, re.DOTALL)
        for json_str in json_matches:
            try:
                data = json.loads(json_str)
                if isinstance(data, dict) and "tool_name" in data and "parameters" in data:
                    tool_calls.append(data)
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "tool_name" in item and "parameters" in item:
                            tool_calls.append(item)
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse JSON: {json_str}")
        
        # If no JSON blocks found, look for TOOL_CALL markers
        if not tool_calls:
            tool_matches = re.findall(tool_call_pattern, response, re.DOTALL)
            for tool_str in tool_matches:
                try:
                    data = json.loads(tool_str)
                    if "tool_name" in data and "parameters" in data:
                        tool_calls.append(data)
                except json.JSONDecodeError:
                    self.logger.warning(f"Failed to parse tool call: {tool_str}")
        
        # Fallback: Basic parsing for format "Tool: X, Parameters: Y"
        if not tool_calls:
            simple_pattern = r'Tool:\s*([^\n,]+)(?:,|\n).*?Parameters:\s*(\{.*?\})'
            simple_matches = re.findall(simple_pattern, response, re.DOTALL)
            for tool_name, params_str in simple_matches:
                try:
                    params = json.loads(params_str)
                    tool_calls.append({
                        "tool_name": tool_name.strip(),
                        "parameters": params
                    })
                except json.JSONDecodeError:
                    self.logger.warning(f"Failed to parse parameters: {params_str}")
        
        return tool_calls
    
    def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute tool calls with parameter validation.
        
        Overrides the base method to add parameter validation.
        
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
            
            # Validate parameters if enabled
            if self.auto_validate_params:
                validation_errors = self._validate_tool_params(tool_name, params)
                if validation_errors:
                    error_msg = f"Parameter validation failed: {json.dumps(validation_errors)}"
                    self.logger.warning(error_msg)
                    results.append({
                        "tool_name": tool_name,
                        "error": error_msg,
                        "validation_errors": validation_errors,
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
        Process user input with multiple tool-calling iterations.
        
        This overrides the base run method to allow for a multi-step
        conversation with tools, where the agent can make multiple
        tool calls in sequence to solve a problem.
        
        Args:
            user_input: User's input message
            
        Returns:
            Final agent response after tool use
        """
        self.logger.info(f"Running tool-using agent with input: {user_input[:50]}...")
        
        # Store user input in memory if available
        if self.memory:
            self.memory.add({"role": "user", "content": user_input})
        
        # Build the initial prompt
        prompt = self._build_prompt(user_input)
        current_response = ""
        
        # Track conversation history for this run
        conversation_history = []
        
        # Iterative tool calling loop
        for iteration in range(self.max_tool_iterations):
            # Get response from LLM
            current_response = self._call_llm(prompt)
            conversation_history.append({
                "role": "assistant",
                "content": current_response
            })
            
            # Parse any tool calls
            tool_calls = self._parse_tool_calls(current_response)
            
            # If no tool calls or reached max iterations, break the loop
            if not tool_calls or iteration == self.max_tool_iterations - 1:
                break
            
            # Execute tool calls
            tool_results = self._execute_tool_calls(tool_calls)
            
            # Format tool results
            tool_results_text = "\n\n".join([
                f"Tool: {result['tool_name']}\n"
                f"Success: {result['success']}\n"
                f"Result: {result.get('result', '')}\n"
                f"Error: {result.get('error', '')}"
                for result in tool_results
            ])
            
            # Add tool results to conversation history
            conversation_history.append({
                "role": "tool",
                "content": tool_results_text
            })
            
            # Update prompt with conversation history
            conversation_text = "\n\n".join([
                f"{'User' if item['role'] == 'user' else 'Assistant' if item['role'] == 'assistant' else 'Tool Results'}: {item['content']}"
                for item in conversation_history
            ])
            
            prompt = f"{self.system_prompt}\n\n{conversation_text}\n\nAssistant:"
        
        # Store final response in memory
        if self.memory and current_response:
            self.memory.add({"role": "assistant", "content": current_response})
        
        return current_response