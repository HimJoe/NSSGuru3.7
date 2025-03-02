"""
Reasoning Agent Implementation

This module defines the ReasoningAgent class, which extends BaseAgent with
advanced reasoning capabilities through Chain of Thought (CoT) prompting
and deliberative problem-solving techniques.
"""

from typing import Dict, List, Optional, Any

from .base_agent import BaseAgent
from ..prompts import system_prompts
from ..memory import BaseMemory
from ..tools import BaseTool

class ReasoningAgent(BaseAgent):
    """
    An agent with enhanced reasoning capabilities using chain-of-thought techniques.
    
    This agent explicitly generates intermediate reasoning steps before reaching
    conclusions, leading to more reliable and explainable outcomes.
    """
    
    def __init__(
        self,
        tools: Optional[List[BaseTool]] = None,
        memory: Optional[BaseMemory] = None,
        system_prompt: str = system_prompts.REASONING_SYSTEM_PROMPT,
        reasoning_steps: int = 3,
        model: str = "claude-3-7-sonnet",
        temperature: float = 0.7,
        max_tokens: int = 8000,  # Increased for reasoning
        name: str = "ReasoningAgent",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a reasoning agent.
        
        Args:
            tools: List of tools the agent can use
            memory: Memory system for the agent
            system_prompt: System prompt that defines agent behavior
            reasoning_steps: Minimum number of reasoning steps to use
            model: LLM model to use for this agent
            temperature: Sampling temperature for the model
            max_tokens: Maximum tokens to generate in responses
            name: Agent name
            metadata: Additional agent metadata
        """
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
        self.reasoning_steps = reasoning_steps
        self.logger.info(f"Initialized {self.name} with {self.reasoning_steps} reasoning steps")
    
    def _build_prompt(self, user_input: str) -> str:
        """
        Build a prompt that explicitly encourages reasoning.
        
        Overrides the base method to add reasoning-specific instructions.
        
        Args:
            user_input: The user's input message
            
        Returns:
            Complete formatted prompt for the model
        """
        # Get the base prompt first
        prompt = super()._build_prompt(user_input)
        
        # Add reasoning-specific instructions
        reasoning_instructions = f"""
Before responding, break down your thinking into at least {self.reasoning_steps} explicit steps.
For each step:
1. Clearly state what you're trying to determine
2. Describe your approach to this step
3. Show your working and intermediate conclusions

Only after these reasoning steps, provide your final answer.

Begin reasoning:
        """
        
        # Insert reasoning instructions before the final "Assistant:" marker
        parts = prompt.rsplit("Assistant:", 1)
        if len(parts) == 2:
            prompt = f"{parts[0]}Assistant: {reasoning_instructions}{parts[1]}"
        
        return prompt
    
    def _extract_final_answer(self, response: str) -> str:
        """
        Extract the final answer from a reasoning-based response.
        
        Args:
            response: The full response with reasoning steps
            
        Returns:
            Just the final answer portion
        """
        # Look for markers that typically indicate the final answer
        final_answer_markers = [
            "Final answer:",
            "In conclusion,",
            "Therefore,",
            "To summarize,",
            "Final response:"
        ]
        
        for marker in final_answer_markers:
            if marker in response:
                parts = response.split(marker, 1)
                if len(parts) == 2:
                    return f"{marker}{parts[1]}"
        
        # If no markers found, return the last 30% of the response
        # This is a heuristic approach
        split_point = int(len(response) * 0.7)
        return response[split_point:]
    
    def run(self, user_input: str, extract_final_answer: bool = False) -> str:
        """
        Process a user input with explicit reasoning steps.
        
        Args:
            user_input: User's input message
            extract_final_answer: Whether to return only the final answer
            
        Returns:
            Full reasoning process or just the final answer
        """
        self.logger.info(f"Running reasoning agent with input: {user_input[:50]}...")
        
        # Get full response with reasoning
        full_response = super().run(user_input)
        
        # Extract final answer if requested
        if extract_final_answer:
            return self._extract_final_answer(full_response)
        
        return full_response
    
    def analyze_problem(self, problem_statement: str) -> Dict[str, Any]:
        """
        Perform a structured analysis of a problem statement.
        
        Args:
            problem_statement: The problem to analyze
            
        Returns:
            Dictionary with analysis components:
            - key_concepts: List of key concepts in the problem
            - constraints: Identified constraints
            - approaches: Possible solution approaches
            - required_info: Information needed but not provided
        """
        self.logger.info(f"Analyzing problem: {problem_statement[:50]}...")
        
        analysis_prompt = f"""
I need a structured analysis of the following problem:

{problem_statement}

Break down your analysis into these components:
1. Key concepts - What are the fundamental concepts involved?
2. Constraints - What limitations or requirements must be respected?
3. Approaches - What potential solution approaches could work?
4. Required information - What critical information is needed but not provided?
5. Assumptions - What assumptions must be made to proceed?

For each component, provide a detailed explanation.
        """
        
        # Get response with reasoning
        response = super().run(analysis_prompt)
        
        # Parse the structured analysis from the response
        # This is a simplified parser - in practice, you'd want more robust extraction
        analysis = {
            "key_concepts": [],
            "constraints": [],
            "approaches": [],
            "required_info": [],
            "assumptions": []
        }
        
        current_section = None
        for line in response.split('\n'):
            line = line.strip()
            
            # Check for section headers
            if "Key concepts" in line or "Key Concepts" in line:
                current_section = "key_concepts"
                continue
            elif "Constraints" in line:
                current_section = "constraints"
                continue
            elif "Approaches" in line:
                current_section = "approaches"
                continue
            elif "Required information" in line or "Required Information" in line:
                current_section = "required_info"
                continue
            elif "Assumptions" in line:
                current_section = "assumptions"
                continue
            
            # If we're in a section and the line has content, add it
            if current_section and line and line[0] in "-*•" or (len(line) > 2 and line[0].isdigit() and line[1] == '.'):
                item = line.strip("-*•0123456789. ")
                if item:
                    analysis[current_section].append(item)
        
        return analysis
