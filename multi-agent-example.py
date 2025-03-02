"""
Multi-Agent System Example

This script demonstrates how to create a multi-agent system where multiple
specialized agents collaborate to solve complex problems.
"""

import os
import sys
import logging
from typing import Dict, List, Any, Optional

# Add parent directory to path to allow importing the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the framework
from agentic_ai import ReasoningAgent, ToolUsingAgent
from agentic_ai import WebSearchTool, CodeExecutionTool, FileOperationsTool
from agentic_ai import WorkingMemory, VectorStore
from agentic_ai.prompts import system_prompts
from agentic_ai.utils.logger import setup_logging

# Set up logging
setup_logging(level=logging.INFO, log_to_console=True)

class MultiAgentSystem:
    """
    A simple multi-agent system with specialized agents that work together.
    
    This class demonstrates how multiple agents with different capabilities
    can collaborate to solve complex problems that require different skills.
    """
    
    def __init__(self):
        """Initialize the multi-agent system."""
        # Create shared memory for collaboration
        self.shared_memory = VectorStore(namespace="shared")
        
        # Initialize tools
        self.web_search = WebSearchTool()
        self.code_execution = CodeExecutionTool()
        self.file_operations = FileOperationsTool()
        
        # Create specialized agents
        self.research_agent = self._create_research_agent()
        self.coding_agent = self._create_coding_agent()
        self.reasoning_agent = self._create_reasoning_agent()
        
        print("Multi-Agent System initialized with 3 specialized agents:")
        print("- Research Agent: Gathers and analyzes information")
        print("- Coding Agent: Writes and executes code")
        print("- Reasoning Agent: Synthesizes information and makes decisions")
    
    def _create_research_agent(self) -> ReasoningAgent:
        """Create a specialized research agent."""
        # Individual memory for the agent
        memory = WorkingMemory(namespace="research")
        
        # Create the agent
        agent = ReasoningAgent(
            tools=[self.web_search, self.file_operations],
            memory=memory,
            system_prompt=system_prompts.RESEARCH_AGENT_PROMPT,
            reasoning_steps=3,
            name="ResearchAgent"
        )
        
        return agent
    
    def _create_coding_agent(self) -> ToolUsingAgent:
        """Create a specialized coding agent."""
        # Individual memory for the agent
        memory = WorkingMemory(namespace="coding")
        
        # Create the agent
        agent = ToolUsingAgent(
            tools=[self.code_execution, self.file_operations],
            memory=memory,
            system_prompt=system_prompts.CODING_AGENT_PROMPT,
            max_tool_iterations=5,
            name="CodingAgent"
        )
        
        return agent
    
    def _create_reasoning_agent(self) -> ReasoningAgent:
        """Create a specialized reasoning agent."""
        # Individual memory for the agent
        memory = WorkingMemory(namespace="reasoning")
        
        # Create the agent
        agent = ReasoningAgent(
            tools=[self.file_operations],
            memory=memory,
            system_prompt=system_prompts.REASONING_SYSTEM_PROMPT,
            reasoning_steps=5,
            name="ReasoningAgent"
        )
        
        return agent
    
    def solve_problem(self, problem_statement: str) -> Dict[str, Any]:
        """
        Solve a complex problem using the multi-agent system.
        
        Args:
            problem_statement: Description of the problem to solve
            
        Returns:
            Dictionary with solution and intermediate steps
        """
        print(f"\nSolving problem: {problem_statement}\n")
        
        # Step 1: Research phase - gather information
        print("Step 1: Research Phase (Research Agent)")
        research_prompt = (
            f"Research information relevant to this problem: {problem_statement}\n\n"
            "Focus on gathering facts, context, and relevant data. "
            "Save your findings to a file named 'research_findings.txt'."
        )
        research_response = self.research_agent.run(research_prompt)
        print(f"Research Agent response: {research_response[:300]}...\n")
        
        # Add research findings to shared memory
        self.shared_memory.add({
            "role": "research_agent",
            "content": research_response,
            "phase": "research"
        })
        
        # Step 2: Code development - implement solution
        print("Step 2: Coding Phase (Coding Agent)")
        coding_prompt = (
            f"Based on the problem: {problem_statement}\n\n"
            "Implement a solution in code. Use the research findings in 'research_findings.txt' "
            "if they're available. Save your solution to 'solution.py'."
        )
        coding_response = self.coding_agent.run(coding_prompt)
        print(f"Coding Agent response: {coding_response[:300]}...\n")
        
        # Add code solution to shared memory
        self.shared_memory.add({
            "role": "coding_agent",
            "content": coding_response,
            "phase": "coding"
        })
        
        # Step 3: Synthesis and evaluation - bring it all together
        print("Step 3: Synthesis Phase (Reasoning Agent)")
        reasoning_prompt = (
            f"Evaluate the overall solution to this problem: {problem_statement}\n\n"
            "Consider the research findings in 'research_findings.txt' and the "
            "code solution in 'solution.py'. Provide a final assessment and summary."
        )
        reasoning_response = self.reasoning_agent.run(reasoning_prompt)
        print(f"Reasoning Agent response: {reasoning_response[:300]}...\n")
        
        # Add reasoning to shared memory
        self.shared_memory.add({
            "role": "reasoning_agent",
            "content": reasoning_response,
            "phase": "synthesis"
        })
        
        # Compile the complete solution
        solution = {
            "problem_statement": problem_statement,
            "research_phase": research_response,
            "coding_phase": coding_response,
            "synthesis_phase": reasoning_response,
            "final_solution": reasoning_response  # The reasoning agent provides the final summary
        }
        
        print("\nProblem solved! Full solution available in the returned dictionary.")
        return solution

def main():
    """Run the multi-agent system example."""
    print("Agentic AI Framework - Multi-Agent System Example")
    print("================================================")
    
    # Create the multi-agent system
    mas = MultiAgentSystem()
    
    # Define a complex problem
    problem = """
    Create a data visualization tool that analyzes historical weather data
    to identify trends related to climate change. Focus on:
    1. Temperature changes over the last century
    2. Precipitation patterns
    3. Extreme weather events frequency
    
    The solution should include data gathering, analysis, and visualization components.
    """
    
    # Solve the problem
    solution = mas.solve_problem(problem)
    
    # Print summary
    print("\nSolution Summary:")
    print("----------------")
    
    # Extract the final paragraph of the reasoning phase as a summary
    summary_lines = solution["synthesis_phase"].strip().split("\n\n")
    if summary_lines:
        summary = summary_lines[-1]
        print(summary)
    
    print("\nExample complete! You've seen how multiple specialized agents can collaborate to solve a complex problem.")

if __name__ == "__main__":
    main()
