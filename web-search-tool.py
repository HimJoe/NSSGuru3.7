"""
Web Search Tool Implementation

This module implements a web search tool that allows agents to retrieve information
from the internet using search engine APIs.
"""

import time
import json
from typing import Dict, List, Any, Optional

from .base_tool import BaseTool, tool
from ..utils.logger import get_logger

class WebSearchTool(BaseTool):
    """
    A tool for performing web searches and retrieving relevant information.
    
    This tool interfaces with search engine APIs to allow agents to access
    up-to-date information from the internet.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        search_engine: str = "google",
        max_results: int = 5,
        timeout: float = 10.0,
        name: str = "web_search",
    ):
        """
        Initialize the web search tool.
        
        Args:
            api_key: API key for the search engine
            search_engine: Search engine to use ('google', 'bing', etc.)
            max_results: Maximum number of results to return
            timeout: Timeout for search requests in seconds
            name: Tool name
        """
        # Define parameter schema
        params_schema = {
            "query": {
                "type": "string",
                "description": "The search query to execute"
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (max 10)",
                "minimum": 1,
                "maximum": 10
            },
            "include_domains": {
                "type": "array",
                "description": "List of domains to prioritize in results",
            },
            "exclude_domains": {
                "type": "array",
                "description": "List of domains to exclude from results",
            },
            "time_period": {
                "type": "string",
                "description": "Time period for results (recent, past_year, etc.)",
                "enum": ["any", "recent", "past_day", "past_week", "past_month", "past_year"]
            }
        }
        
        super().__init__(
            name=name,
            description="Search the web for information on a given topic",
            usage=f'{name}(query="latest AI research", num_results=3, time_period="past_month")',
            params_schema=params_schema,
            required_params=["query"],
            returns_description="List of search results with title, URL, and snippet for each"
        )
        
        self.api_key = api_key
        self.search_engine = search_engine
        self.max_results = max_results
        self.timeout = timeout
        
        # Create a specific logger for this tool
        self.logger = get_logger(f"tool.{name}")
        
        self.logger.info(f"Initialized web search tool using {search_engine}")
    
    def execute(
        self,
        query: str,
        num_results: int = 5,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        time_period: str = "any"
    ) -> List[Dict[str, str]]:
        """
        Execute a web search query.
        
        Args:
            query: Search query string
            num_results: Number of results to return (capped by max_results)
            include_domains: List of domains to prioritize
            exclude_domains: List of domains to exclude
            time_period: Time period for search results
            
        Returns:
            List of search result dictionaries with title, URL, and snippet
        """
        # Cap the number of results to the maximum
        num_results = min(num_results, self.max_results)
        
        self.logger.info(f"Searching for: {query}")
        
        # In a real implementation, this would call a search API
        # For demonstration, we'll return mock results
        
        # Simulate network delay
        time.sleep(0.5)
        
        # Generate mock results
        results = []
        for i in range(num_results):
            result = {
                "title": f"Result {i+1} for '{query}'",
                "url": f"https://example.com/result-{i+1}",
                "snippet": f"This is a summary of search result {i+1} for the query '{query}'. It would contain relevant text from the webpage.",
                "published_date": "2024-01-15"
            }
            results.append(result)
        
        self.logger.info(f"Found {len(results)} results for query: {query}")
        
        return results


# Alternative using the decorator approach
@tool(
    name="web_search_simple",
    description="Simple interface to search the web for information",
    params_schema={
        "query": {
            "type": "string",
            "description": "The search query to execute"
        },
        "num_results": {
            "type": "integer",
            "description": "Number of results to return",
            "minimum": 1,
            "maximum": 5
        }
    },
    required_params=["query"]
)
def web_search_simple(query: str, num_results: int = 3) -> List[Dict[str, str]]:
    """
    A simplified web search function.
    
    Args:
        query: Search query
        num_results: Number of results to return
        
    Returns:
        List of search results
    """
    # Simplified implementation
    results = []
    for i in range(num_results):
        results.append({
            "title": f"Result {i+1} for '{query}'",
            "url": f"https://example.com/result-{i+1}",
            "snippet": f"Summary of result {i+1} for query '{query}'."
        })
    
    return results
