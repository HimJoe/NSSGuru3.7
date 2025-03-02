"""
Pytest configuration file for the Agentic AI framework.

This file contains fixtures and configuration settings for the test suite.
"""

import os
import sys
import pytest
import tempfile
import shutil
from typing import Dict, List, Any, Optional, Generator

# Add parent directory to path to allow importing the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the framework
from src.agentic_ai.utils.logger import setup_logging
from src.agentic_ai.config import reset_config

# Set up logging for tests with a lower level
setup_logging(level="ERROR", log_to_console=False)

@pytest.fixture(scope="function")
def temp_directory() -> Generator[str, None, None]:
    """
    Create a temporary directory for file operations during tests.
    
    Yields:
        Path to the temporary directory
    """
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Yield the directory path for use in tests
    yield temp_dir
    
    # Clean up after test completes
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="function")
def reset_framework_config() -> None:
    """
    Reset the framework configuration to default values.
    """
    reset_config()

@pytest.fixture(scope="function")
def mock_llm_response() -> Dict[str, Any]:
    """
    Create a mock LLM response for testing.
    
    Returns:
        Mock LLM response dictionary
    """
    return {
        "id": "mock-response-id",
        "model": "mock-model",
        "created": 1677825464,
        "content": "This is a mock LLM response for testing.",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }

@pytest.fixture(scope="function")
def sample_tools() -> List[Dict[str, Any]]:
    """
    Create sample tool definitions for testing.
    
    Returns:
        List of sample tool dictionaries
    """
    return [
        {
            "name": "test_tool1",
            "description": "A test tool for testing",
            "parameters": {
                "param1": {
                    "type": "string",
                    "description": "A string parameter"
                },
                "param2": {
                    "type": "integer",
                    "description": "An integer parameter"
                }
            },
            "required": ["param1"]
        },
        {
            "name": "test_tool2",
            "description": "Another test tool for testing",
            "parameters": {
                "param1": {
                    "type": "boolean",
                    "description": "A boolean parameter"
                }
            },
            "required": ["param1"]
        }
    ]

@pytest.fixture(scope="function")
def sample_memory_items() -> List[Dict[str, Any]]:
    """
    Create sample memory items for testing.
    
    Returns:
        List of sample memory item dictionaries
    """
    return [
        {
            "role": "user",
            "content": "This is a user message for testing.",
            "timestamp": 1677825464
        },
        {
            "role": "assistant",
            "content": "This is an assistant response for testing.",
            "timestamp": 1677825470
        },
        {
            "role": "user",
            "content": "Another user message for testing.",
            "timestamp": 1677825480
        }
    ]

@pytest.fixture(scope="session")
def mock_api_key() -> str:
    """
    Create a mock API key for testing.
    
    Returns:
        Mock API key string
    """
    return "test-api-key-1234567890"

class MockResponse:
    """Mock response class for testing HTTP requests."""
    
    def __init__(self, status_code: int = 200, json_data: Optional[Dict[str, Any]] = None, text: Optional[str] = None):
        """
        Initialize a mock response.
        
        Args:
            status_code: HTTP status code
            json_data: JSON data for the response
            text: Text content for the response
        """
        self.status_code = status_code
        self.json_data = json_data or {}
        self.text = text or ""
    
    def json(self) -> Dict[str, Any]:
        """Return JSON data."""
        return self.json_data
    
    def raise_for_status(self) -> None:
        """Raise an exception if status code indicates an error."""
        if self.status_code >= 400:
            raise Exception(f"Mock HTTP error: {self.status_code}")

@pytest.fixture(scope="function")
def mock_http_response() -> MockResponse:
    """
    Create a mock HTTP response for testing.
    
    Returns:
        Mock HTTP response object
    """
    return MockResponse(
        status_code=200,
        json_data={"result": "success", "data": {"key": "value"}},
        text="Success"
    )

@pytest.fixture(scope="function")
def mock_http_error() -> MockResponse:
    """
    Create a mock HTTP error response for testing.
    
    Returns:
        Mock HTTP error response object
    """
    return MockResponse(
        status_code=404,
        json_data={"error": "Not found", "message": "Resource not found"},
        text="Not Found"
    )
