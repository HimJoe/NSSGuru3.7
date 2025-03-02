"""
Configuration Module

This module manages configuration settings for the Agentic AI framework,
supporting multiple configuration sources including environment variables,
configuration files, and programmatic settings.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from copy import deepcopy

from .utils.logger import get_logger

# Get module logger
logger = get_logger("config")

# Default configuration values
DEFAULT_CONFIG = {
    # General settings
    "environment": "development",
    "log_level": "INFO",
    
    # LLM settings
    "llm": {
        "provider": "anthropic",
        "model": "claude-3-7-sonnet",
        "temperature": 0.7,
        "max_tokens": 4000,
        "api_key": None,  # Should be set via environment variable or config file
    },
    
    # Agent settings
    "agents": {
        "default_system_prompt": None,  # Will use the framework default
        "max_tool_iterations": 5,
        "timeout": 60,  # Seconds
    },
    
    # Memory settings
    "memory": {
        "vector_store": {
            "storage_path": None,  # Will default to ~/.agentic_ai/vector_store
            "similarity_threshold": 0.7,
            "max_entries": 1000,
        },
        "working_memory": {
            "max_entries": 100,
            "buffer_type": "fifo",
        },
    },
    
    # Tool settings
    "tools": {
        "allowed_tools": [
            "web_search",
            "execute_python",
            "file_operations",
        ],
        "web_search": {
            "search_engine": "google",
            "max_results": 5,
            "timeout": 10.0,
        },
        "execute_python": {
            "timeout": 5.0,
            "max_output_length": 8192,
            "allow_plots": True,
            "additional_allowed_modules": [],
        },
    },
    
    # Security settings
    "security": {
        "max_api_calls_per_minute": 60,
        "max_tokens_per_request": 16000,
        "max_conversation_length": 100,
    },
}

# Global configuration instance
_config = deepcopy(DEFAULT_CONFIG)
_config_loaded = False

def reset_config() -> None:
    """Reset configuration to default values."""
    global _config, _config_loaded
    _config = deepcopy(DEFAULT_CONFIG)
    _config_loaded = False
    logger.info("Configuration reset to default values")

def get_config() -> Dict[str, Any]:
    """
    Get the current configuration.
    
    If the configuration hasn't been loaded yet, it will attempt to load it
    from environment variables and default files.
    
    Returns:
        Complete configuration dictionary
    """
    global _config_loaded
    
    if not _config_loaded:
        # Load configuration from default sources
        load_config()
    
    return deepcopy(_config)

def load_config(
    config_file: Optional[str] = None,
    env_prefix: str = "AGENTIC_AI_",
    merge: bool = True
) -> Dict[str, Any]:
    """
    Load configuration from files and environment variables.
    
    Args:
        config_file: Path to configuration file (JSON or YAML)
        env_prefix: Prefix for environment variables
        merge: Whether to merge with existing config or replace entirely
        
    Returns:
        Loaded configuration dictionary
    """
    global _config, _config_loaded
    
    # Start with default or current config
    if merge:
        config = deepcopy(_config)
    else:
        config = deepcopy(DEFAULT_CONFIG)
    
    # Load from default config file locations if no specific file provided
    if not config_file:
        # Check standard locations
        potential_locations = [
            # Current directory
            Path.cwd() / "agentic_ai_config.json",
            Path.cwd() / "agentic_ai_config.yaml",
            Path.cwd() / "agentic_ai_config.yml",
            
            # User config directory
            Path.home() / ".agentic_ai" / "config.json",
            Path.home() / ".agentic_ai" / "config.yaml",
            Path.home() / ".agentic_ai" / "config.yml",
            
            # System config directory
            Path("/etc/agentic_ai/config.json"),
            Path("/etc/agentic_ai/config.yaml"),
            Path("/etc/agentic_ai/config.yml"),
        ]
        
        # Use the first file that exists
        for path in potential_locations:
            if path.exists():
                config_file = str(path)
                logger.info(f"Found configuration file: {config_file}")
                break
    
    # Load from config file if specified or found
    if config_file:
        try:
            file_path = Path(config_file)
            
            if not file_path.exists():
                logger.warning(f"Configuration file not found: {config_file}")
            else:
                # Determine file type and load accordingly
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    with open(file_path, 'r') as f:
                        file_config = yaml.safe_load(f)
                else:
                    # Default to JSON
                    with open(file_path, 'r') as f:
                        file_config = json.load(f)
                
                # Merge file config into current config
                _deep_update(config, file_config)
                logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_file}: {str(e)}")
    
    # Load from environment variables
    env_config = _load_from_env(env_prefix)
    if env_config:
        _deep_update(config, env_config)
        logger.info("Loaded configuration from environment variables")
    
    # Update global config
    _config = config
    _config_loaded = True
    
    return deepcopy(config)

def _load_from_env(prefix: str) -> Dict[str, Any]:
    """
    Load configuration from environment variables.
    
    Environment variables should be in the format:
    PREFIX_SECTION_KEY=value
    
    For example:
    AGENTIC_AI_LLM_MODEL=claude-3-7-sonnet
    
    Args:
        prefix: Prefix for environment variables
        
    Returns:
        Configuration dictionary built from environment variables
    """
    config = {}
    
    # Find all environment variables with the prefix
    for key, value in os.environ.items():
        if key.startswith(prefix):
            # Remove prefix and split into parts
            key_path = key[len(prefix):].lower().split('_')
            
            # Build nested dictionary
            current = config
            for part in key_path[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Set the value
            current[key_path[-1]] = _parse_env_value(value)
    
    return config

def _parse_env_value(value: str) -> Any:
    """Parse environment variable value to appropriate type."""
    # Check for boolean
    if value.lower() in ['true', 'yes', '1']:
        return True
    if value.lower() in ['false', 'no', '0']:
        return False
    
    # Check for number
    try:
        if '.' in value:
            return float(value)
        else:
            return int(value)
    except ValueError:
        pass
    
    # Check for JSON
    if value.startswith('{') or value.startswith('['):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    
    # Default to string
    return value

def _deep_update(base_dict: Dict, update_dict: Dict) -> None:
    """
    Recursively update a dictionary with another dictionary.
    
    Args:
        base_dict: Dictionary to update
        update_dict: Dictionary with updates
    """
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value

def update_config(updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with new values.
    
    Args:
        updates: Dictionary with configuration updates
        
    Returns:
        Updated configuration dictionary
    """
    global _config
    
    # Apply updates
    _deep_update(_config, updates)
    logger.info("Configuration updated programmatically")
    
    return deepcopy(_config)

def get_config_value(path: Union[str, List[str]], default: Any = None) -> Any:
    """
    Get a specific configuration value using a dot-notation path.
    
    Args:
        path: Configuration path (e.g., "llm.model" or ["llm", "model"])
        default: Default value if path doesn't exist
        
    Returns:
        Configuration value or default
    """
    # Ensure config is loaded
    config = get_config()
    
    # Convert string path to list
    if isinstance(path, str):
        path_parts = path.split('.')
    else:
        path_parts = path
    
    # Traverse the config dictionary
    current = config
    try:
        for part in path_parts:
            current = current[part]
        return current
    except (KeyError, TypeError):
        return default
