"""
Helper Utilities

This module provides various helper functions used throughout the framework.
"""

import json
import re
import os
import urllib.parse
from typing import Any, Dict, List, Optional, Union

def format_json(obj: Any, indent: int = 2) -> str:
    """
    Format an object as a JSON string with proper indentation.
    
    Args:
        obj: Object to format
        indent: Indentation level
        
    Returns:
        Formatted JSON string
    """
    try:
        return json.dumps(obj, indent=indent, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)

def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    # Try to truncate at a space to avoid cutting words
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > 0:
        truncated = truncated[:last_space]
    
    return truncated + suffix

def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to be safe for file system operations.
    
    Args:
        filename: Filename to sanitize
        
    Returns:
        Sanitized filename
    """
    # Replace invalid characters with underscores
    sanitized = re.sub(r'[\\/*?:"<>|]', '_', filename)
    
    # Ensure filename is not too long (max 255 characters)
    if len(sanitized) > 255:
        base, ext = os.path.splitext(sanitized)
        sanitized = base[:255 - len(ext)] + ext
    
    return sanitized

def is_valid_url(url: str) -> bool:
    """
    Check if a string is a valid URL.
    
    Args:
        url: URL to check
        
    Returns:
        True if the URL is valid, False otherwise
    """
    try:
        result = urllib.parse.urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract a JSON object from text.
    
    Args:
        text: Text containing JSON
        
    Returns:
        Extracted JSON object or empty dict if not found
    """
    # Find JSON objects in the text using regex
    json_pattern = r'```(?:json)?\s*((?:\{|\[).*?(?:\}|\]))```'
    json_match = re.search(json_pattern, text, re.DOTALL)
    
    if json_match:
        json_str = json_match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try to clean up the JSON string
            cleaned_str = re.sub(r'^\s*```.*\n', '', json_str)
            cleaned_str = re.sub(r'\n\s*```\s*$', '', cleaned_str)
            try:
                return json.loads(cleaned_str)
            except json.JSONDecodeError:
                pass
    
    # Try finding JSON without code blocks
    json_pattern = r'(?:\{|\[).*?(?:\}|\])'
    json_matches = re.findall(json_pattern, text, re.DOTALL)
    
    for json_str in json_matches:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            continue
    
    return {}

def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (overrides dict1)
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result

def parse_list_string(list_str: str) -> List[str]:
    """
    Parse a string representation of a list into a list of strings.
    
    Args:
        list_str: String representation of a list (comma-separated or JSON)
        
    Returns:
        List of strings
    """
    if not list_str:
        return []
    
    # Try parsing as JSON
    try:
        parsed = json.loads(list_str)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    
    # Parse as comma-separated list
    items = [item.strip() for item in list_str.split(',') if item.strip()]
    return items

def extract_code_from_text(text: str, language: Optional[str] = None) -> Optional[str]:
    """
    Extract code blocks from text.
    
    Args:
        text: Text containing code blocks
        language: Optional language filter
        
    Returns:
        Extracted code or None if not found
    """
    # If language is specified, look for that specific language
    if language:
        code_pattern = rf'```(?:{language})\s*(.*?)```'
        code_match = re.search(code_pattern, text, re.DOTALL)
        
        if code_match:
            return code_match.group(1).strip()
    
    # Otherwise, look for any code block
    code_pattern = r'```(?:\w+)?\s*(.*?)```'
    code_match = re.search(code_pattern, text, re.DOTALL)
    
    if code_match:
        return code_match.group(1).strip()
    
    return None

def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds as a human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 0.001:
        return f"{seconds * 1000000:.2f} µs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    else:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes} min {seconds:.2f} s"

def count_tokens(text: str, model: str = "claude-3-7-sonnet") -> int:
    """
    Estimate the number of tokens in a text string.
    
    This is a simple approximation. For accurate counts, use the tokenizer
    specific to your LLM provider.
    
    Args:
        text: Text to count tokens for
        model: Model to use for estimation
        
    Returns:
        Estimated token count
    """
    # Very rough approximation: ~4 characters per token for English text
    # This varies by model and content
    if not text:
        return 0
    
    char_count = len(text)
    
    # Different models have different tokenization approaches
    if "claude" in model.lower():
        # Claude models: approximately 3.5-4 chars per token for English
        return char_count // 4 + 1
    elif "gpt" in model.lower():
        # GPT models: approximately 4 chars per token for English
        return char_count // 4 + 1
    else:
        # Generic fallback
        return char_count // 4 + 1
