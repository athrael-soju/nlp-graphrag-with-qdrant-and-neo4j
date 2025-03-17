"""
Configuration utilities for GraphRAG
"""

import os
import logging
from typing import Any, Dict, Optional
from dotenv import load_dotenv, find_dotenv

# Initialize logger
logger = logging.getLogger(__name__)

def reload_env():
    """
    Reload environment variables from .env file
    
    Returns:
        bool: True if .env file was found and loaded
    """
    # Find the .env file
    dotenv_path = find_dotenv()
    if not dotenv_path:
        logger.warning("No .env file found. Using default values.")
        return False
        
    # Load environment variables with override=True to refresh values
    load_dotenv(dotenv_path, override=True)
    logger.info(f"Loaded environment variables from {dotenv_path}")
    return True

# Load environment variables on module import
reload_env()

def get_config(key: str, default: Optional[Any] = None) -> Any:
    """
    Get a configuration value from environment variables
    
    Args:
        key: Configuration key
        default: Default value if not found
        
    Returns:
        Configuration value
    """
    # Always reload env to get fresh values
    reload_env()
    
    # Get from environment variables
    env_value = os.getenv(key.upper())
    
    # Debug logging
    logger.info(f"Config: {key.upper()}={env_value}, default={default}, type(default)={type(default)}")
    
    # If not found, return default
    if env_value is None:
        return default
        
    # Handle boolean values
    if isinstance(default, bool):
        # Convert string to boolean
        if env_value.lower() in ('true', 'yes', '1', 'y', 'on'):
            logger.info(f"Converting {key.upper()}={env_value} to boolean True")
            return True
        elif env_value.lower() in ('false', 'no', '0', 'n', 'off'):
            logger.info(f"Converting {key.upper()}={env_value} to boolean False")
            return False
        # If not a recognized boolean string, use default
        logger.warning(f"Unrecognized boolean value: {key.upper()}={env_value}, using default={default}")
        return default
    
    # Handle integer values
    if isinstance(default, int):
        try:
            return int(env_value)
        except ValueError:
            return default
            
    # Handle float values
    if isinstance(default, float):
        try:
            return float(env_value)
        except ValueError:
            return default
            
    # Return as string for other types
    return env_value

def get_neo4j_config() -> Dict[str, Any]:
    """
    Get Neo4j configuration
    
    Returns:
        Dict with Neo4j configuration
    """
    return {
        "uri": get_config("NEO4J_URI", "bolt://localhost:7687"),
        "auth": (get_config("NEO4J_USER", "neo4j"), get_config("NEO4J_PASSWORD", "testpassword"))
    }

def get_qdrant_config() -> Dict[str, Any]:
    """
    Get Qdrant configuration
    
    Returns:
        Dict with Qdrant configuration
    """
    config = {
        "url": get_config("QDRANT_URL", "http://localhost:6333"),
    }
    
    # Add API key only if it exists
    api_key = get_config("QDRANT_API_KEY")
    if api_key:
        config["api_key"] = api_key
        
    return config

def get_model_config() -> Dict[str, Any]:
    """
    Get model configuration
    
    Returns:
        Dict with model configuration
    """
    return {
        "triplet_model": get_config("TRIPLET_MODEL", "bew/t5_sentence_to_triplet_xl"),
        "embedding_model": get_config("EMBEDDING_MODEL", "intfloat/e5-base-v2")
    }

def get_process_config() -> Dict[str, Any]:
    """
    Get processing configuration
    
    Returns:
        Dict with processing configuration
    """
    return {
        "max_tokens_per_chunk": get_config("MAX_TOKENS_PER_CHUNK", 200),
        "top_k_retrieval": get_config("TOP_K_RETRIEVAL", 10),
        "with_context": get_config("WITH_CONTEXT", False),
        "context_size": get_config("CONTEXT_SIZE", 2)
    }
