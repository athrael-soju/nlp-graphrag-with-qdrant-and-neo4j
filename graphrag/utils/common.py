"""
Shared utilities for GraphRAG
"""

import numpy as np
import os
from typing import List, Union
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import Loguru logger
from graphrag.utils.logger import logger

# Default embedding model from environment variables
DEFAULT_EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'intfloat/e5-base-v2')

# Global cache for embedding models to avoid reloading
_embedding_models = {}

def get_embedding_model(model_name=None):
    """Get or load an embedding model
    
    Args:
        model_name: Name or path of the embedding model
        
    Returns:
        SentenceTransformer model instance
    """
    global _embedding_models
    
    # Use environment variable if not provided
    model_name = model_name or DEFAULT_EMBEDDING_MODEL
    
    if model_name in _embedding_models:
        return _embedding_models[model_name]
    
    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name)
        _embedding_models[model_name] = model
        logger.info(f"Successfully loaded embedding model: {model_name}")
        return model
    except Exception as e:
        logger.error(f"Failed to load embedding model {model_name}: {str(e)}")
        raise

def embed_text(text: Union[str, List[str]], model_name=None, 
                prefix=None, normalize=True) -> np.ndarray:
    """Generate embeddings for text
    
    Args:
        text: Input text or list of texts
        model_name: Name or path of the embedding model
        prefix: Optional prefix to add to the text (e.g., "passage:", "query:")
        normalize: Whether to normalize the embeddings
        
    Returns:
        np.ndarray: Text embedding(s)
    """
    # Use environment variable if not provided
    model_name = model_name or DEFAULT_EMBEDDING_MODEL
    
    model = get_embedding_model(model_name)
    
    # Handle single string vs list
    is_single = isinstance(text, str)
    texts = [text] if is_single else text
    
    # Add prefixes if required
    if prefix:
        texts = [f"{prefix} {t}" for t in texts]
    # Special handling for E5 models
    elif 'e5' in model_name.lower():
        if any('query:' in t.lower() for t in texts) or any('passage:' in t.lower() for t in texts):
            # Prefixes already present
            pass
        else:
            # Assume these are passages if no explicit prefix
            texts = [f"passage: {t}" for t in texts]
    
    # Generate embeddings
    try:
        embeddings = model.encode(texts, normalize_embeddings=normalize)
        
        # Return single vector or array of vectors
        if is_single:
            return embeddings[0]
        return embeddings
    except Exception as e:
        logger.error(f"Error embedding text: {str(e)}")
        # Return zero vector(s) as fallback
        dim = model.get_sentence_embedding_dimension()
        if is_single:
            return np.zeros(dim)
        return np.zeros((len(texts), dim))
