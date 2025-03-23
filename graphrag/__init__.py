"""
GraphRAG - Graph-based Retrieval Augmented Generation System
"""

__version__ = "0.1.0"

# Ensure NLTK resources are available
import nltk
import os
from nltk.data import find as nltk_find

def ensure_nltk_resources():
    """Ensure required NLTK resources are available for tokenization"""
    resources_ok = True
    
    # Try to use punkt_tab (newer NLTK versions)
    try:
        nltk.download('punkt_tab', quiet=True)
        try:
            nltk_find('tokenizers/punkt_tab')
            return True  # punkt_tab is available
        except LookupError:
            # punkt_tab download didn't work, fall back to punkt
            pass
    except:
        # punkt_tab not available, fall back to punkt
        pass
        
    # Fall back to punkt (older NLTK versions)
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        return True
    except Exception as e:
        # Both punkt and punkt_tab failed
        resources_ok = False
    
    return resources_ok

# Ensure NLTK resources are available
ensure_nltk_resources()

# Import Loguru logger (must be first to configure logging for all modules)
from graphrag.utils.logger import logger
logger.debug("NLTK resources initialized")

# Import core functionality to make it available at the top level
from graphrag.cli.main import main, query_graphrag, process_files, setup_database
# Import commonly used functions from core modules
from graphrag.core.ingest import process_document
from graphrag.core.retrieval import hybrid_retrieve, hybrid_retrieve_with_triplets 