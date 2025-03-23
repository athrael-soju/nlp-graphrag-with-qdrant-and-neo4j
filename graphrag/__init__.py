"""
GraphRAG - Graph-based Retrieval Augmented Generation System
"""

__version__ = "0.1.0"

# Apply NLTK punkt_tab patch to fix tokenization issues
import nltk

def _apply_punkt_tab_patch():
    """Apply patch to fix NLTK punkt_tab issue"""
    # Store the original load function
    _original_load = nltk.data.load
    
    def patched_load(resource_url, format='auto', cache=True, verbose=False, *args, **kwargs):
        """
        Patched version of nltk.data.load that handles punkt_tab issues
        """
        # If trying to load punkt_tab, redirect to punkt
        if 'punkt_tab' in resource_url:
            fixed_url = resource_url.replace('punkt_tab', 'punkt')
            if 'english/' in fixed_url:
                fixed_url = fixed_url.replace('english/', '')
            resource_url = fixed_url
        
        # Call the original function
        return _original_load(resource_url, format, cache, verbose, *args, **kwargs)
    
    # Apply the patch
    nltk.data.load = patched_load

# Apply the patch immediately
_apply_punkt_tab_patch()

# Ensure NLTK resources are downloaded
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception:
    pass

# Import Loguru logger (must be first to configure logging for all modules)
from graphrag.utils.logger import logger
logger.debug("NLTK punkt_tab patch applied")

# Import core functionality to make it available at the top level
from graphrag.cli.main import main, query_graphrag, process_files, setup_database
# Import commonly used functions from core modules
from graphrag.core.ingest import process_document
from graphrag.core.retrieval import hybrid_retrieve, hybrid_retrieve_with_triplets 