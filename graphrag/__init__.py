"""
GraphRAG - Graph-based Retrieval Augmented Generation System
"""

__version__ = "0.1.0"

# Import core functionality to make it available at the top level
from graphrag.cli.main import main, query_graphrag, process_files, setup_database
# Import commonly used functions from core modules
from graphrag.core.ingest import process_document
from graphrag.core.retrieval import hybrid_retrieve, hybrid_retrieve_with_triplets 