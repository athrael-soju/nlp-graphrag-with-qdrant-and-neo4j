# GraphRAG: Graph-based Retrieval Augmented Generation

GraphRAG is a Python system that implements a knowledge graph enhanced retrieval-augmented generation (RAG) pipeline. It combines traditional vector-based document retrieval with graph-based knowledge representation for improved context retrieval.

## Features

- Document ingestion from plain text or PDF files
- Automated chunking and embedding of text segments 
- NLP-powered knowledge graph construction with:
  - Term nodes (tokens, n-grams) linked to document chunks
  - Entity-relationship triplets extracted from text
  - Full semantic search capabilities
- Hybrid retrieval combining vector search and graph traversal
- Interactive querying interface

## Installation

### Prerequisites

GraphRAG requires Neo4j database (Community Edition 5.x or higher) to be installed and running. You can run Neo4j via Docker:

```bash
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/testpassword neo4j:5.8.0
```

Or install Neo4j directly from [neo4j.com](https://neo4j.com/download/).

### Install GraphRAG

Clone this repository and install the package:

```bash
git clone https://github.com/yourusername/graphrag.git
cd graphrag
pip install -e .
```

## Quick Start

1. First, set up the Neo4j database with required indexes:

```bash
graphrag setup
```

2. Process a text or PDF document:

```bash
# Process a single text file
graphrag process path/to/document.txt

# Process a PDF file
graphrag process --pdf path/to/document.pdf

# Process multiple files
graphrag process file1.txt file2.txt file3.txt
```

3. Query the system:

```bash
# Run a single query
graphrag query "What type of business is Hugging Face?"

# Start interactive session
graphrag interactive
```

## Configuration

GraphRAG uses environment variables for configuration. You can set these in a `.env` file in the project root directory:

```
# Neo4j Connection Settings
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=testpassword

# Qdrant Connection Settings
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=

# Database Collection Settings
QDRANT_COLLECTION_NAME=tokens
VECTOR_SIZE=768

# Model Settings
TRIPLET_MODEL=bew/t5_sentence_to_triplet_xl
EMBEDDING_MODEL=intfloat/e5-base-v2

# Processing Settings
MAX_TOKENS_PER_CHUNK=200
TOP_K_RETRIEVAL=10
```

You can modify these settings to customize the behavior of GraphRAG without changing the code.

## Architecture

GraphRAG consists of several key components:

1. **Document Ingestion** - Loads text/PDF files and splits them into manageable chunks
2. **Vector Indexing** - Embeds text chunks using a transformer model (default: E5-base)
3. **Term Graph Construction** - Extracts tokens, bigrams, and trigrams, linking them to chunks
4. **Entity Extraction** - Creates entity nodes from subjects and objects in text
5. **Triplet Extraction** - Uses a T5 model to identify subject-relation-object triplets
6. **Hybrid Retrieval** - Combines vector similarity with graph traversal for better context
7. **CLI Interface** - Provides command-line tools for processing and querying

The system follows a modular design with the following structure:

```
graphrag/
├── core/           # Core functionality
│   ├── ingest.py   # Document ingestion
│   ├── retrieval.py # Retrieval mechanisms
│   ├── nlp_graph.py # Graph construction
│   └── triplets.py  # Triplet extraction
├── connectors/     # Database connectors
│   ├── neo4j_connection.py
│   └── qdrant_connection.py
├── models/         # ML models and embeddings
├── utils/          # Utility functions
│   ├── common.py   # Shared utilities like embedding
│   └── config.py   # Configuration
├── cli/            # Command-line interface
│   └── main.py
├── data/           # Data handling
```

## Advanced Usage

### Using Spark NLP for Large-Scale Processing (Optional)

For processing large document collections, GraphRAG can use Spark NLP for distributed NLP processing. Install the additional dependencies:

```bash
pip install pyspark spark-nlp
```

The system will automatically detect and use Spark NLP if available.

### Using GPU for Embedding and Triplet Extraction

To accelerate embedding and triplet extraction, ensure you have PyTorch installed with CUDA support and the required GPU drivers.

## Examples

### Process a document about a tech company:

```bash
echo "Hugging Face, Inc. is an American company that develops tools for building applications using machine learning. It was founded in 2016 and its headquarters is in New York City." > huggingface.txt

graphrag process huggingface.txt
```

### Query the system:

```bash
graphrag query "Where is Hugging Face headquartered?"
```

Output might include:
- Retrieved chunks containing the headquarters information
- A knowledge graph triplet: (Hugging Face, Inc.) -[HEADQUARTERS_IN]-> (New York City)

## Limitations

- The triplet extraction model works best on simple, factual statements
- Large documents may require significant processing time and memory
- Complex queries might need refinement for optimal results

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use GraphRAG in your research, please cite:

```
@software{graphrag2023,
  author = {GraphRAG Team},
  title = {GraphRAG: Graph-based Retrieval Augmented Generation},
  year = {2023},
  url = {https://github.com/yourusername/graphrag}
}
``` 