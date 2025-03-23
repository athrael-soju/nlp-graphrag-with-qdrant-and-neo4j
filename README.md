# GraphRAG: Graph-based Retrieval Augmented Generation

GraphRAG is a Python system that implements a knowledge graph enhanced retrieval-augmented generation (RAG) pipeline. It combines traditional vector-based document retrieval with graph-based knowledge representation for improved context retrieval.

## Features

- Document ingestion from plain text or PDF files
- Automated chunking and embedding of text segments 
- Document chain modeling with NEXT/PREV relationships between sequential chunks
- Context-aware retrieval that provides surrounding chunks for better context
- NLP-powered knowledge graph construction with:
  - Term nodes (tokens, n-grams) linked to document chunks
  - Entity-relationship triplets extracted from text
  - Full semantic search capabilities
- Hybrid retrieval combining vector search and graph traversal
- Interactive querying interface with context customization

## Architecture

GraphRAG consists of several key components:

1. **Document Ingestion** - Loads text/PDF files and splits them into manageable chunks
2. **Document Chaining** - Creates NEXT/PREV relationships between sequential chunks
3. **Vector Indexing** - Embeds text chunks using a transformer model (default: E5-base)
4. **Term Graph Construction** - Extracts tokens, bigrams, and trigrams, linking them to chunks
5. **Entity Extraction** - Creates entity nodes from subjects and objects in text
6. **Triplet Extraction** - Uses a T5 model to identify subject-relation-object triplets
7. **Hybrid Retrieval** - Combines vector similarity with graph traversal for better context
8. **Context-aware Retrieval** - Includes surrounding chunks in results to provide more coherent context
9. **CLI Interface** - Provides command-line tools for processing and querying

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

## Core Algorithms

### Document Ingestion and Chunking (ingest.py)

The document ingestion process follows these steps:

1. **Text Extraction**: For PDF files, PyMuPDF (fitz) extracts text while preserving structure
2. **Semantic Chunking**: Uses NLTK's sentence tokenizer to split text into sentences, then groups them into chunks while respecting semantic boundaries
3. **Chunk Embedding**: Each chunk is embedded using the E5-base model with prefix-based tuning to generate dense vector representations
4. **Document Chain Construction**: Creates a linked list of chunks with NEXT/PREV relationships to preserve document flow

The chunking algorithm balances two objectives:
- Keeping semantic units (like paragraphs) together
- Maintaining a maximum token count per chunk to optimize for context windows

### NLP Graph Construction (nlp_graph.py)

The graph construction process builds a rich knowledge graph with these components:

1. **N-gram Extraction**: 
   - Extracts unigrams (tokens), bigrams, and trigrams from text
   - Filters noise using stop words and frequency analysis
   - Creates weighted relationships between terms and document chunks

2. **Term Relationship Modeling**:
   - Builds CO-OCCURS relationships between terms that appear in proximity
   - Establishes weighted edges based on co-occurrence frequency
   - Creates a semantic network that can be traversed to find related concepts

3. **Entity Recognition**:
   - Identifies named entities and key concepts
   - Links entities to their source chunks
   - Creates a subgraph of important domain concepts

### Triplet Extraction (triplets.py)

The system extracts semantic triplets (subject-relation-object) using:

1. **T5-based Triplet Extraction**:
   - Uses a fine-tuned T5 model (bew/t5_sentence_to_triplet_xl) to transform text into SPO triplets
   - Applies threshold filtering to ensure quality extractions
   - Creates entity nodes for subjects and objects

2. **Relationship Modeling**:
   - Builds typed relationships between entities based on extracted predicates
   - Maintains provenance by linking relationships to source chunks
   - Creates a semantic network of factual knowledge

3. **Coreference Resolution** (optional):
   - Resolves pronouns and references across sentences
   - Links entities across document boundaries
   - Improves graph connectedness

### Hybrid Retrieval (retrieval.py)

The core of GraphRAG is its hybrid retrieval algorithm:

1. **Vector-based Search**:
   - Embeds the query using the same model as documents
   - Performs ANN search with Qdrant to find semantically similar chunks
   - Returns top-k matches based on cosine similarity

2. **Graph Traversal Enhancement**:
   - Identifies key terms in the query
   - Traverses the knowledge graph to find relevant chunks connected to these terms
   - Performs multi-hop exploration to discover indirectly relevant content

3. **Hybrid Scoring**:
   - Combines vector similarity and graph relevance scores
   - Uses a configurable weighting mechanism to balance semantic and structural relevance
   - Sorts results by composite score

4. **Context-aware Result Processing**:
   - For each matched chunk, retrieves surrounding chunks (PREV/NEXT)
   - Provides a coherent context window that preserves document flow
   - Deduplicates and merges overlapping contexts

This hybrid approach outperforms pure vector search by leveraging structural information from the knowledge graph while maintaining the semantic power of dense embeddings.

## Installation

### Prerequisites

GraphRAG requires Neo4j database (Community Edition 5.x or higher) to be installed and running. You can run Neo4j via Docker:

```bash
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/testpassword neo4j:5.8.0
```

Or install Neo4j directly from [neo4j.com](https://neo4j.com/download/).

GraphRAG also uses Qdrant for vector similarity search, which can be run with Docker:

```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

### Using Docker Compose

The easiest way to get started is using the provided docker-compose file:

```bash
docker-compose up -d
```

This will start both Neo4j and Qdrant with the correct configuration.

### Install GraphRAG

Clone this repository and install the package:

```bash
git clone https://github.com/yourusername/graphrag.git
cd graphrag

python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

pip install -e .
```

> **Note:** Required NLTK resources (tokenizers and stopwords) are automatically downloaded during package installation. GraphRAG supports both older NLTK versions (using punkt) and newer versions (using punkt_tab).

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

# Enable context-aware retrieval (if WITH_CONTEXT=False in .env)
graphrag query "What type of business is Hugging Face?" --with-context

# Disable context-aware retrieval (if WITH_CONTEXT=True in .env)
graphrag query "What type of business is Hugging Face?" --no-context

# Customize context window size
graphrag query "What type of business is Hugging Face?" --context-size 3

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
MAX_TOKENS_PER_CHUNK=50
TOP_K_RETRIEVAL=5
WITH_CONTEXT=True
CONTEXT_SIZE=2

# Logging Settings
LOG_LEVEL=INFO
LOG_FILE=logs/graphrag.log
```

You can modify these settings to customize the behavior of GraphRAG without changing the code.

### Context Settings

- `WITH_CONTEXT`: Controls whether context-aware retrieval is enabled by default
  - When `True`: Context-aware retrieval is enabled by default, and a `--no-context` flag is available to disable it
  - When `False`: Standard retrieval is used by default, and a `--with-context` flag is available to enable context-aware retrieval
- `CONTEXT_SIZE`: Number of chunks to include before and after each matching chunk (default: 2)

Context-aware retrieval leverages the document structure by returning not just the matching chunks, but also the surrounding chunks (previous and next) to provide better context for the LLM.

### Logging System

GraphRAG uses Loguru for enhanced, structured logging:

- `LOG_LEVEL`: Set the minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `LOG_FILE`: If specified, logs will be written to this file with automatic rotation (10MB size, 1 week retention)

Loguru provides several benefits over standard logging:
- Colorized output for better readability
- Structured format with timestamps and source information
- Automatic log rotation and compression
- Intercepts and formats logs from third-party libraries

Example log output:
```
2023-07-21 15:30:45.123 | INFO     | graphrag.core.ingest:process_document:122 - Processing document example_doc...
```

## Advanced Usage

### Context-Aware Retrieval

GraphRAG's context-aware retrieval enhances traditional RAG by maintaining the flow of information from the source documents. The behavior depends on your environment settings in `.env`:

```bash
# If WITH_CONTEXT=False in .env, use this to enable context:
graphrag query "Who is Hitomi Kanzaki?" --with-context

# If WITH_CONTEXT=True in .env, context is enabled by default:
graphrag query "Who is Hitomi Kanzaki?"

# Increase context window size (chunks before/after matches)
graphrag query "Who is Hitomi Kanzaki?" --context-size 5
```

In the output, matches are marked with 🔍 MATCH and context chunks with 📄 CONTEXT.

### Interactive Mode with Context

You can also use context-aware retrieval in interactive mode:

```bash
graphrag interactive

# In the interactive session:
GraphRAG> set context on
Set context to on
GraphRAG> set context_size 3
Set context_size to 3
GraphRAG> Who is Van Fanel?
```

### Using Spark NLP for Large-Scale Processing (Optional)

For processing large document collections, GraphRAG can use Spark NLP for distributed NLP processing. Install the additional dependencies:

```bash
pip install pyspark spark-nlp
```

The system will automatically detect and use Spark NLP if available.

### Using GPU for Embedding and Triplet Extraction

To accelerate embedding and triplet extraction, ensure you have PyTorch installed with CUDA support and the required GPU drivers.

## Examples

### Process a document about fantasy anime:

```bash
graphrag process escaflowne.txt
```

### Query with context-aware retrieval:

```bash
graphrag query "What is Escaflowne about?" --with-context
```

Sample output:
```
================================================================================
QUERY RESULTS:
================================================================================

Retrieved chunks with context:

1. 🔍 MATCH: Chunk escaflowne_chunk1 (score: 0.852):
----------------------------------------
The narrative follows Hitomi Kanzaki, a seemingly ordinary high school girl from Earth who is transported to the mystical world of Gaea, a planet where the Earth and Moon hang in the sky. There, she becomes entangled in a conflict involving powerful nations, ancient prophecies, and a legendary mecha called Escaflowne, piloted by the enigmatic prince Van Fanel.

2. 📄 CONTEXT: Chunk escaflowne_chunk0:
----------------------------------------
Escaflowne: A Blend of Fantasy, Mecha, and Romance

The Vision of Escaflowne (often simply called Escaflowne) is a unique and genre-defying anime that originally aired in 1996. Created by Sunrise and directed by Kazuki Akane, the series combines elements of fantasy, science fiction, romance, and mecha—blending them into a rich, emotional, and visually captivating story.

3. 📄 CONTEXT: Chunk escaflowne_chunk2:
----------------------------------------
As Hitomi grapples with her newfound powers of clairvoyance, she also navigates complex relationships and the harsh realities of war.

================================================================================
```

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