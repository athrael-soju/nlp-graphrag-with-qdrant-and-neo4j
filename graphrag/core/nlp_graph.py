"""
NLP graph creation utilities for GraphRAG
"""

import nltk
from typing import List, Tuple, Optional
from itertools import chain
from graphrag.connectors.neo4j_connection import get_connection
from graphrag.utils.logger import logger

# No need to download NLTK resources here as it's handled in __init__.py

# Get stopwords
try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
except Exception as e:
    logger.warning(f"Failed to load NLTK stopwords: {str(e)}")
    STOPWORDS = set()

class NLPGraphBuilder:
    """Builds NLP-enhanced knowledge graph from text chunks"""
    
    def __init__(self, neo4j_conn=None, remove_stopwords=True):
        """Initialize NLPGraphBuilder
        
        Args:
            neo4j_conn: Neo4j connection instance
            remove_stopwords: Whether to remove stopwords from unigrams
        """
        self.neo4j = neo4j_conn or get_connection()
        self.remove_stopwords = remove_stopwords
        logger.info("Initialized NLPGraphBuilder")
        
    def extract_ngrams(self, text: str) -> Tuple[List[str], List[str], List[str]]:
        """Extract unigrams, bigrams, and trigrams from text
        
        Args:
            text: Input text
            
        Returns:
            Tuple[List[str], List[str], List[str]]: Tuple of (unigrams, bigrams, trigrams)
        """
        # Tokenize and normalize text
        tokens = [w.lower() for w in nltk.word_tokenize(text) if w.isalnum()]
        
        # Filter stopwords if required
        if self.remove_stopwords:
            unigrams = [t for t in tokens if t not in STOPWORDS]
        else:
            unigrams = tokens
            
        # Generate bigrams and trigrams
        bigrams = [' '.join(b) for b in nltk.bigrams(tokens)]
        trigrams = [' '.join(t) for t in nltk.trigrams(tokens)]
        
        logger.debug(f"Extracted {len(unigrams)} unigrams, {len(bigrams)} bigrams, {len(trigrams)} trigrams")
        return unigrams, bigrams, trigrams
        
    def store_terms_for_chunk(self, chunk_id: str, unigrams: List[str], 
                          bigrams: List[str], trigrams: List[str]) -> None:
        """Store terms (n-grams) and connect them to their chunk in Neo4j
        
        Args:
            chunk_id: Chunk identifier
            unigrams: List of unigrams (single tokens)
            bigrams: List of bigrams (two-word phrases)
            trigrams: List of trigrams (three-word phrases)
        """
        logger.info(f"Storing terms for chunk {chunk_id}")
        
        # Combine all terms with a type indicator
        terms = [(t, "unigram") for t in unigrams] + \
                [(t, "bigram") for t in bigrams] + \
                [(t, "trigram") for t in trigrams]
                
        batch_size = 100  # Process in batches to avoid large transactions
        for i in range(0, len(terms), batch_size):
            batch = terms[i:i+batch_size]
            
            # Use a single parameterized query for the batch
            params = {
                "chunk_id": chunk_id,
                "terms": [{"text": term, "type": term_type} for term, term_type in batch]
            }
            
            try:
                # Use a query that doesn't involve vector operations
                self.neo4j.run_query(
                    """
                    MATCH (c:Chunk {id: $chunk_id})
                    UNWIND $terms AS term
                    MERGE (t:Term {text: term.text, type: term.type})
                    MERGE (c)-[:HAS_TERM]->(t)
                    """,
                    params
                )
                
                logger.debug(f"Stored batch of {len(batch)} terms for chunk {chunk_id}")
            except Exception as e:
                logger.error(f"Error storing terms batch for chunk {chunk_id}: {str(e)}")
                raise
                
        logger.info(f"Successfully stored all terms for chunk {chunk_id}")
        
    def process_chunk(self, chunk_id: str, chunk_text: str) -> Tuple[List[str], List[str], List[str]]:
        """Process a text chunk, extract n-grams, and store in Neo4j
        
        Args:
            chunk_id: Chunk identifier
            chunk_text: Chunk text content
            
        Returns:
            Tuple[List[str], List[str], List[str]]: Tuple of (unigrams, bigrams, trigrams)
        """
        logger.info(f"Processing chunk {chunk_id}")
        
        # Extract n-grams
        unigrams, bigrams, trigrams = self.extract_ngrams(chunk_text)
        
        # Store in Neo4j
        self.store_terms_for_chunk(chunk_id, unigrams, bigrams, trigrams)
        
        return unigrams, bigrams, trigrams

    def store_video_segments_in_neo4j(self, video_id: str, segments: List[dict]) -> None:
        """
        Stores video segments in Neo4j and establishes BEFORE/AFTER relationships
        between consecutive segments.

        Args:
            video_id (str): Identifier for the parent video.
            segments (List[dict]): A list of segment data. Each dictionary should have
                                   at least 'id' (unique segment identifier),
                                   'sequence_number' (for ordering), and other properties like
                                   'start_time', 'end_time', 'frame_paths'.
        """
        if not segments: # Modified this: if not segments and not self.neo4j.run_query("MATCH (v:Video {id: $video_id}) RETURN v", {"video_id": video_id}):
            logger.info(f"No segments provided for video {video_id}. Nothing to store unless Video node needs creation.")
            # Original logic: if not segments: return. Now, ensure Video node is created.

        logger.info(f"Processing {len(segments)} segments for video {video_id} in Neo4j.") # This log might be confusing if segments is empty

        # Create Video node if it doesn't exist (or match it)
        # This should be done even if there are no segments, to represent the video's existence.
        self.neo4j.run_query(
            "MERGE (v:Video {id: $video_id}) RETURN v",
            {"video_id": video_id}
        )

        if not segments: # Check for empty segments after creating the Video node
            logger.info(f"No segments were provided for video {video_id}. Only Video node was ensured.")
            return

        # Ensure segments are sorted by sequence_number if not already
        # For this example, we assume the input list 'segments' is already sorted.
        # If not, they should be sorted: sorted_segments = sorted(segments, key=lambda x: x['sequence_number'])

        for i, segment_data in enumerate(segments):
            segment_id = segment_data.get("id")
            if not segment_id:
                logger.warning(f"Segment at index {i} for video {video_id} is missing an 'id'. Skipping.")
                continue

            node_properties = {
                "id": segment_id,
                "video_id": video_id,
                "start_time": segment_data.get("start_time"),
                "end_time": segment_data.get("end_time"),
                "sequence_number": segment_data.get("sequence_number", i),
            }

            query = """
            MATCH (vid:Video {id: $video_id})
            MERGE (s:VideoSegment {id: $segment_id})
            ON CREATE SET s = $props
            ON MATCH SET s += $props
            MERGE (vid)-[:HAS_SEGMENT]->(s)
            """
            self.neo4j.run_query(query, {"video_id": video_id, "segment_id": segment_id, "props": node_properties})
            logger.debug(f"Stored VideoSegment node: {segment_id}")

            if i > 0:
                prev_segment_data = segments[i-1]
                prev_segment_id = prev_segment_data.get("id")
                if prev_segment_id:
                    rel_query_corrected = """
                    MATCH (s1:VideoSegment {id: $prev_id}) // Previous segment
                    MATCH (s2:VideoSegment {id: $curr_id}) // Current segment
                    MERGE (s1)-[:BEFORE]->(s2)
                    MERGE (s2)-[:AFTER]->(s1)
                    """
                    self.neo4j.run_query(rel_query_corrected, {"prev_id": prev_segment_id, "curr_id": segment_id})
                    logger.debug(f"Linked {prev_segment_id} -BEFORE-> {segment_id}")
                    logger.debug(f"Linked {segment_id} -AFTER-> {prev_segment_id}")

        logger.info(f"Successfully processed and linked segments for video {video_id}.")


try:
    # Check if pyspark and spark-nlp are available
    import pyspark
    import sparknlp
    
    SPARK_AVAILABLE = True
    logger.info("PySpark and Spark NLP are available. Spark-based processing enabled.")
    
    class SparkNLPGraphBuilder(NLPGraphBuilder):
        """NLP Graph Builder using Spark NLP for more scalable processing"""
        
        def __init__(self, neo4j_conn=None, remove_stopwords=True):
            """Initialize SparkNLPGraphBuilder
            
            Args:
                neo4j_conn: Neo4j connection instance
                remove_stopwords: Whether to remove stopwords from unigrams
            """
            super().__init__(neo4j_conn, remove_stopwords)
            
            # Initialize Spark session with Spark NLP
            try:
                self.spark = sparknlp.start()
            except Exception as e:
                logger.error(f"Failed to start Spark NLP session: {e}")
                logger.warning("SparkNLPGraphBuilder will not be fully functional.")
                self.spark = None
                return

            if not self.spark:
                return

            # Import required Spark NLP components
            from sparknlp.base import DocumentAssembler, Finisher
            from sparknlp.annotator import Tokenizer, Normalizer, NGramGenerator
            from pyspark.ml import Pipeline
            
            # Define Spark NLP pipeline
            document_assembler = DocumentAssembler().setInputCol("text").setOutputCol("document")
            tokenizer = Tokenizer().setInputCols(["document"]).setOutputCol("token")
            normalizer = Normalizer().setInputCols(["token"]).setOutputCol("normalized").setLowercase(True)
            
            # Generate unigrams (normalized tokens), bigrams, trigrams
            bigram_generator = NGramGenerator().setInputCols(["normalized"]).setOutputCol("bigrams").setN(2)
            trigram_generator = NGramGenerator().setInputCols(["normalized"]).setOutputCol("trigrams").setN(3)
            
            # Finisher to output results as Python lists
            finisher_unigram = Finisher().setInputCols(["normalized"]).setOutputCols(["tokens_out"])
            finisher_bigram = Finisher().setInputCols(["bigrams"]).setOutputCols(["bigrams_out"])
            finisher_trigram = Finisher().setInputCols(["trigrams"]).setOutputCols(["trigrams_out"])
            
            # Create pipeline
            self.pipeline = Pipeline(stages=[
                document_assembler, tokenizer, normalizer,
                bigram_generator, trigram_generator,
                finisher_unigram, finisher_bigram, finisher_trigram
            ])
            
            # Fit the pipeline (this creates a model that can transform data)
            if self.spark:
                self.model = self.pipeline.fit(self.spark.createDataFrame([("",)], ["text"]))
                logger.info("Spark NLP pipeline initialized")
            else:
                self.model = None
                logger.warning("Spark NLP model not fitted as Spark session is unavailable.")

        def process_chunks(self, chunks: List[Tuple[str, str]]) -> List[Tuple[str, List[str], List[str], List[str]]]:
            """Process multiple chunks with Spark NLP
            
            Args:
                chunks: List of (chunk_id, chunk_text) tuples
                
            Returns:
                List[Tuple[str, List[str], List[str], List[str]]]: List of (chunk_id, unigrams, bigrams, trigrams) tuples
            """
            if not self.spark or not self.model:
                logger.error("Spark session or model not available. Cannot process chunks with Spark.")
                logger.warning("Falling back to sequential NLTK processing for text chunks due to Spark issue.")
                processed_chunks_fallback = []
                for chunk_id, chunk_text in chunks:
                    unigrams, bigrams, trigrams = super().extract_ngrams(chunk_text)
                    super().store_terms_for_chunk(chunk_id, unigrams, bigrams, trigrams)
                    processed_chunks_fallback.append((chunk_id, unigrams, bigrams, trigrams))
                return processed_chunks_fallback

            logger.info(f"Processing {len(chunks)} chunks with Spark NLP")
            
            # Create Spark DataFrame from chunks
            chunk_df = self.spark.createDataFrame([(cid, txt) for cid, txt in chunks], ["id", "text"])
            
            # Apply the pipeline to transform the data
            result_df = self.model.transform(chunk_df)
            
            # Collect results and process
            processed_chunks = []
            for row in result_df.select("id", "tokens_out", "bigrams_out", "trigrams_out").collect():
                chunk_id = row["id"]
                unigrams = row["tokens_out"]
                bigrams = row["bigrams_out"]
                trigrams = row["trigrams_out"]
                
                # Remove stopwords if required
                if self.remove_stopwords:
                    unigrams = [t for t in unigrams if t not in STOPWORDS]
                
                processed_chunks.append((chunk_id, unigrams, bigrams, trigrams))
                
                # Store in Neo4j
                self.store_terms_for_chunk(chunk_id, unigrams, bigrams, trigrams)
                
            logger.info(f"Successfully processed {len(chunks)} chunks with Spark NLP")
            return processed_chunks
            
except ImportError:
    SPARK_AVAILABLE = False
    logger.info("PySpark and/or Spark NLP not available. Using NLTK for processing.")
    SparkNLPGraphBuilder = None

# Convenience functions

def extract_ngrams(text: str) -> Tuple[List[str], List[str], List[str]]:
    """Extract unigrams, bigrams, and trigrams from text
    
    Args:
        text: Input text
        
    Returns:
        Tuple[List[str], List[str], List[str]]: Tuple of (unigrams, bigrams, trigrams)
    """
    builder = NLPGraphBuilder()
    return builder.extract_ngrams(text)
    
def store_terms_for_chunk(chunk_id: str, unigrams: List[str], 
                      bigrams: List[str], trigrams: List[str]) -> None:
    """Store terms (n-grams) and connect them to their chunk in Neo4j
    
    Args:
        chunk_id: Chunk identifier
        unigrams: List of unigrams (single tokens)
        bigrams: List of bigrams (two-word phrases)
        trigrams: List of trigrams (three-word phrases)
    """
    builder = NLPGraphBuilder()
    builder.store_terms_for_chunk(chunk_id, unigrams, bigrams, trigrams)
    
def process_chunk(chunk_id: str, chunk_text: str) -> Tuple[List[str], List[str], List[str]]:
    """Process a text chunk, extract n-grams, and store in Neo4j
    
    Args:
        chunk_id: Chunk identifier
        chunk_text: Chunk text content
        
    Returns:
        Tuple[List[str], List[str], List[str]]: Tuple of (unigrams, bigrams, trigrams)
    """
    builder = NLPGraphBuilder()
    return builder.process_chunk(chunk_id, chunk_text)
    
def process_chunks_with_spark(chunks: List[Tuple[str, str]]) -> List[Tuple[str, List[str], List[str], List[str]]]:
    """Process multiple chunks with Spark NLP
    
    Args:
        chunks: List of (chunk_id, chunk_text) tuples
        
    Returns:
        List[Tuple[str, List[str], List[str], List[str]]]: List of (chunk_id, unigrams, bigrams, trigrams) tuples
    """
    if not SPARK_AVAILABLE:
        logger.warning("Spark NLP is not available. Falling back to sequential processing.")
        results = []
        for chunk_id, chunk_text in chunks:
            unigrams, bigrams, trigrams = process_chunk(chunk_id, chunk_text)
            results.append((chunk_id, unigrams, bigrams, trigrams))
        return results
        
    builder = SparkNLPGraphBuilder()
    return builder.process_chunks(chunks)
    
if __name__ == "__main__":
    # Demo with example text
    example_text = """
    Hugging Face, Inc. is an American company that develops tools for building applications using machine learning.
    It was founded in 2016 and its headquarters is in New York City. The company is known for its libraries in natural
    language processing (NLP) and its platform that allows users to share machine learning models and datasets.
    """
    
    print("Extracting n-grams from example text...")
    unigrams, bigrams, trigrams = extract_ngrams(example_text)
    print(f"Extracted {len(unigrams)} unigrams, {len(bigrams)} bigrams, {len(trigrams)} trigrams")
    print("Sample unigrams:", unigrams[:5])
    print("Sample bigrams:", bigrams[:5])
    print("Sample trigrams:", trigrams[:5])

    # --- Mock Neo4j Connection and Test for store_video_segments_in_neo4j ---
    class MockNeo4jConnection:
        def __init__(self):
            self.queries_run = []

        def run_query(self, query: str, params: Optional[dict] = None) -> None:
            logger.info(f"MOCK NEO4J: Running query: {query.strip()} with params: {params}")
            self.queries_run.append({"query": query, "params": params})

        def close(self):
            logger.info("MOCK NEO4J: Connection closed.")

    logger.info("\n--- Testing store_video_segments_in_neo4j with Mock Connection ---")
    mock_conn = MockNeo4jConnection()

    builder_opts = {"neo4j_conn": mock_conn, "remove_stopwords": True}
    video_graph_builder = NLPGraphBuilder(**builder_opts)

    sample_video_id = "video123"
    sample_segments = [
        {"id": "seg_001", "sequence_number": 0, "start_time": 0.0, "end_time": 5.0, "frame_paths": ["f1.jpg", "f2.jpg"]},
        {"id": "seg_002", "sequence_number": 1, "start_time": 5.0, "end_time": 10.0, "frame_paths": ["f3.jpg", "f4.jpg"]},
        {"id": "seg_003", "sequence_number": 2, "start_time": 10.0, "end_time": 15.0, "frame_paths": ["f5.jpg", "f6.jpg"]},
    ]

    video_graph_builder.store_video_segments_in_neo4j(sample_video_id, sample_segments)

    logger.info("\n--- Queries Run by store_video_segments_in_neo4j ---")
    for i, item in enumerate(mock_conn.queries_run):
        logger.info(f"Query {i+1}:")
        logger.info(f"  Cypher: {item['query'].strip()}")
        logger.info(f"  Params: {item['params']}")

    logger.info("\n--- End of NLP Graph Demo ---")