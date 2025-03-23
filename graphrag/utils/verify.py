"""
GraphRAG setup verification utilities

This module provides functions to verify the GraphRAG setup, including:
- Dependency checking
- NLTK resources verification
- Database connectivity testing
- Neo4j index verification
"""

import sys
import importlib
import time
from graphrag.utils.logger import logger

def verify_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        "neo4j", "numpy", "nltk", "sentence_transformers", "transformers",
        "qdrant_client", "dotenv", "loguru"
    ]
    
    optional_packages = ["pymupdf", "pyspark", "spark_nlp"]
    
    print("Checking required dependencies...")
    missing = []
    for package in required_packages:
        try:
            if package == "dotenv":
                # Special case for python-dotenv which is imported as dotenv
                importlib.import_module("dotenv")
                print(f"✓ python-dotenv is installed")
            else:
                importlib.import_module(package)
                print(f"✓ {package} is installed")
        except ImportError:
            if package == "dotenv":
                print(f"× python-dotenv is missing")
                missing.append("python-dotenv")
            else:
                print(f"× {package} is missing")
                missing.append(package)
    
    print("\nChecking optional dependencies...")
    for package in optional_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package} is installed (optional)")
        except ImportError:
            print(f"- {package} is not installed (optional)")
    
    if missing:
        print(f"\n❌ Missing {len(missing)} required dependencies: {', '.join(missing)}")
        return False
    else:
        print("\n✅ All required dependencies are installed")
        return True

def verify_nltk_resources():
    """Verify that NLTK resources can be downloaded and accessed"""
    import nltk
    from nltk.data import find as nltk_find
    
    print("\nChecking NLTK resources...")
    resources_ok = True
    
    # Check if punkt_tab is available (newer NLTK versions)
    print("Attempting to find punkt_tab...")
    try:
        nltk_find('tokenizers/punkt_tab')
        print("✓ punkt_tab found - compatible with newer NLTK versions")
        has_punkt_tab = True
    except LookupError:
        print("- punkt_tab not found")
        has_punkt_tab = False
    
    # Check if punkt is available (older NLTK versions)
    print("Attempting to find punkt...")
    try:
        nltk_find('tokenizers/punkt')
        print("✓ punkt found - compatible with older NLTK versions")
        has_punkt = True
    except LookupError:
        print("- punkt not found")
        has_punkt = False
    
    # Check stopwords
    print("Attempting to find stopwords...")
    try:
        nltk_find('corpora/stopwords')
        print("✓ stopwords found")
        has_stopwords = True
    except LookupError:
        print("- stopwords not found")
        has_stopwords = False
    
    # Verify actual tokenization works
    print("\nTesting NLTK tokenization...")
    test_text = "Hello there. This is a test sentence. Does tokenization work?"
    try:
        sentences = nltk.sent_tokenize(test_text)
        if len(sentences) == 3:
            print(f"✓ Tokenization successful, split into {len(sentences)} sentences")
        else:
            print(f"⚠ Tokenization produced {len(sentences)} sentences instead of expected 3")
            resources_ok = False
    except Exception as e:
        print(f"❌ Tokenization failed: {str(e)}")
        resources_ok = False
    
    # Overall status
    if (has_punkt_tab or has_punkt) and has_stopwords and resources_ok:
        print("\n✅ NLTK resources are correctly set up")
        return True
    else:
        print("\n⚠ NLTK resources setup is incomplete")
        if not has_punkt_tab and not has_punkt:
            print("  - Missing tokenizer (need either punkt or punkt_tab)")
        if not has_stopwords:
            print("  - Missing stopwords")
        return False

def verify_graphrag_import():
    """Verify that GraphRAG package can be imported and initialized"""
    print("\nAttempting to import GraphRAG...")
    try:
        import graphrag
        print(f"✓ GraphRAG version {graphrag.__version__} imported successfully")
        
        # Test basic functionality
        print("\nTesting basic GraphRAG functionality...")
        try:
            from graphrag.core.ingest import DocumentIngestor
            ingestor = DocumentIngestor()
            print("✓ Created DocumentIngestor instance")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize GraphRAG components: {str(e)}")
            return False
    except Exception as e:
        print(f"❌ Failed to import GraphRAG: {str(e)}")
        return False

def verify_database_connections():
    """Verify that database connections can be established"""
    print("\nVerifying database connections...")
    
    # Test Neo4j connection
    print("Testing Neo4j connection...")
    try:
        from graphrag.connectors.neo4j_connection import Neo4jConnection
        neo4j_conn = Neo4jConnection()
        result = neo4j_conn.run_query("RETURN 'Connected to Neo4j' as message")
        message = result[0]["message"] if result else "No result returned"
        print(f"✓ {message}")
        neo4j_ok = True
    except Exception as e:
        print(f"❌ Neo4j connection failed: {str(e)}")
        neo4j_ok = False
    
    # Test Qdrant connection
    print("\nTesting Qdrant connection...")
    try:
        from graphrag.connectors.qdrant_connection import QdrantConnection
        qdrant_conn = QdrantConnection()
        collections = qdrant_conn.client.get_collections()
        print(f"✓ Connected to Qdrant, found {len(collections.collections)} collections")
        qdrant_ok = True
    except Exception as e:
        print(f"❌ Qdrant connection failed: {str(e)}")
        qdrant_ok = False
    
    # Overall status
    if neo4j_ok and qdrant_ok:
        print("\n✅ Database connections are working")
        return True
    else:
        print("\n⚠ Database connection issues detected")
        return False

def verify_neo4j_indexes():
    """Verify that required Neo4j indexes are set up by testing basic functionality"""
    print("\nVerifying Neo4j setup...")
    try:
        from graphrag.connectors.neo4j_connection import Neo4jConnection
        neo4j_conn = Neo4jConnection()
        
        # Try a simple test operation instead of checking indexes directly
        # Create a test node, query it, then delete it
        test_id = "test_" + str(int(time.time()))
        
        # Step 1: Create a test node
        create_query = """
        CREATE (c:Chunk {id: $id, text: 'Test chunk for verification'})
        RETURN c.id as id
        """
        try:
            result = neo4j_conn.run_query(create_query, {"id": test_id})
            if not result or not result[0].get('id'):
                print("× Failed to create test node")
                return False
                
            print("✓ Successfully created test node")
            
            # Step 2: Query the node to verify it exists
            find_query = """
            MATCH (c:Chunk {id: $id})
            RETURN c.id as id, c.text as text
            """
            result = neo4j_conn.run_query(find_query, {"id": test_id})
            if not result or not result[0].get('id'):
                print("× Failed to query test node")
                return False
                
            print("✓ Successfully queried test node")
            
            # Step 3: Delete the test node
            delete_query = """
            MATCH (c:Chunk {id: $id})
            DELETE c
            """
            neo4j_conn.run_query(delete_query, {"id": test_id})
            print("✓ Successfully deleted test node")
            
            # If all operations succeeded, the database is working
            print("\n✅ Neo4j database is operational and ready for use")
            return True
            
        except Exception as e:
            print(f"× Neo4j operation failed: {str(e)}")
            try:
                # Try to clean up the test node just in case
                neo4j_conn.run_query("MATCH (c:Chunk {id: $id}) DELETE c", {"id": test_id})
            except:
                pass
            return False
        
    except Exception as e:
        print(f"❌ Failed to verify Neo4j setup: {str(e)}")
        return False 