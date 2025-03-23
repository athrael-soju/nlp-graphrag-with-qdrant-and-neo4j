from setuptools import setup, find_packages
import os
import subprocess
import sys
from setuptools.command.develop import develop
from setuptools.command.install import install

def execute_nltk_download():
    """Execute the NLTK download script to get required tokenization resources"""
    try:
        # Try to download both punkt and punkt_tab to ensure compatibility
        # with different NLTK versions
        print("Downloading NLTK resources...")
        
        # Try punkt_tab first (newer NLTK versions)
        try:
            subprocess.check_call([sys.executable, '-c', 
                                  'import nltk; nltk.download("punkt_tab", quiet=True)'])
            print("Successfully downloaded punkt_tab resources.")
        except:
            # Fall back to punkt if punkt_tab fails
            print("punkt_tab not available, falling back to punkt...")
            subprocess.check_call([sys.executable, '-c', 
                                  'import nltk; nltk.download("punkt", quiet=True); nltk.download("stopwords", quiet=True)'])
            print("Successfully downloaded punkt and stopwords resources.")
            
    except Exception as e:
        print(f"NLTK data download failed: {str(e)}")
        print("You may need to manually download NLTK data by running:")
        print("  python -m nltk.downloader punkt stopwords")
        print("  or for newer NLTK versions:")
        print("  python -m nltk.downloader punkt_tab")

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        execute_nltk_download()

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        execute_nltk_download()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="graphrag",
    version="0.1.0",
    author="GraphRAG Team",
    author_email="example@example.com",
    description="Graph-based Retrieval Augmented Generation system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/athrael-soju/graphrag",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "neo4j",
        "numpy",
        "nltk",
        "sentence-transformers",
        "transformers",
        "peft",
        "pymupdf",  # Optional: for PDF support
        "pyspark",  # Optional: for Spark NLP
        "spark-nlp",  # Optional: for Spark NLP 
        "qdrant-client",
        "python-dotenv",  # For loading environment variables
        "loguru",  # For enhanced logging capabilities
    ],
    entry_points={
        "console_scripts": [
            "graphrag=graphrag.cli.main:main",
        ],
    },
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
) 