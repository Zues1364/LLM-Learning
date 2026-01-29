import sys
import os
import time
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils import FAISSVectorStore, process_pdf
from pathlib import Path

# Setup Console Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Determine the project root (assuming script is in <root>/test/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESOURCE_DIR = PROJECT_ROOT / "data" / "resources" / "pdfs"

BATCH_SIZE = 32  # Batch size for embedding generation if supported, or just chunk processing

def test_vector_ingestion():
    if not RESOURCE_DIR.exists():
        logger.error(f"Resource directory not found: {RESOURCE_DIR}")
        logger.info(f"Current CWD: {os.getcwd()}")
        return

    logger.info("=== STARTING VECTOR INGESTION TEST ===")
    
    # 1. Initialize Vector Store
    logger.info("Initializing FAISS Vector Store...")
    start_init = time.time()
    
    # Initialize implementation-specific embedder
    from utils import VietnameseEmbedder
    embedder = VietnameseEmbedder()
    
    # Start with empty store
    vector_store = FAISSVectorStore(documents=[], embedder=embedder)
    logger.info(f"Vector Store Initialized in {time.time() - start_init:.2f}s")

    # 2. List Files
    pdf_files = list(RESOURCE_DIR.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {RESOURCE_DIR}")

    total_chunks = 0
    
    # 3. Process Each File
    for i, pdf_path in enumerate(pdf_files, 1):
        try:
            logger.info(f"\n--- Processing File {i}/{len(pdf_files)}: {pdf_path.name} ---")
            
            # Step A: Text Extraction & Chunking
            t0 = time.time()
            docs = process_pdf(str(pdf_path))
            t1 = time.time()
            logger.info(f"-> Extracted {len(docs)} chunks in {t1 - t0:.2f}s")
            
            if not docs:
                logger.warning(f"No chunks found for {pdf_path.name}")
                continue

            # Step B: Vectorization (Embedding)
            # FAISSVectorStore.add_documents handles embedding generation
            logger.info(f"-> Generating Embeddings & Adding to Index (This is the slow part)...")
            t2 = time.time()
            
            # Use batching if list is huge? But add_documents usually handles list.
            # We'll just pass all docs for this file.
            vector_store.add_documents(docs)
            
            t3 = time.time()
            logger.info(f"-> Vectorization completed in {t3 - t2:.2f}s (Avg {(t3-t2)/len(docs):.4f}s per chunk)")
            
            total_chunks += len(docs)
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")

    logger.info("\n=== INGESTION SUMMARY ===")
    logger.info(f"Total Files Processed: {len(pdf_files)}")
    logger.info(f"Total Chunks Vectorized: {total_chunks}")
    logger.info(f"Total Time: {time.time() - start_init:.2f}s")
    
    # Optional: Save index
    logger.info("Saving Vector Store Index...")
    vector_store.save_local("data/faiss_index")
    logger.info("Index saved.")

if __name__ == "__main__":
    test_vector_ingestion()
