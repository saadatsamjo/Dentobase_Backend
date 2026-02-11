# scripts/ingest_documents.py
#  to run the script, run the following command:
#  python scripts/ingest_documents.py

"""
Document Ingestion Script
Loads PDF, splits into chunks, embeds, and stores in ChromaDB
"""
import sys
import logging
from pathlib import Path
import shutil
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from app.RAGsystem.embeddings import embedding_provider
from config.ragconfig import rag_settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ingest_pdf():
    """
    Load PDF, split into chunks, and store in ChromaDB.
    """
    start_time = time.time()
    # Check if PDF exists
    pdf_path = Path(rag_settings.PDF_PATH)
    if not pdf_path.exists():
        print("=======================================================================\n")
        logger.error(f"❌ PDF not found at: {pdf_path}")
        logger.error(f"   Please place your clinical guidelines PDF at: {pdf_path}")
        print("=======================================================================\n")
        return False
    
    print("=======================================================================\n")
    logger.info(f"📄 Loading PDF: {pdf_path}")
    logger.info(f"   File size: {pdf_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    
    # Load PDF
    try:
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()
        logger.info(f"✓ Loaded {len(documents)} pages from PDF")
        print("=======================================================================\n")
    except Exception as e:
        logger.error(f"❌ Failed to load PDF: {e}")
        print("=======================================================================\n")
        return False
    
    # Split into chunks
    print("=======================================================================\n")
    logger.info(f"✂️  Splitting into chunks...")
    logger.info(f"   Chunk size: {rag_settings.CHUNK_SIZE}")
    logger.info(f"   Chunk overlap: {rag_settings.CHUNK_OVERLAP}")
    
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=rag_settings.CHUNK_SIZE,
        chunk_overlap=rag_settings.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    logger.info(f"✓ Created {len(chunks)} chunks")
    print("=======================================================================\n")
    
    # Add metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = i
        chunk.metadata['source'] = pdf_path.name
        # Preserve page number from PyPDF
        if 'page' not in chunk.metadata:
            chunk.metadata['page'] = 0
    
    # Get embeddings
    print("=======================================================================\n")
    logger.info(f"🔤 Initializing embeddings...")
    logger.info(f"   Provider: {rag_settings.EMBEDDING_PROVIDER}")
    logger.info(f"   Model: {rag_settings.current_embedding_model}")
    
    embeddings = embedding_provider.get_embeddings()
    
    # Create/update ChromaDB
    logger.info(f"💾 Creating vector store...")
    logger.info(f"   Directory: {rag_settings.PERSIST_DIR}")
    print("=======================================================================\n")

    # Remove old ChromaDB if it exists
    if Path(rag_settings.PERSIST_DIR).exists():
        print("=======================================================================\n")
        logger.info(f"   Deleting old ChromaDB at: {rag_settings.PERSIST_DIR}")
        shutil.rmtree(rag_settings.PERSIST_DIR)
        logger.info(f"   Old ChromaDB deleted.")

    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=rag_settings.PERSIST_DIR
        )
        logger.info(f"✓ Vector store created successfully")
        
        # Verify
        collection = vectorstore._collection
        count = collection.count()
        logger.info(f"✓ Stored {count} chunks in ChromaDB")
        print("=======================================================================\n")
        
        # Test retrieval
        print("=======================================================================\n")
        logger.info(f"\n🧪 Testing retrieval...")
        test_query = "treatment for pulpitis"
        results = vectorstore.similarity_search(test_query, k=3)
        
        logger.info(f"✓ Test query: '{test_query}'")
        logger.info(f"✓ Retrieved {len(results)} chunks")
        for i, doc in enumerate(results, 1):
            page = doc.metadata.get('page', 'Unknown')
            preview = doc.page_content[:100].replace('\n', ' ')
            logger.info(f"   [{i}] Page {page}: {preview}...")
            print("=======================================================================\n")
        
        
        print("=======================================================================\n")
        logger.info(f"\n✅ INGESTION COMPLETE!")
        logger.info(f"   Total chunks: {count}")
        logger.info(f"   Persist directory: {rag_settings.PERSIST_DIR}")
        logger.info(f"   Ready for retrieval!")
        
        end_time = time.time()
        duration = end_time - start_time
        num_pages = len(documents)
        logger.info(f"⏱️ Ingestion took {duration:.2f} seconds for {num_pages} pages.")
        print("=======================================================================\n")
        
        return True
        
    except Exception as e:
        print("=======================================================================\n")
        logger.error(f"❌ Failed to create vector store: {e}", exc_info=True)
        print("=======================================================================\n")
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("   DENTOBASE - DOCUMENT INGESTION")
    print("="*60 + "\n")
    print("=======================================================================\n")
    
    
    success = ingest_pdf()
    
    if success:
        print("\n" + "="*60)
        print("   ✅ SUCCESS - Knowledge base ready!")
        print("="*60 + "\n")
        print("=======================================================================\n")
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("   ❌ FAILED - Check errors above")
        print("="*60 + "\n")
        print("=======================================================================\n")
        sys.exit(1)