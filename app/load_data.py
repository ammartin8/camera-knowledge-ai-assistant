"""Load PDF documents into the vector store for RAG retrieval."""

import os
import sys
from pathlib import Path


def load_pdf_to_vectorstore(pdf_path: str, vector_store=None):
    """
    Load a PDF file into the vector store.
    
    Args:
        pdf_path: Path to the PDF file
        vector_store: Optional MinsearchVectorStore instance (creates new if not provided)
    
    Returns:
        Number of chunks loaded
    """
    # Ensure proper Python path setup for imports
    app_root = Path(__file__).parent.resolve()
    sys.path.insert(0, str(app_root.parent))  # Add project root
    sys.path.insert(0, str(app_root))  # Add app directory
    
    # Import here to avoid conflicts
    from src.pipeline import BasicPDFIngestor
    
    # Debug: Check if path exists
    if not os.path.exists(pdf_path):
        print(f"❌ PDF NOT FOUND: {pdf_path}")
        return 0
    
    print(f"📄 Loading PDF: {pdf_path}")
    print(f"   Vector store type: {type(vector_store).__name__ if vector_store else None}")
    
    # Initialize ingestor
    ingestor = BasicPDFIngestor(chunk_size=500, overlap=100)
    
    # Build index (extracts and chunks text from PDF)
    chunks = ingestor.build_index(pdf_path)
    print(f"   ✅ Extracted {len(chunks)} chunks")
    
    # Add chunks to vector store
    if chunks:
        from src.vector_store import Document
        
        # Debug: Log before adding
        print(f"   Adding {len(chunks)} documents to vector store...")
        
        documents = [
            Document(
                id=chunk.id,
                text=chunk.text,
                metadata={
                    "source": chunk.metadata.source,
                    "page": chunk.metadata.page,
                    "section_path": str(chunk.metadata.section_path),
                    "chunking_strategy": chunk.metadata.chunking_strategy,
                }
            )
            for chunk in chunks
        ]
        
        if vector_store:
            vector_store.add_documents(documents)
            print(f"   ✅ Added {len(documents)} documents to existing vector store")
        else:
            from src.vector_store import MinsearchVectorStore
            # Use persistence so data survives between app restarts
            persist_path = str(app_root / "data" / "vector_store_index.pkl")
            # Create data directory if it doesn't exist
            os.makedirs(os.path.dirname(persist_path), exist_ok=True)
            vs = MinsearchVectorStore(persist_path=persist_path)
            vs.add_documents(documents)
            print(f"   ✅ Added {len(documents)} documents to vector store (persisted to {persist_path})")
    
    return len(chunks)


def main():
    """Load all available PDFs into the vector store."""
    # Ensure proper Python path setup
    app_root = Path(__file__).parent.resolve()
    sys.path.insert(0, str(app_root.parent))  # Add project root
    sys.path.insert(0, str(app_root))  # Add app directory
    
    # Ensure data directory exists before loading
    persist_path = str(app_root / "data" / "vector_store_index.pkl")
    os.makedirs(os.path.dirname(persist_path), exist_ok=True)
    
    # Debug: Check current directory
    print(f"\n🔍 Current working directory: {os.getcwd()}\n")
    print(f"📂 Persist path: {persist_path}")
    print(f"   Exists: {os.path.exists(persist_path)}")
    
    pdf_paths = [
        os.path.join(os.path.dirname(__file__), "../data/c012.pdf"),  # Use relative path from app dir
        "/workspace/camera-knowledge-ai-assistant/app/tests/fixtures/canon_r6_mark_ii_sample.pdf",
    ]
    
    # Debug: Check each path
    print(f"🔍 Checking PDF paths:\n")
    for path in pdf_paths:
        exists = os.path.exists(path)
        print(f"  {'✅' if exists else '❌'} {path}")
    
    existing_files = [p for p in pdf_paths if os.path.exists(p)]
    
    if not existing_files:
        print("\n❌ No PDF files found!")
        return
    
    print(f"\n📂 Found {len(existing_files)} PDF file(s) to load:\n")
    
    total_chunks = 0
    for pdf_path in existing_files:
        chunks_loaded = load_pdf_to_vectorstore(pdf_path)
        total_chunks += chunks_loaded
        print()
    
    print(f"\n✅ Loaded {total_chunks} total chunks from {len(existing_files)} PDF(s)")


if __name__ == "__main__":
    main()
