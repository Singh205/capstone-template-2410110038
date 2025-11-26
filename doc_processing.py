# Day 1: Document Processing & Vector Store Setup
# Install required packages:
# pip install langchain langchain-community langchain-openai pypdf chromadb python-dotenv

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from typing import List
from langchain_core.documents import Document


load_dotenv()

class DocumentProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
    def load_document(self, file_path: str) -> List[Document]:
        """Load document based on file type"""
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        else:
            raise ValueError("Only PDF and TXT files are supported")
        
        documents = loader.load()
        print(f"✓ Loaded {len(documents)} page(s) from {file_path}")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        chunks = self.text_splitter.split_documents(documents)
        print(f"✓ Split into {len(chunks)} chunks")
        return chunks
    
    def create_vectorstore(self, chunks: List[Document], collection_name: str = "contracts") -> Chroma:
        """Create vector store from document chunks"""
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory="./chroma_db"
        )
        print(f"✓ Created vector store with {len(chunks)} chunks")
        return vectorstore
    
    def search_similar(self, vectorstore: Chroma, query: str, k: int = 3) -> List[Document]:
        """Search for similar documents"""
        results = vectorstore.similarity_search(query, k=k)
        return results


# Test the setup
if __name__ == "__main__":
    print("=== Day 1: Document Processing & Vector Store Test ===\n")
    
    # Create sample contract for testing
    sample_contract = """
    SERVICE AGREEMENT
    
    1. TERM: This agreement shall commence on January 1, 2024 and continue for 12 months.
    The agreement automatically renews unless either party provides 90 days written notice.
    
    2. PAYMENT TERMS: Client agrees to pay $5,000 monthly. Late payments incur 5% penalty per month.
    All fees are non-refundable under any circumstances.
    
    3. LIABILITY: Provider's liability is limited to the amount paid in the last 3 months.
    Provider is not liable for any indirect, consequential, or punitive damages.
    
    4. TERMINATION: Either party may terminate with 30 days notice. Upon termination,
    client must pay all remaining months in the contract term.
    
    5. DISPUTE RESOLUTION: All disputes must be resolved through binding arbitration in
    Provider's jurisdiction. Client waives right to class action lawsuits.
    
    6. DATA USAGE: Provider may use client data for any purpose including sharing with
    third parties without notification.
    
    7. MODIFICATION: Provider may modify terms at any time without notice. Continued use
    constitutes acceptance of new terms.
    """
    
    # Save sample contract
    os.makedirs("test_contracts", exist_ok=True)
    with open("test_contracts/sample_contract.txt", "w") as f:
        f.write(sample_contract)
    
    # Initialize processor
    processor = DocumentProcessor()
    
    # Load document
    documents = processor.load_document("test_contracts/sample_contract.txt")
    
    # Split into chunks
    chunks = processor.split_documents(documents)
    
    # Create vector store
    vectorstore = processor.create_vectorstore(chunks)
    
    # Test semantic search
    print("\n--- Testing Semantic Search ---")
    test_queries = [
        "What are the payment terms?",
        "What happens if I want to cancel?",
        "Who is liable for damages?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = processor.search_similar(vectorstore, query, k=1)
        print(f"Result: {results[0].page_content[:200]}...")
    
    print("\n✓ Day 1 Complete! Vector store ready for Day 2.")