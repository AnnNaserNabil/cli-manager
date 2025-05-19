import os
from typing import List, Optional
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_generativeai import GoogleGenerativeAI

class ResearchAgent:
    def __init__(self, data_dir: str = "data", embeddings_model: str = "text-embedding-3-small"):
        """
        Initialize the research agent with document storage and retrieval capabilities.
        
        Args:
            data_dir: Directory to store research documents
            embeddings_model: Model to use for text embeddings
        """
        load_dotenv()
        self.data_dir = data_dir
        self.embeddings = GooglePalmEmbeddings(model="gemini-2-flash")
        self.db = None
        
    def ingest_documents(self, documents_dir: str):
        """
        Ingest documents from a directory and create embeddings.
        
        Args:
            documents_dir: Directory containing documents to ingest
        """
        # Load documents from directory
        loader = DirectoryLoader(documents_dir)
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        # Create vector store
        self.db = Chroma.from_documents(texts, self.embeddings)
        
    def research(self, query: str, k: int = 3) -> str:
        """
        Perform research on the ingested documents.
        
        Args:
            query: Research question
            k: Number of relevant documents to retrieve
            
        Returns:
            Research response
        """
        if not self.db:
            raise ValueError("No documents have been ingested yet. Please call ingest_documents first.")
            
        # Create retrieval chain
        retriever = self.db.as_retriever(search_kwargs={"k": k})
        llm = GoogleGenerativeAI(model="gemini-2.0-flash")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )
        
        return qa_chain.run(query)

    def add_document(self, document_path: str):
        """
        Add a single document to the knowledge base.
        """
        loader = DirectoryLoader(os.path.dirname(document_path), glob=os.path.basename(document_path))
        new_docs = loader.load()
        
        if not new_docs:
            raise ValueError(f"No document found at {document_path}")
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(new_docs)
        
        if not self.db:
            self.db = Chroma.from_documents(texts, self.embeddings)
        else:
            self.db.add_documents(texts)

if __name__ == "__main__":
    # Example usage
    agent = ResearchAgent()
    
    # Ingest some example documents
    if not os.path.exists("data"):
        os.makedirs("data")
    
    # Add example documents
    with open("data/example.txt", "w") as f:
        f.write("""This is an example document about AI research.
        It contains information about various AI topics including machine learning,
        natural language processing, and computer vision.""")
    
    agent.ingest_documents("data")
    
    # Perform research
    query = "What topics are covered in the documents?"
    result = agent.research(query)
    print(f"\nResearch Result:\n{result}")
