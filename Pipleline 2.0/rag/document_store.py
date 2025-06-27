import os
import uuid
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
import PyPDF2
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class DocumentStore:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(
            name="lab_documents",
            metadata={"hnsw:space": "cosine"}
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv('GOOGLE_API_KEY')
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from various file formats"""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.pdf':
            return self._extract_from_pdf(file_path)
        elif file_path.suffix.lower() in ['.docx', '.doc']:
            return self._extract_from_docx(file_path)
        elif file_path.suffix.lower() == '.txt':
            return self._extract_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _extract_from_pdf(self, file_path: Path) -> str:
        text = ""
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _extract_from_docx(self, file_path: Path) -> str:
        doc = docx.Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    
    def _extract_from_txt(self, file_path: Path) -> str:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    
    def add_document(self, file_path: str, metadata: Dict[str, Any] = None) -> str:
        """Add a document to the vector store"""
        doc_id = str(uuid.uuid4())
        text = self.extract_text_from_file(file_path)
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Generate embeddings
        embeddings = self.embeddings.embed_documents(chunks)
        
        # Prepare metadata
        base_metadata = {
            "source": str(file_path),
            "doc_id": doc_id,
            "total_chunks": len(chunks)
        }
        if metadata:
            base_metadata.update(metadata)
        
        # Add to ChromaDB
        chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                "chunk_index": i,
                "chunk_text": chunk[:100] + "..." if len(chunk) > 100 else chunk
            })
            metadatas.append(chunk_metadata)
        
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=chunk_ids
        )
        
        return doc_id
    
    def search_documents(self, query: str, n_results: int = 5, filter_dict: Dict = None) -> List[Dict]:
        """Search for relevant documents"""
        query_embedding = self.embeddings.embed_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_dict
        )
        
        return [
            {
                "content": doc,
                "metadata": metadata,
                "score": 1 - distance  # Convert distance to similarity score
            }
            for doc, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0], 
                results["distances"][0]
            )
        ]
    
    def get_all_documents(self) -> List[Dict]:
        """Get all documents metadata"""
        results = self.collection.get()
        unique_docs = {}
        
        for metadata in results["metadatas"]:
            doc_id = metadata["doc_id"]
            if doc_id not in unique_docs:
                unique_docs[doc_id] = metadata
        
        return list(unique_docs.values())
    
    def delete_document(self, doc_id: str):
        """Delete a document and all its chunks"""
        # Get all chunk IDs for this document
        results = self.collection.get(where={"doc_id": doc_id})
        if results["ids"]:
            self.collection.delete(ids=results["ids"])