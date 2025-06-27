import os

class RAGConfig:
    # ChromaDB settings
    CHROMA_PERSIST_DIRECTORY = "./chroma_db"
    COLLECTION_NAME = "lab_documents"
    
    # Text splitting settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Search settings
    DEFAULT_SEARCH_RESULTS = 5
    SIMILARITY_THRESHOLD = 0.7
    
    # Supported file types
    SUPPORTED_FILE_TYPES = ['.pdf', '.docx', '.doc', '.txt']
    
    # Embedding model
    EMBEDDING_MODEL = "models/embedding-001"
    
    @classmethod
    def get_chroma_settings(cls):
        return {
            "persist_directory": cls.CHROMA_PERSIST_DIRECTORY,
            "collection_name": cls.COLLECTION_NAME
        }