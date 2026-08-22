import os
from typing import Optional

try:
    from pydantic_settings import BaseSettings
    class Settings(BaseSettings):
        APP_NAME: str = "Voice RAG AI Assistant"
        VERSION: str = "1.0.0"
        OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", "")
        GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY", "")
        GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY", "")
        DEFAULT_CHUNKING_STRATEGY: str = "recursive"
        DEFAULT_CHUNK_SIZE: int = 500
        DEFAULT_CHUNK_OVERLAP: int = 50
        DEFAULT_TOP_K: int = 4
        STORAGE_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        DOCUMENTS_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "documents")
        INDEX_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "vector_store")
        class Config:
            env_file = ".env"
            extra = "ignore"
except ImportError:
    from pydantic import BaseModel
    class Settings(BaseModel):
        APP_NAME: str = "Voice RAG AI Assistant"
        VERSION: str = "1.0.0"
        OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", "")
        GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY", "")
        GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY", "")
        DEFAULT_CHUNKING_STRATEGY: str = "recursive"
        DEFAULT_CHUNK_SIZE: int = 500
        DEFAULT_CHUNK_OVERLAP: int = 50
        DEFAULT_TOP_K: int = 4
        STORAGE_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        DOCUMENTS_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "documents")
        INDEX_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "vector_store")

settings = Settings()

os.makedirs(settings.STORAGE_DIR, exist_ok=True)
os.makedirs(settings.DOCUMENTS_DIR, exist_ok=True)
os.makedirs(settings.INDEX_DIR, exist_ok=True)
