"""
config.py - Configuration management for AI Teacher Malaysia RAG System
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    # Zilliz Cloud / Milvus
    ZILLIZ_URI: str = os.getenv("ZILLIZ_URI", "")
    ZILLIZ_TOKEN: str = os.getenv("ZILLIZ_TOKEN", "")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "malaysia_kssm_math")

    # Groq API
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # PDF files
    PDF_DIR: str = os.getenv("PDF_DIR", "./")

    # Embedding model
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
    )
    EMBEDDING_DIM: int = 384  # dimension for MiniLM-L12-v2

    # Text chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # Retrieval
    TOP_K: int = int(os.getenv("TOP_K", "5"))

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Form level mapping from filename
    FORM_LEVEL_MAP: dict = {
        "T1": "Form 1 (Tingkatan 1)",
        "T2": "Form 2 (Tingkatan 2)",
        "T3": "Form 3 (Tingkatan 3)",
        "T4": "Form 4 (Tingkatan 4)",
        "T5": "Form 5 (Tingkatan 5)",
    }

    @classmethod
    def validate(cls):
        """Validate that required configuration values are set."""
        errors = []
        if not cls.ZILLIZ_URI:
            errors.append("ZILLIZ_URI is not set in .env")
        if not cls.ZILLIZ_TOKEN:
            errors.append("ZILLIZ_TOKEN is not set in .env")
        if not cls.GROQ_API_KEY:
            errors.append("GROQ_API_KEY is not set in .env")
        if errors:
            raise ValueError(
                "Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors)
            )


config = Config()
