"""
backend/utils/config.py
Central configuration loader using environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # LLM
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))

    # Embeddings
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector Store
    VECTORSTORE_PATH: str = os.getenv("VECTORSTORE_PATH", "./data/vectorstore")
    VECTORSTORE_TYPE: str = os.getenv("VECTORSTORE_TYPE", "faiss")

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # Retrieval
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    RETRIEVAL_SCORE_THRESHOLD: float = float(
        os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.3")
    )

    # Backend
    BACKEND_HOST: str = os.getenv("BACKEND_HOST", "0.0.0.0")
    BACKEND_PORT: int = int(os.getenv("BACKEND_PORT", "8000"))

    # Guardrails
    ENABLE_GUARDRAILS: bool = os.getenv("ENABLE_GUARDRAILS", "true").lower() == "true"
    GUARDRAILS_CONFIG_PATH: str = os.getenv(
        "GUARDRAILS_CONFIG_PATH", "./backend/guardrails_config"
    )

    @classmethod
    def validate(cls):
        if not cls.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. Please set it in your .env file."
            )


config = Config()
