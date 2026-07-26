import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # groq llm
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    llm_model: str = os.getenv(
        "LLM_MODEL", "llama-3.1-8b-instant"
    )

    # embedding handled by chromadb internally
    # no separate embedding model needed
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "default"
    )

    # chromadb
    chroma_persist_dir: str = os.getenv(
        "CHROMA_PERSIST_DIR", "./data/chroma_db"
    )
    chroma_collection_name: str = os.getenv(
        "CHROMA_COLLECTION_NAME", "agentic_ai_rag"
    )

    # retrieval
    top_k_results: int = int(
        os.getenv("TOP_K_RESULTS", "5")
    )
    relevance_threshold: float = float(
        os.getenv("RELEVANCE_THRESHOLD", "0.30")
    )

    # chunking
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "700"))
    chunk_overlap: int = int(
        os.getenv("CHUNK_OVERLAP", "100")
    )

    # pdf
    pdf_url: str = os.getenv(
        "PDF_URL",
        "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf",
    )
    pdf_local_path: str = os.getenv(
        "PDF_LOCAL_PATH", "./data/Ebook-Agentic-AI.pdf"
    )

    # api
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))

    def validate(self):
        if not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. "
                "Get free key at https://console.groq.com"
            )


settings = Settings()
