import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # groq llm settings
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

    # embedding model runs locally no key needed
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
    )

    # chromadb local storage paths
    chroma_persist_dir: str = os.getenv(
        "CHROMA_PERSIST_DIR", "./data/chroma_db"
    )
    chroma_collection_name: str = os.getenv(
        "CHROMA_COLLECTION_NAME", "agentic_ai_rag"
    )

    # retrieval behaviour
    top_k_results: int = int(os.getenv("TOP_K_RESULTS", "5"))
    relevance_threshold: float = float(
        os.getenv("RELEVANCE_THRESHOLD", "0.70")
    )

    # chunking settings
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "700"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    # pdf source
    pdf_url: str = os.getenv(
        "PDF_URL",
        "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf",
    )
    pdf_local_path: str = os.getenv(
        "PDF_LOCAL_PATH", "./data/Ebook-Agentic-AI.pdf"
    )

    # api server
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))

    def validate(self):
        if not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. "
                "Get a free key at https://console.groq.com"
            )


# single instance used across the whole project
settings = Settings()
