import os
import sys
import requests
import fitz
import chromadb
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# try new import first then fall back to old one
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# make sure we can import from project root
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings


def download_pdf():
    pdf_path = Path(settings.pdf_local_path)

    if pdf_path.exists():
        print(f"PDF already exists at {pdf_path} — skipping download")
        return str(pdf_path)

    print(f"Downloading PDF from {settings.pdf_url}")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(settings.pdf_url, timeout=60)
    response.raise_for_status()

    with open(pdf_path, "wb") as f:
        f.write(response.content)

    size_kb = pdf_path.stat().st_size / 1024
    print(f"Downloaded {size_kb:.1f} KB to {pdf_path}")
    return str(pdf_path)


def extract_text_from_pdf(pdf_path: str):
    print(f"Extracting text from {pdf_path}")
    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        raw_text = page.get_text()

        lines = [line.strip() for line in raw_text.splitlines()]
        cleaned = "\n".join(line for line in lines if line)

        if len(cleaned) > 50:
            pages.append(
                {
                    "text": cleaned,
                    "page_number": page_num + 1,
                }
            )

    doc.close()
    print(f"Extracted text from {len(pages)} pages")
    return pages


def chunk_pages(pages: list):
    print("Chunking extracted text")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    all_chunks = []
    chunk_counter = 0

    for page_data in pages:
        raw_chunks = splitter.split_text(page_data["text"])

        for chunk_text in raw_chunks:
            if len(chunk_text.strip()) < 30:
                continue

            all_chunks.append(
                {
                    "chunk_id": f"chunk_{chunk_counter:04d}",
                    "text": chunk_text.strip(),
                    "page_number": page_data["page_number"],
                    "source": "Ebook-Agentic-AI.pdf",
                }
            )
            chunk_counter += 1

    print(f"Created {len(all_chunks)} chunks total")
    return all_chunks


def load_embedding_model():
    print(f"Loading embedding model: {settings.embedding_model}")
    model = SentenceTransformer(settings.embedding_model)
    print("Embedding model loaded")
    return model


def generate_embeddings(chunks: list, model: SentenceTransformer):
    print(f"Generating embeddings for {len(chunks)} chunks")
    texts = [c["text"] for c in chunks]

    batch_size = 32
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        all_embeddings.extend(batch_embeddings.tolist())

        done = min(i + batch_size, len(texts))
        print(f"  Embedded {done}/{len(texts)} chunks")

    print("All embeddings generated")
    return all_embeddings


def store_in_chromadb(
    chunks: list,
    embeddings: list,
    rebuild: bool = False,
):
    print("Connecting to ChromaDB")

    persist_path = Path(settings.chroma_persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_path))

    if rebuild:
        try:
            client.delete_collection(
                settings.chroma_collection_name
            )
            print("Deleted existing collection for rebuild")
        except Exception:
            pass

    try:
        collection = client.get_collection(
            settings.chroma_collection_name
        )
        existing_count = collection.count()

        if existing_count > 0 and not rebuild:
            print(
                f"Collection already has {existing_count} chunks. "
                "Skipping upsert."
            )
            return collection
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 100
    total = len(chunks)

    for i in range(0, total, batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_embeddings = embeddings[i : i + batch_size]

        ids = [c["chunk_id"] for c in batch_chunks]
        docs = [c["text"] for c in batch_chunks]
        metas = [
            {
                "page_number": c["page_number"],
                "source": c["source"],
                "chunk_id": c["chunk_id"],
            }
            for c in batch_chunks
        ]

        collection.upsert(
            ids=ids,
            embeddings=batch_embeddings,
            documents=docs,
            metadatas=metas,
        )

        done = min(i + batch_size, total)
        print(f"  Stored {done}/{total} chunks in ChromaDB")

    final_count = collection.count()
    print(f"ChromaDB ready with {final_count} chunks")
    return collection


def run_ingestion(rebuild: bool = False):
    print("=" * 50)
    print("Starting ingestion pipeline")
    print("=" * 50)

    pdf_path = download_pdf()
    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_pages(pages)
    model = load_embedding_model()
    embeddings = generate_embeddings(chunks, model)
    store_in_chromadb(chunks, embeddings, rebuild=rebuild)

    print("=" * 50)
    print("Ingestion complete!")
    print("=" * 50)


if __name__ == "__main__":
    rebuild_flag = "--rebuild" in sys.argv
    run_ingestion(rebuild=rebuild_flag)
