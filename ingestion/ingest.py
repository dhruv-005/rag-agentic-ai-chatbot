import os
import sys
import requests
import fitz
import chromadb
import numpy as np
from pathlib import Path
from chromadb.utils import embedding_functions

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings


def get_embedding_function():
    """
    use chromadb built in embedding function
    this uses onnxruntime under the hood
    no torch no torchvision needed at all
    model is all-MiniLM-L6-v2 which is small and fast
    """
    return embedding_functions.DefaultEmbeddingFunction()


def download_pdf():
    pdf_path = Path(settings.pdf_local_path)

    if pdf_path.exists():
        print("PDF already exists — skipping download")
        return str(pdf_path)

    print(f"Downloading PDF from {settings.pdf_url}")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(settings.pdf_url, timeout=60)
    response.raise_for_status()

    with open(pdf_path, "wb") as f:
        f.write(response.content)

    size_kb = pdf_path.stat().st_size / 1024
    print(f"Downloaded {size_kb:.1f} KB")
    return str(pdf_path)


def extract_text_from_pdf(pdf_path: str):
    print("Extracting text from PDF")
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
    print(f"Extracted {len(pages)} pages")
    return pages


def chunk_pages(pages: list):
    print("Chunking text")

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

    print(f"Created {len(all_chunks)} chunks")
    return all_chunks


def store_in_chromadb(
    chunks: list,
    rebuild: bool = False,
):
    print("Storing in ChromaDB")

    persist_path = Path(settings.chroma_persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_path))

    # use chromadb default embedding function
    # no torch needed uses onnxruntime instead
    ef = get_embedding_function()

    if rebuild:
        try:
            client.delete_collection(
                settings.chroma_collection_name
            )
            print("Deleted old collection")
        except Exception:
            pass

    try:
        collection = client.get_collection(
            name=settings.chroma_collection_name,
            embedding_function=ef,
        )
        if collection.count() > 0 and not rebuild:
            print(f"Already has {collection.count()} chunks")
            return collection
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=settings.chroma_collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    # upsert in small batches
    batch_size = 50
    total = len(chunks)

    for i in range(0, total, batch_size):
        bc = chunks[i : i + batch_size]

        collection.upsert(
            ids=[c["chunk_id"] for c in bc],
            documents=[c["text"] for c in bc],
            metadatas=[
                {
                    "page_number": c["page_number"],
                    "source": c["source"],
                    "chunk_id": c["chunk_id"],
                }
                for c in bc
            ],
        )
        done = min(i + batch_size, total)
        print(f"  Stored {done}/{total}")

    print(f"ChromaDB ready — {collection.count()} chunks")
    return collection


def run_ingestion(rebuild: bool = False):
    print("Starting ingestion")

    pdf_path = download_pdf()
    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_pages(pages)

    # no separate embedding step needed
    # chromadb handles embeddings internally
    store_in_chromadb(chunks, rebuild=rebuild)

    print("Ingestion complete")


if __name__ == "__main__":
    rebuild_flag = "--rebuild" in sys.argv
    run_ingestion(rebuild=rebuild_flag)
