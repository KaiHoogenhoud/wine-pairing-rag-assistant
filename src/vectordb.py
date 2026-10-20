"""ChromaDB vector store with sentence-transformers embeddings and Markdown-aware chunking."""

from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from sentence_transformers import SentenceTransformer

import config

CHROMA_BATCH_SIZE = 1000  # Chroma limits the size of a single write


def chunk_markdown(
    text: str,
    title: str = "",
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> List[str]:
    """
    Split a Markdown document into chunks along its section headings.

    Each section is split further only if it is longer than chunk_size.
    Every chunk starts with a "Title > Section" line, so a chunk from the
    "Viticulture" section of the Pinot noir page still says it is about
    Pinot noir, which helps both the embedding and the LLM.

    Args:
        text: Markdown text (without the metadata header)
        title: Document title used in the chunk prefix
        chunk_size: Approximate number of characters per chunk
        chunk_overlap: Characters shared between neighbouring chunks

    Returns:
        List of text chunks
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator="end",  # keep the period with its sentence
    )

    chunks = []
    for section in header_splitter.split_text(text):
        path = [section.metadata[h] for h in ("h1", "h2", "h3") if h in section.metadata]
        if path and path[0] == title:
            path = path[1:]
        prefix = " > ".join([title, *path]) if title else " > ".join(path)
        for piece in size_splitter.split_text(section.page_content):
            chunks.append(f"{prefix}\n{piece}" if prefix else piece)
    return chunks


class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """

    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace model name for embeddings
        """
        self.collection_name = collection_name or config.COLLECTION_NAME
        self.embedding_model_name = embedding_model or config.EMBEDDING_MODEL

        # Initialize ChromaDB client (stored next to the project, not the cwd)
        self.client = chromadb.PersistentClient(
            path=str(config.CHROMA_DIR), settings=Settings(anonymized_telemetry=False)
        )

        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Get or create collection. Cosine distance suits normalized sentence embeddings.
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Wine pairing knowledge base", "hnsw:space": "cosine"},
        )

        print(f"Vector database initialized with collection: {self.collection_name}")

    def count(self) -> int:
        """Number of chunks currently stored."""
        return self.collection.count()

    def reset(self) -> None:
        """Delete and recreate the collection, e.g. after the data changed."""
        metadata = self.collection.metadata
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name, metadata=metadata
        )

    def chunk_text(self, text: str, chunk_size: int = config.CHUNK_SIZE, title: str = "") -> List[str]:
        """
        Split text into chunks for retrieval (see chunk_markdown).

        Args:
            text: Markdown text (without the metadata header)
            chunk_size: Approximate number of characters per chunk
            title: Document title used in the chunk prefix

        Returns:
            List of text chunks
        """
        return chunk_markdown(text, title=title, chunk_size=chunk_size)

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector database.

        Args:
            documents: List of {"content": str, "metadata": dict} documents.
                metadata needs at least "doc_id" and "title".
        """
        print(f"Processing {len(documents)} documents...")
        ids, texts, metadatas = [], [], []
        for doc in documents:
            meta = doc["metadata"]
            for i, chunk in enumerate(self.chunk_text(doc["content"], title=meta["title"])):
                # Stable IDs, so re-ingesting a file overwrites instead of duplicating
                ids.append(f"{meta['doc_id']}::{i}")
                texts.append(chunk)
                metadatas.append({**meta, "chunk": i})

        print(f"Embedding {len(texts)} chunks...")
        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=config.EMBEDDING_BATCH_SIZE,
        )

        for start in range(0, len(ids), CHROMA_BATCH_SIZE):
            end = start + CHROMA_BATCH_SIZE
            self.collection.upsert(
                ids=ids[start:end],
                documents=texts[start:end],
                embeddings=embeddings[start:end].tolist(),
                metadatas=metadatas[start:end],
            )
        print(f"Documents added to vector database ({self.count()} chunks stored)")

    def search(
        self, query: str, n_results: int = 5, where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database.

        Args:
            query: Search query
            n_results: Number of results to return
            where: Optional Chroma metadata filter, e.g. {"category": "recipe"}

        Returns:
            Dictionary containing search results with keys: 'documents', 'metadatas', 'distances', 'ids'
        """
        if self.count() == 0:
            return {"documents": [], "metadatas": [], "distances": [], "ids": []}

        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            where=where,
        )
        # Chroma returns one result list per query; we only send one query
        return {
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0],
            "ids": results["ids"][0],
        }
