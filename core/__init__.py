"""DocChat core — ingestion, embeddings, and RAG chain."""
from .ingestion import load_and_chunk
from .embeddings import get_embeddings
from .chain import build_vectorstore, build_rag_chain

__all__ = ["load_and_chunk", "get_embeddings", "build_vectorstore", "build_rag_chain"]
