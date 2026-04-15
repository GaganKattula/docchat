"""Unit tests for core/chain.py — LLM and embeddings are mocked."""
from unittest.mock import MagicMock, patch
import pytest

from core.chain import build_vectorstore, build_rag_chain, _format_docs, DEFAULT_TOP_K


class TestFormatDocs:
    def test_single_doc(self):
        doc = MagicMock()
        doc.page_content = "hello world"
        assert _format_docs([doc]) == "hello world"

    def test_multiple_docs_separated(self):
        docs = [MagicMock(page_content=f"doc{i}") for i in range(3)]
        result = _format_docs(docs)
        assert "doc0" in result
        assert "doc1" in result
        assert "---" in result


class TestBuildVectorstore:
    def test_empty_chunks_raises(self):
        with pytest.raises(ValueError, match="empty"):
            build_vectorstore([], embeddings=MagicMock())

    def test_returns_faiss_instance(self):
        mock_embeddings = MagicMock()
        mock_vs = MagicMock()
        with patch("core.chain.FAISS.from_texts", return_value=mock_vs) as mock_ft:
            result = build_vectorstore(["chunk1", "chunk2"], mock_embeddings)
            mock_ft.assert_called_once_with(["chunk1", "chunk2"], mock_embeddings)
            assert result is mock_vs


class TestBuildRagChain:
    def test_returns_chain_and_retriever(self):
        mock_vs = MagicMock()
        mock_retriever = MagicMock()
        mock_vs.as_retriever.return_value = mock_retriever
        mock_llm = MagicMock()

        chain, retriever = build_rag_chain(mock_vs, mock_llm)

        mock_vs.as_retriever.assert_called_once_with(
            search_kwargs={"k": DEFAULT_TOP_K}
        )
        assert retriever is mock_retriever
        assert chain is not None

    def test_custom_top_k(self):
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = MagicMock()
        build_rag_chain(mock_vs, MagicMock(), top_k=8)
        mock_vs.as_retriever.assert_called_once_with(search_kwargs={"k": 8})
