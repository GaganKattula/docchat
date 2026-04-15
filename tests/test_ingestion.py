"""Unit tests for core/ingestion.py — no API calls required."""
import io
import pytest

from core.ingestion import (
    extract_text_from_txt,
    extract_text,
    chunk_text,
    load_and_chunk,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


class TestExtractTextFromTxt:
    def test_bytes_input(self):
        result = extract_text_from_txt(b"hello world")
        assert result == "hello world"

    def test_bytesio_input(self):
        result = extract_text_from_txt(io.BytesIO(b"hello bytesio"))
        assert result == "hello bytesio"

    def test_multiline(self):
        text = b"line one\nline two\nline three"
        result = extract_text_from_txt(text)
        assert "line one" in result
        assert "line three" in result

    def test_utf8_characters(self):
        text = "héllo wörld".encode("utf-8")
        result = extract_text_from_txt(text)
        assert "héllo" in result


class TestExtractTextDispatch:
    def test_dispatch_by_mime_type(self, sample_txt_bytes):
        result = extract_text(io.BytesIO(sample_txt_bytes), mime_type="text/plain")
        assert "test document" in result

    def test_dispatch_by_filename(self, sample_txt_bytes):
        result = extract_text(io.BytesIO(sample_txt_bytes), filename="notes.txt")
        assert "test document" in result

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            extract_text(io.BytesIO(b"data"), mime_type="image/png")

    def test_no_type_info_raises(self):
        with pytest.raises(ValueError):
            extract_text(io.BytesIO(b"data"))


class TestChunkText:
    def test_short_text_single_chunk(self):
        chunks = chunk_text("short text")
        assert len(chunks) == 1
        assert chunks[0] == "short text"

    def test_long_text_multiple_chunks(self, long_text):
        chunks = chunk_text(long_text, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1

    def test_chunk_size_respected(self, long_text):
        chunk_size = 200
        chunks = chunk_text(long_text, chunk_size=chunk_size, chunk_overlap=20)
        for chunk in chunks:
            # LangChain may slightly exceed chunk_size at word boundaries
            assert len(chunk) <= chunk_size * 1.5

    def test_empty_text_returns_empty(self):
        chunks = chunk_text("")
        assert chunks == []


class TestLoadAndChunk:
    def test_single_txt_file(self, sample_txt_file):
        chunks, names = load_and_chunk([sample_txt_file])
        assert len(chunks) >= 1
        assert names == ["test.txt"]

    def test_empty_file_skipped(self):
        empty = io.BytesIO(b"   ")
        empty.name = "empty.txt"
        empty.type = "text/plain"
        chunks, names = load_and_chunk([empty])
        assert chunks == []
        assert names == []

    def test_unsupported_file_skipped(self):
        bad = io.BytesIO(b"data")
        bad.name = "image.png"
        bad.type = "image/png"
        chunks, names = load_and_chunk([bad])
        assert chunks == []
        assert names == []

    def test_multiple_files_combined(self, sample_txt_file):
        f2 = io.BytesIO(b"Second document content here.")
        f2.name = "doc2.txt"
        f2.type = "text/plain"
        chunks, names = load_and_chunk([sample_txt_file, f2])
        assert len(names) == 2
        assert len(chunks) >= 1
