"""Shared fixtures for DocChat tests."""
import io
import pytest


@pytest.fixture
def sample_txt_bytes() -> bytes:
    return b"This is a test document.\nIt has multiple lines.\nUsed for unit testing."


@pytest.fixture
def sample_txt_file(sample_txt_bytes):
    """Mimics a Streamlit UploadedFile for plain text."""
    f = io.BytesIO(sample_txt_bytes)
    f.name = "test.txt"
    f.type = "text/plain"
    return f


@pytest.fixture
def long_text() -> str:
    return ("The quick brown fox jumps over the lazy dog. " * 60).strip()
