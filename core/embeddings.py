"""
Embedding model factory — returns the appropriate LangChain embedding
model for the chosen LLM provider.
"""
from __future__ import annotations

_HUGGINGFACE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings(provider: str, api_key: str | None = None):
    """
    Return a LangChain embeddings instance for *provider*.

    Anthropic and Local (Ollama) fall back to a free HuggingFace
    sentence-transformer because they have no native embeddings API.

    Parameters
    ----------
    provider : one of "OpenAI", "Anthropic", "Google Gemini", "Local (Ollama)"
    api_key  : API key for cloud providers; ignored for local/HuggingFace

    Returns
    -------
    LangChain Embeddings instance
    """
    if provider == "OpenAI":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")

    if provider == "Google Gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            google_api_key=api_key,
            model="models/text-embedding-004",
        )

    # Anthropic has no embeddings API; Ollama uses local compute.
    # Both fall back to a lightweight, free HuggingFace model.
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=_HUGGINGFACE_MODEL)
