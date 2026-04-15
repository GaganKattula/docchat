"""
Embedding model factory — returns the appropriate LangChain embedding
model for the chosen LLM provider.

Provider → embedding strategy:
  OpenAI        → text-embedding-3-small (native)
  Google Gemini → text-embedding-004 (native)
  Anthropic     → requires a separate OpenAI or Gemini key for embeddings
                  (Anthropic has no embeddings API)
  Local (Ollama) → Ollama's own /v1/embeddings endpoint (nomic-embed-text)
"""
from __future__ import annotations


def get_embeddings(provider: str, api_key: str | None = None,
                   embed_provider: str | None = None,
                   embed_api_key: str | None = None):
    """
    Return a LangChain embeddings instance.

    Parameters
    ----------
    provider        : chat LLM provider ("OpenAI", "Anthropic", "Google Gemini", "Local (Ollama)")
    api_key         : API key for the chat provider
    embed_provider  : override embedding provider ("OpenAI" or "Google Gemini")
                      only needed when provider is Anthropic or Ollama
    embed_api_key   : API key for the embedding provider (when different from chat key)
    """
    # Use the chat provider's native embeddings when available
    if provider == "OpenAI":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")

    if provider == "Google Gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            google_api_key=api_key,
            model="models/text-embedding-004",
        )

    if provider == "Local (Ollama)":
        # Ollama exposes an OpenAI-compatible /v1/embeddings endpoint.
        # Requires: ollama pull nomic-embed-text
        from langchain_openai import OpenAIEmbeddings
        import streamlit as st
        base_url = st.session_state.get("ollama_base_url", "http://localhost:11434/v1")
        return OpenAIEmbeddings(
            base_url=base_url,
            api_key="ollama",
            model="nomic-embed-text",
        )

    # Anthropic — no native embeddings API. Delegate to a specified provider.
    if provider == "Anthropic":
        key = embed_api_key or api_key
        if embed_provider == "Google Gemini":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(
                google_api_key=key,
                model="models/text-embedding-004",
            )
        # Default: OpenAI
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(api_key=key, model="text-embedding-3-small")

    # Fallback
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")
