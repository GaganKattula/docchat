"""
Multi-provider LLM selector for Streamlit apps.
Supports OpenAI, Anthropic, Google Gemini, and local Ollama models.
"""
import streamlit as st

PROVIDERS = {
    "OpenAI": {
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        "key_label": "OpenAI API Key",
        "key_placeholder": "sk-...",
        "key_help": "Get yours at platform.openai.com/api-keys",
        "requires_key": True,
        "color": "#10A37F",
        "icon": "⬛",
    },
    "Anthropic": {
        "models": ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022", "claude-3-opus-20240229"],
        "key_label": "Anthropic API Key",
        "key_placeholder": "sk-ant-...",
        "key_help": "Get yours at console.anthropic.com/keys",
        "requires_key": True,
        "color": "#D97757",
        "icon": "🔶",
    },
    "Google Gemini": {
        "models": ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
        "key_label": "Gemini API Key",
        "key_placeholder": "AIza...",
        "key_help": "Get yours at aistudio.google.com/apikey",
        "requires_key": True,
        "color": "#4285F4",
        "icon": "💎",
    },
    "Local (Ollama)": {
        "models": ["llama3.2", "llama3.1", "mistral", "phi3", "codellama", "custom..."],
        "key_label": None,
        "key_placeholder": None,
        "key_help": None,
        "requires_key": False,
        "color": "#8B5CF6",
        "icon": "🖥️",
    },
}


def render_llm_selector(default_temp: float = 0.3):
    """
    Renders the provider/model selector in the Streamlit sidebar.
    Returns (llm, api_key_or_none, is_configured: bool).
    Call inside a `with st.sidebar:` block or at top level — it writes to the sidebar.
    """
    st.markdown('<div class="section-label">LLM Provider</div>', unsafe_allow_html=True)

    provider = st.selectbox(
        "provider", list(PROVIDERS.keys()),
        label_visibility="collapsed",
        key="llm_provider",
    )
    cfg = PROVIDERS[provider]

    models = cfg["models"]
    model = st.selectbox("model", models, label_visibility="collapsed", key="llm_model")

    # Custom model name for Ollama
    if model == "custom...":
        model = st.text_input(
            "Custom model name", placeholder="e.g. llama3.2:8b",
            label_visibility="visible", key="llm_custom_model"
        )

    # API key input
    api_key = None
    if cfg["requires_key"]:
        st.markdown(f'<div class="section-label">{cfg["key_label"]}</div>', unsafe_allow_html=True)
        api_key = st.text_input(
            "api_key", label_visibility="collapsed",
            type="password",
            placeholder=cfg["key_placeholder"],
            help=cfg["key_help"],
            key="llm_api_key",
        )
    else:
        # Ollama: base URL
        st.markdown('<div class="section-label">Ollama base URL</div>', unsafe_allow_html=True)
        base_url = st.text_input(
            "ollama_url", label_visibility="collapsed",
            value="http://localhost:11434/v1",
            key="ollama_base_url",
        )
        st.caption("Make sure Ollama is running locally (`ollama serve`).")

    # Ready state
    is_configured = bool(
        (cfg["requires_key"] and api_key) or
        (not cfg["requires_key"] and model)
    )

    # Provider badge
    color = cfg["color"]
    icon = cfg["icon"]
    status_color = "#10B981" if is_configured else "#374151"
    status_text = f"{icon} {provider} · {model}" if is_configured else "Not configured"
    st.markdown(
        f'<div style="margin-top:8px;display:flex;align-items:center;gap:6px;'
        f'background:#0A0F1E;border:1px solid {color}22;border-radius:6px;'
        f'padding:6px 10px;">'
        f'<div style="width:6px;height:6px;border-radius:50%;background:{status_color};flex-shrink:0;"></div>'
        f'<span style="font-size:0.72rem;color:{color if is_configured else "#374151"};font-weight:500;">'
        f'{status_text}</span></div>',
        unsafe_allow_html=True,
    )

    return provider, model, api_key, is_configured


def build_llm(provider: str, model: str, api_key: str | None,
              temperature: float = 0.3, streaming: bool = False):
    """Instantiate the correct LangChain chat model for the chosen provider."""
    if provider == "OpenAI":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(api_key=api_key, model=model,
                          temperature=temperature, streaming=streaming)

    elif provider == "Anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(api_key=api_key, model=model,  # type: ignore
                             temperature=temperature, streaming=streaming)

    elif provider == "Google Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(google_api_key=api_key, model=model,
                                      temperature=temperature)

    elif provider == "Local (Ollama)":
        from langchain_openai import ChatOpenAI
        base_url = st.session_state.get("ollama_base_url", "http://localhost:11434/v1")
        return ChatOpenAI(base_url=base_url, api_key="ollama",
                          model=model, temperature=temperature, streaming=streaming)

    raise ValueError(f"Unknown provider: {provider}")
