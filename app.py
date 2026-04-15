"""
DocChat — RAG-powered document chatbot.
Supports OpenAI, Anthropic, Google Gemini, and local Ollama.

Entry point: `streamlit run app.py`
"""
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from core import load_and_chunk, get_embeddings, build_vectorstore, build_rag_chain
from llm_config import render_llm_selector, build_llm

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocChat — Talk to Your Documents",
    page_icon="📄",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { max-width: 860px; padding: 2rem 2rem 6rem; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

section[data-testid="stSidebar"] { background: #0A0F1E; border-right: 1px solid #1E2940; }
div[data-testid="stSidebarContent"] { padding: 1.5rem 1rem; }

.sidebar-logo { display: flex; align-items: center; gap: 10px; margin-bottom: 4px; }
.sidebar-logo-icon {
  width: 36px; height: 36px;
  background: linear-gradient(135deg, #2563EB, #7C3AED);
  border-radius: 8px; display: flex; align-items: center; justify-content: center;
  font-size: 18px; line-height: 1;
}
.sidebar-logo-text { font-size: 1.25rem; font-weight: 700; color: #F1F5F9; letter-spacing: -0.3px; }
.sidebar-tagline { font-size: 0.78rem; color: #64748B; margin-bottom: 1.2rem; }
.section-label {
  font-size: 0.68rem; font-weight: 600; letter-spacing: 0.08em;
  text-transform: uppercase; color: #475569; margin: 1rem 0 0.4rem;
}

.file-pill {
  display: flex; align-items: center; gap: 8px;
  background: #111827; border: 1px solid #1E2940;
  border-radius: 6px; padding: 6px 10px; margin-bottom: 6px;
  font-size: 0.8rem; color: #CBD5E1;
}

.stats-row { display: flex; gap: 8px; margin-top: 0.8rem; }
.stat-card {
  flex: 1; background: #111827; border: 1px solid #1E2940;
  border-radius: 8px; padding: 10px 12px; text-align: center;
}
.stat-card-value { font-size: 1.3rem; font-weight: 700; color: #60A5FA; line-height: 1.2; }
.stat-card-label { font-size: 0.65rem; color: #475569; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 2px; }

.status-badge {
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.3);
  color: #34D399; border-radius: 20px; padding: 4px 10px;
  font-size: 0.72rem; font-weight: 500; margin-top: 0.8rem;
}
.status-dot {
  width: 6px; height: 6px; background: #10B981; border-radius: 50%;
  animation: pulse 2s infinite;
}
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

.hero { text-align: center; padding: 4rem 2rem 2rem; }
.hero-badge {
  display: inline-block; background: rgba(37,99,235,0.12);
  border: 1px solid rgba(37,99,235,0.3); color: #60A5FA;
  border-radius: 20px; padding: 4px 14px; font-size: 0.78rem;
  font-weight: 500; margin-bottom: 1.2rem; letter-spacing: 0.02em;
}
.hero h1 { font-size: 2.4rem; font-weight: 700; color: #F1F5F9; letter-spacing: -0.5px; line-height: 1.2; margin-bottom: 0.8rem; }
.hero-sub { font-size: 1rem; color: #64748B; max-width: 480px; margin: 0 auto 2.5rem; line-height: 1.6; }

.features { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; max-width: 680px; margin: 0 auto; }
.feature-card { background: #0F172A; border: 1px solid #1E293B; border-radius: 12px; padding: 18px 16px; text-align: left; }
.feature-icon { font-size: 1.4rem; margin-bottom: 8px; }
.feature-title { font-size: 0.82rem; font-weight: 600; color: #E2E8F0; margin-bottom: 4px; }
.feature-desc { font-size: 0.74rem; color: #475569; line-height: 1.5; }

.step-row { display: flex; align-items: flex-start; gap: 14px; padding: 1rem 1.2rem; background: #0F172A; border: 1px solid #1E293B; border-radius: 10px; margin-bottom: 10px; }
.step-num { width: 26px; height: 26px; min-width: 26px; border-radius: 50%; background: rgba(37,99,235,0.15); border: 1px solid rgba(37,99,235,0.4); color: #60A5FA; font-size: 0.75rem; font-weight: 600; display: flex; align-items: center; justify-content: center; }
.step-done .step-num { background: rgba(16,185,129,0.15); border-color: rgba(16,185,129,0.4); color: #34D399; }
.step-text { font-size: 0.85rem; color: #94A3B8; padding-top: 3px; }
.step-text strong { color: #E2E8F0; }

.stChatMessage { padding: 0.75rem 0 !important; border-radius: 0 !important; border-bottom: 1px solid #1E293B; }
.stChatMessage:last-child { border-bottom: none; }

.source-header { font-size: 0.72rem; font-weight: 600; color: #475569; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px; }
.source-chunk { background: #0F172A; border: 1px solid #1E293B; border-left: 3px solid #2563EB; border-radius: 6px; padding: 10px 12px; font-size: 0.8rem; color: #94A3B8; line-height: 1.6; margin-bottom: 8px; font-family: monospace; }

textarea { background: #111827 !important; border: 1px solid #1E2940 !important; border-radius: 8px !important; color: #F1F5F9 !important; font-size: 0.85rem !important; }
textarea:focus { border-color: #2563EB !important; box-shadow: 0 0 0 2px rgba(37,99,235,0.2) !important; }
.stButton > button { background: #111827 !important; border: 1px solid #1E2940 !important; border-radius: 8px !important; color: #94A3B8 !important; font-size: 0.82rem !important; transition: all 0.15s ease; }
.stButton > button:hover { border-color: #3B82F6 !important; color: #60A5FA !important; background: rgba(37,99,235,0.08) !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("messages", []), ("vectorstore", None), ("chain", None),
    ("retriever", None), ("file_names", []), ("chunk_count", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
      <div class="sidebar-logo-icon">📄</div>
      <span class="sidebar-logo-text">DocChat</span>
    </div>
    <div class="sidebar-tagline">Instant answers from your documents</div>
    """, unsafe_allow_html=True)

    provider, model, api_key, is_configured = render_llm_selector()

    # Anthropic has no embeddings API — ask for an OpenAI or Gemini key
    embed_provider = None
    embed_api_key = None
    if provider == "Anthropic" and is_configured:
        st.markdown(
            '<div style="background:#1a1208;border:1px solid #3d2e00;border-left:3px solid #F59E0B;'
            'border-radius:6px;padding:8px 10px;margin:8px 0;">'
            '<div style="font-size:0.72rem;color:#F59E0B;font-weight:600;margin-bottom:3px;">Embeddings</div>'
            '<div style="font-size:0.72rem;color:#78716c;line-height:1.5;">'
            "Anthropic has no embeddings API. Provide an OpenAI or Gemini key for document indexing."
            "</div></div>",
            unsafe_allow_html=True,
        )
        embed_provider = st.selectbox("Embed with", ["OpenAI", "Google Gemini"],
                                      key="embed_provider", label_visibility="collapsed")
        embed_api_key = st.text_input(
            f"{embed_provider} key for embeddings", label_visibility="collapsed",
            type="password",
            placeholder="sk-..." if embed_provider == "OpenAI" else "AIza...",
            key="embed_api_key",
        )

    st.markdown('<div class="section-label">Documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "files", label_visibility="collapsed",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files and is_configured:
        file_set = frozenset((f.name, f.size) for f in uploaded_files)
        if file_set != st.session_state.get("_last_file_set"):
            with st.spinner("Embedding documents..."):
                chunks, file_names = load_and_chunk(uploaded_files)
                if chunks:
                    embeddings = get_embeddings(provider, api_key, embed_provider, embed_api_key)
                    vs = build_vectorstore(chunks, embeddings)
                    llm = build_llm(provider, model, api_key, streaming=True)
                    chain, retriever = build_rag_chain(vs, llm)
                    st.session_state.update({
                        "vectorstore": vs, "chain": chain, "retriever": retriever,
                        "file_names": file_names, "chunk_count": len(chunks),
                        "messages": [], "_last_file_set": file_set,
                    })
                else:
                    st.error("Could not extract text from the uploaded files.")

    if st.session_state.file_names:
        for name in st.session_state.file_names:
            ext = name.rsplit(".", 1)[-1].upper()
            icon = {"PDF": "📕", "DOCX": "📘", "TXT": "📄"}.get(ext, "📄")
            st.markdown(
                f'<div class="file-pill"><span>{icon}</span>{name}</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<div class="stats-row">'
            f'<div class="stat-card"><div class="stat-card-value">{len(st.session_state.file_names)}</div>'
            f'<div class="stat-card-label">Files</div></div>'
            f'<div class="stat-card"><div class="stat-card-value">{st.session_state.chunk_count}</div>'
            f'<div class="stat-card-label">Chunks</div></div>'
            f'<div class="stat-card"><div class="stat-card-value">{len(st.session_state.messages) // 2}</div>'
            f'<div class="stat-card-label">Turns</div></div>'
            f'</div>'
            f'<div class="status-badge"><div class="status-dot"></div>Ready to chat</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.messages:
        st.write("")
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.vectorstore and is_configured:
                llm = build_llm(provider, model, api_key, streaming=True)
                st.session_state.chain, st.session_state.retriever = build_rag_chain(
                    st.session_state.vectorstore, llm)
            st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────
if not is_configured:
    st.markdown("""
    <div class="hero">
      <div class="hero-badge">OpenAI · Anthropic · Gemini · Local Ollama</div>
      <h1>Talk to your documents</h1>
      <div class="hero-sub">Upload any PDF, Word doc, or text file and get precise answers grounded in your content.</div>
      <div class="features">
        <div class="feature-card"><div class="feature-icon">🔍</div><div class="feature-title">Semantic Search</div><div class="feature-desc">Finds relevant content even when phrasing differs</div></div>
        <div class="feature-card"><div class="feature-icon">🧠</div><div class="feature-title">Context-Aware</div><div class="feature-desc">Remembers earlier turns in the conversation</div></div>
        <div class="feature-card"><div class="feature-icon">📎</div><div class="feature-title">Source Traced</div><div class="feature-desc">Every answer links back to the exact passage</div></div>
      </div>
    </div>
    <div style="max-width:480px;margin:2rem auto 0;padding:1.2rem 1.4rem;background:#0F172A;border:1px solid #1E293B;border-radius:12px;">
      <div style="font-size:0.72rem;color:#475569;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.8rem;font-weight:600;">Get started</div>
      <div class="step-row"><div class="step-num">1</div><div class="step-text"><strong>Choose a provider</strong> and configure it in the sidebar</div></div>
      <div class="step-row"><div class="step-num">2</div><div class="step-text"><strong>Upload your documents</strong> — PDF, DOCX, or TXT</div></div>
      <div class="step-row"><div class="step-num">3</div><div class="step-text"><strong>Start asking questions</strong> in the chat below</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if not st.session_state.vectorstore and is_configured:
    st.markdown("""
    <div class="hero" style="padding-top:3rem;">
      <div class="hero-badge">Provider configured</div>
      <h1 style="font-size:1.9rem;">Upload your documents</h1>
      <div class="hero-sub">Drag and drop PDFs, Word docs, or text files into the sidebar.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask anything about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        chat_history = [
            HumanMessage(content=m["content"]) if m["role"] == "user"
            else AIMessage(content=m["content"])
            for m in st.session_state.messages[:-1]
        ]
        answer = st.write_stream(
            st.session_state.chain.stream({"question": prompt, "chat_history": chat_history})
        )
        source_docs = st.session_state.retriever.invoke(prompt)
        if source_docs:
            with st.expander("Sources", expanded=False):
                st.markdown('<div class="source-header">Retrieved passages</div>', unsafe_allow_html=True)
                for i, doc in enumerate(source_docs, 1):
                    snippet = doc.page_content[:280].strip()
                    st.markdown(
                        f'<div class="source-chunk"><strong style="color:#60A5FA">#{i}</strong>  {snippet}…</div>',
                        unsafe_allow_html=True,
                    )

    st.session_state.messages.append({"role": "assistant", "content": answer})
