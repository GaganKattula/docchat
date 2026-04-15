import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from pypdf import PdfReader
from docx import Document as DocxDocument
import tempfile
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocChat — Talk to Your Documents",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Base ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* ── Layout ── */
  .block-container {
    max-width: 860px;
    padding: 2rem 2rem 6rem;
  }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: #0A0F1E;
    border-right: 1px solid #1E2940;
  }

  div[data-testid="stSidebarContent"] {
    padding: 1.5rem 1rem;
  }

  .sidebar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 4px;
  }

  .sidebar-logo-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #2563EB, #7C3AED);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; line-height: 1;
  }

  .sidebar-logo-text {
    font-size: 1.25rem;
    font-weight: 700;
    color: #F1F5F9;
    letter-spacing: -0.3px;
  }

  .sidebar-tagline {
    font-size: 0.78rem;
    color: #64748B;
    margin-bottom: 1.2rem;
  }

  .sidebar-section-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #475569;
    margin: 1rem 0 0.4rem;
  }

  /* ── File pills ── */
  .file-pill {
    display: flex;
    align-items: center;
    gap: 8px;
    background: #111827;
    border: 1px solid #1E2940;
    border-radius: 6px;
    padding: 6px 10px;
    margin-bottom: 6px;
    font-size: 0.8rem;
    color: #CBD5E1;
  }

  .file-pill-icon { color: #3B82F6; font-size: 0.9rem; }

  /* ── Stats ── */
  .stats-row {
    display: flex;
    gap: 8px;
    margin-top: 0.8rem;
  }

  .stat-card {
    flex: 1;
    background: #111827;
    border: 1px solid #1E2940;
    border-radius: 8px;
    padding: 10px 12px;
    text-align: center;
  }

  .stat-card-value {
    font-size: 1.3rem;
    font-weight: 700;
    color: #60A5FA;
    line-height: 1.2;
  }

  .stat-card-label {
    font-size: 0.65rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 2px;
  }

  /* ── Status badge ── */
  .status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #34D399;
    border-radius: 20px;
    padding: 4px 10px;
    font-size: 0.72rem;
    font-weight: 500;
    margin-top: 0.8rem;
  }

  .status-dot {
    width: 6px; height: 6px;
    background: #10B981;
    border-radius: 50%;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }

  /* ── Hero (empty state) ── */
  .hero {
    text-align: center;
    padding: 4rem 2rem 2rem;
  }

  .hero-badge {
    display: inline-block;
    background: rgba(37, 99, 235, 0.12);
    border: 1px solid rgba(37, 99, 235, 0.3);
    color: #60A5FA;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 500;
    margin-bottom: 1.2rem;
    letter-spacing: 0.02em;
  }

  .hero h1 {
    font-size: 2.4rem;
    font-weight: 700;
    color: #F1F5F9;
    letter-spacing: -0.5px;
    line-height: 1.2;
    margin-bottom: 0.8rem;
  }

  .hero-sub {
    font-size: 1rem;
    color: #64748B;
    max-width: 480px;
    margin: 0 auto 2.5rem;
    line-height: 1.6;
  }

  /* ── Feature cards ── */
  .features {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    max-width: 680px;
    margin: 0 auto;
  }

  .feature-card {
    background: #0F172A;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 18px 16px;
    text-align: left;
  }

  .feature-icon {
    font-size: 1.4rem;
    margin-bottom: 8px;
  }

  .feature-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: #E2E8F0;
    margin-bottom: 4px;
  }

  .feature-desc {
    font-size: 0.74rem;
    color: #475569;
    line-height: 1.5;
  }

  /* ── Step indicator (upload state) ── */
  .step-row {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 1rem 1.2rem;
    background: #0F172A;
    border: 1px solid #1E293B;
    border-radius: 10px;
    margin-bottom: 10px;
  }

  .step-num {
    width: 26px; height: 26px;
    min-width: 26px;
    border-radius: 50%;
    background: rgba(37, 99, 235, 0.15);
    border: 1px solid rgba(37, 99, 235, 0.4);
    color: #60A5FA;
    font-size: 0.75rem;
    font-weight: 600;
    display: flex; align-items: center; justify-content: center;
  }

  .step-done .step-num {
    background: rgba(16, 185, 129, 0.15);
    border-color: rgba(16, 185, 129, 0.4);
    color: #34D399;
  }

  .step-text { font-size: 0.85rem; color: #94A3B8; padding-top: 3px; }
  .step-text strong { color: #E2E8F0; }

  /* ── Chat ── */
  .stChatMessage {
    padding: 0.75rem 0 !important;
    border-radius: 0 !important;
    border-bottom: 1px solid #1E293B;
  }

  .stChatMessage:last-child { border-bottom: none; }

  /* ── Source expander ── */
  .source-header {
    font-size: 0.72rem;
    font-weight: 600;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 8px;
  }

  .source-chunk {
    background: #0F172A;
    border: 1px solid #1E293B;
    border-left: 3px solid #2563EB;
    border-radius: 6px;
    padding: 10px 12px;
    font-size: 0.8rem;
    color: #94A3B8;
    line-height: 1.6;
    margin-bottom: 8px;
    font-family: monospace;
  }

  /* ── Input override ── */
  .stTextInput input {
    background: #111827 !important;
    border: 1px solid #1E2940 !important;
    border-radius: 8px !important;
    color: #F1F5F9 !important;
    font-size: 0.85rem !important;
  }

  .stTextInput input:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 2px rgba(37,99,235,0.2) !important;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: #111827 !important;
    border: 1px solid #1E2940 !important;
    border-radius: 8px !important;
    color: #94A3B8 !important;
    font-size: 0.82rem !important;
    transition: all 0.15s ease;
  }

  .stButton > button:hover {
    border-color: #3B82F6 !important;
    color: #60A5FA !important;
    background: rgba(37,99,235,0.08) !important;
  }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_text_from_pdf(file) -> str:
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_text_from_docx(file) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file.getvalue())
        tmp_path = tmp.name
    doc = DocxDocument(tmp_path)
    os.unlink(tmp_path)
    return "\n".join(p.text for p in doc.paragraphs)


def extract_text_from_txt(file) -> str:
    return file.getvalue().decode("utf-8")


EXTRACTORS = {
    "application/pdf": extract_text_from_pdf,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": extract_text_from_docx,
    "text/plain": extract_text_from_txt,
}


def process_documents(files, api_key: str):
    all_text, file_names = [], []
    for f in files:
        extractor = EXTRACTORS.get(f.type)
        if extractor:
            text = extractor(f)
            if text.strip():
                all_text.append(text)
                file_names.append(f.name)

    if not all_text:
        return None, [], 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text("\n\n".join(all_text))
    embeddings = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore, file_names, len(chunks)


def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vectorstore, api_key: str):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0.3, streaming=True)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a precise document assistant. Answer only from the provided context. "
         "If the answer isn't in the context, say so. Be concise and clear.\n\nContext:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])
    chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["question"])),
            "chat_history": lambda x: x["chat_history"],
            "question": lambda x: x["question"],
        }
        | prompt | llm | StrOutputParser()
    )
    return chain, retriever


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

    st.markdown('<div class="sidebar-section-label">API Key</div>', unsafe_allow_html=True)
    api_key = st.text_input(
        "key", label_visibility="collapsed",
        type="password", placeholder="sk-...",
        help="Never stored. Lives only in your browser session.",
    )

    st.markdown('<div class="sidebar-section-label">Documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "files", label_visibility="collapsed",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files and api_key:
        file_set = frozenset((f.name, f.size) for f in uploaded_files)
        if file_set != st.session_state.get("_last_file_set"):
            with st.spinner("Embedding documents..."):
                vs, names, chunks = process_documents(uploaded_files, api_key)
                if vs:
                    st.session_state.vectorstore = vs
                    st.session_state.chain, st.session_state.retriever = build_rag_chain(vs, api_key)
                    st.session_state.file_names = names
                    st.session_state.chunk_count = chunks
                    st.session_state.messages = []
                    st.session_state._last_file_set = file_set
                else:
                    st.error("Could not extract text from files.")

    if st.session_state.file_names:
        for name in st.session_state.file_names:
            ext = name.rsplit(".", 1)[-1].upper()
            icon = {"PDF": "📕", "DOCX": "📘", "TXT": "📄"}.get(ext, "📄")
            st.markdown(
                f'<div class="file-pill"><span class="file-pill-icon">{icon}</span>{name}</div>',
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
            if st.session_state.vectorstore and api_key:
                st.session_state.chain, st.session_state.retriever = build_rag_chain(
                    st.session_state.vectorstore, api_key)
            st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────

# ── State: no API key ──
if not api_key:
    st.markdown("""
    <div class="hero">
      <div class="hero-badge">Powered by RAG + GPT-4o mini</div>
      <h1>Talk to your documents</h1>
      <div class="hero-sub">
        Upload any PDF, Word doc, or text file and ask questions in plain English.
        Get precise answers grounded in your content.
      </div>
      <div class="features">
        <div class="feature-card">
          <div class="feature-icon">🔍</div>
          <div class="feature-title">Semantic Search</div>
          <div class="feature-desc">Finds relevant content even when phrasing differs</div>
        </div>
        <div class="feature-card">
          <div class="feature-icon">🧠</div>
          <div class="feature-title">Context-Aware</div>
          <div class="feature-desc">Remembers earlier turns in the conversation</div>
        </div>
        <div class="feature-card">
          <div class="feature-icon">📎</div>
          <div class="feature-title">Source Traced</div>
          <div class="feature-desc">Every answer links back to the exact passage</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="max-width:480px;margin:2rem auto 0;padding:1.2rem 1.4rem;
    background:#0F172A;border:1px solid #1E293B;border-radius:12px;">
      <div style="font-size:0.72rem;color:#475569;text-transform:uppercase;
      letter-spacing:0.06em;margin-bottom:0.8rem;font-weight:600;">Get started</div>
      <div class="step-row step-done">
        <div class="step-num">1</div>
        <div class="step-text"><strong>Paste your OpenAI API key</strong> in the sidebar</div>
      </div>
      <div class="step-row">
        <div class="step-num">2</div>
        <div class="step-text"><strong>Upload your documents</strong> — PDF, DOCX, or TXT</div>
      </div>
      <div class="step-row">
        <div class="step-num">3</div>
        <div class="step-text"><strong>Start asking questions</strong> in the chat below</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── State: API key set, no docs ──
if not st.session_state.vectorstore:
    st.markdown("""
    <div class="hero" style="padding-top:3rem;">
      <div class="hero-badge">API key detected</div>
      <h1 style="font-size:1.9rem;">Upload your documents</h1>
      <div class="hero-sub">
        Drag and drop PDFs, Word docs, or text files into the sidebar to get started.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="max-width:480px;margin:0 auto;padding:1.2rem 1.4rem;
    background:#0F172A;border:1px solid #1E293B;border-radius:12px;">
      <div class="step-row step-done">
        <div class="step-num">✓</div>
        <div class="step-text"><strong>API key set</strong></div>
      </div>
      <div class="step-row step-done">
        <div class="step-num">2</div>
        <div class="step-text"><strong>Upload documents</strong> in the sidebar ←</div>
      </div>
      <div class="step-row">
        <div class="step-num">3</div>
        <div class="step-text"><strong>Ask questions</strong> — coming right up</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── State: ready to chat ──
# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if prompt := st.chat_input("Ask anything about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        chat_history = []
        for msg in st.session_state.messages[:-1]:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            else:
                chat_history.append(AIMessage(content=msg["content"]))

        # Stream response token by token
        answer = st.write_stream(
            st.session_state.chain.stream({
                "question": prompt,
                "chat_history": chat_history,
            })
        )

        # Source passages
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
