# DocChat — RAG-Powered Document Chatbot

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![CI](https://github.com/YOUR_USERNAME/docchat/actions/workflows/ci.yml/badge.svg)

Upload any PDF, Word document, or plain text file and ask questions in plain English. DocChat uses **Retrieval-Augmented Generation (RAG)** to find answers directly from your content — no hallucination, every answer is grounded in the source.

**Works with any LLM provider — switch between OpenAI, Anthropic Claude, Google Gemini, or a fully local Ollama model without changing a single line of code.**

---

## Features

- **Multi-provider LLM support** — OpenAI, Anthropic, Google Gemini, or local Ollama (no API key needed)
- **Multi-format ingestion** — PDF, DOCX, and TXT
- **Semantic retrieval** — FAISS vector search with configurable top-k
- **Streaming responses** — token-by-token output for a natural chat experience
- **Source transparency** — every answer shows the exact document passages retrieved
- **Conversational memory** — follows up on previous questions within a session
- **Clean dark UI** — polished Streamlit interface with custom theming

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                        DocChat                           │
│                                                          │
│  ┌─────────────┐    ┌────────────────────────────────┐   │
│  │  core/      │    │  app.py (Streamlit UI)          │   │
│  │             │    │                                 │   │
│  │ ingestion   │───▶│  Sidebar: provider selector     │   │
│  │    ├─ PDF   │    │          file uploader          │   │
│  │    ├─ DOCX  │    │                                 │   │
│  │    └─ TXT   │    │  Main:   chat history           │   │
│  │             │    │          streaming response     │   │
│  │ embeddings  │    │          source expander        │   │
│  │    ├─ OAI   │    └────────────────────────────────┘   │
│  │    ├─ HF    │                                          │
│  │    └─ Gemini│    ┌────────────────────────────────┐   │
│  │             │    │  llm_config.py                  │   │
│  │ chain       │    │  ├─ render_llm_selector()       │   │
│  │    ├─ FAISS │    │  └─ build_llm()                 │   │
│  │    └─ LCEL  │    └────────────────────────────────┘   │
│  └─────────────┘                                          │
└──────────────────────────────────────────────────────────┘
```

**RAG pipeline:**
```
Documents → Text extraction → Chunking → Embeddings → FAISS index
                                                            │
User query ──────────────────────────────▶ Similarity search
                                                            │
                                          Top-k chunks + query
                                                            │
                                          LLM → Streamed answer
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/docchat.git
cd docchat
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

### 3. Configure a provider

| Provider | Where to get a key | Cost |
|---|---|---|
| OpenAI | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | Pay-per-use |
| Anthropic | [console.anthropic.com/keys](https://console.anthropic.com/keys) | Pay-per-use |
| Google Gemini | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | Free tier available |
| **Local Ollama** | No key — run `ollama serve` | **Free** |

**Using Ollama (completely free, fully private):**
```bash
# Install Ollama — https://ollama.com
ollama pull llama3.2      # ~2 GB download
ollama serve              # starts on localhost:11434
```
Then select **"Local (Ollama)"** in the app sidebar.

---

## Docker

```bash
# Build
docker build -t docchat .

# Run
docker run -p 8501:8501 docchat
```

With environment variables:
```bash
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=sk-... \
  docchat
```

---

## Deploy to Streamlit Community Cloud (free)

1. Fork this repo on GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your fork → branch `main` → main file `app.py`
4. Click **Deploy**

No secrets needed at deploy time — users enter their own API key in the sidebar.

---

## Deploy to a VPS (DigitalOcean / Hetzner / Fly.io)

```bash
# On your server
git clone https://github.com/YOUR_USERNAME/docchat.git
cd docchat
docker build -t docchat .
docker run -d -p 80:8501 --restart unless-stopped docchat
```

Or with Docker Compose alongside a reverse proxy — see the [Wiki](../../wiki).

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
make test

# Run with coverage
make test-cov

# Lint
make lint
```

### Project structure

```
docchat/
├── core/
│   ├── __init__.py       # Public API
│   ├── ingestion.py      # PDF/DOCX/TXT extraction + chunking
│   ├── embeddings.py     # Embedding model factory
│   └── chain.py          # FAISS vectorstore + LCEL RAG chain
├── tests/
│   ├── conftest.py       # Shared fixtures
│   ├── test_ingestion.py # Extraction + chunking tests
│   └── test_chain.py     # Chain builder tests (mocked)
├── .github/workflows/
│   └── ci.yml            # GitHub Actions: test + lint on push
├── app.py                # Streamlit UI entry point
├── llm_config.py         # Multi-provider LLM selector widget
├── pyproject.toml        # Project metadata + tool config
├── Makefile              # Developer shortcuts
├── Dockerfile            # Container image
├── .env.example          # API key template
└── requirements.txt      # Pinned dependencies
```

---

## Configuration

Copy `.env.example` to `.env` and fill in the keys you want to use:

```bash
cp .env.example .env
```

Keys are loaded automatically if present; users can also enter them directly in the app sidebar.

---

## Contributing

Pull requests welcome. Please:
1. Fork the repo and create a feature branch
2. Run `make test` and `make lint` before opening a PR
3. Keep PRs focused — one feature or fix per PR

---

## License

MIT — see [LICENSE](LICENSE).

---

*Built with [LangChain](https://langchain.com), [Streamlit](https://streamlit.io), and [FAISS](https://faiss.ai).*
