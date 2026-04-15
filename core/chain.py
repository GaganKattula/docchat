"""
RAG chain — builds a FAISS vectorstore from text chunks and wires
it into a conversational retrieval chain using LangChain LCEL.
"""
from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM_PROMPT = (
    "You are a precise document assistant. Answer only from the provided context. "
    "If the answer isn't in the context, say so clearly — do not fabricate information.\n\n"
    "Context:\n{context}"
)

DEFAULT_TOP_K = 4


def build_vectorstore(chunks: list[str], embeddings):
    """
    Embed *chunks* and store them in an in-memory FAISS index.

    Parameters
    ----------
    chunks     : list of text strings to embed
    embeddings : LangChain Embeddings instance

    Returns
    -------
    FAISS vectorstore
    """
    if not chunks:
        raise ValueError("Cannot build a vectorstore from an empty chunk list.")
    return FAISS.from_texts(chunks, embeddings)


def _format_docs(docs) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vectorstore, llm, top_k: int = DEFAULT_TOP_K):
    """
    Build a stateless LCEL RAG chain.

    The caller is responsible for maintaining chat_history across turns.

    Parameters
    ----------
    vectorstore : FAISS (or any LangChain vectorstore)
    llm         : LangChain chat model
    top_k       : number of chunks to retrieve per query

    Returns
    -------
    (chain, retriever) tuple
        chain     — accepts {"question": str, "chat_history": list}
        retriever — for fetching source documents separately
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])

    chain = (
        {
            "context": lambda x: _format_docs(retriever.invoke(x["question"])),
            "chat_history": lambda x: x["chat_history"],
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever
