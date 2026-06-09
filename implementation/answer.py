from pathlib import Path
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document

from dotenv import load_dotenv


load_dotenv(override=True)

# Lazily initialized clients to avoid failing at import time when env vars are missing
MODEL = "gpt-4.1-nano"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")
RETRIEVAL_K = 5

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

# placeholders
embeddings = None
vectorstore = None
retriever = None
llm = None


def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question.
    """
    if retriever is None:
        init_clients()
    return retriever.invoke(question)


def combined_question(question: str, history: list[dict] = None) -> str:
    """
    Combine all the user's messages into a single string.
    """
    if history is None:
        history = []
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" + question


def answer_question(question: str, history: list[dict] = None) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """
    if history is None:
        history = []
    
    combined = combined_question(question, history)
    docs = fetch_context(combined)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)

    if llm is None:
        init_clients()

    messages = [SystemMessage(content=system_prompt)]
    
    # Manually convert Gradio history format to LangChain messages
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content, docs


def init_clients() -> None:
    """Initialize embeddings, vectorstore, retriever and llm. Raises a clear error if credentials are missing."""
    global embeddings, vectorstore, retriever, llm

    # Basic credential check for OpenAI
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI API_KEY") or os.getenv("OPENAI_ADMIN_KEY")):
        raise RuntimeError("OpenAI credentials not found. Set OPENAI_API_KEY in your environment or .env file.")

    # initialize clients
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    llm = ChatOpenAI(temperature=0, model_name=MODEL)