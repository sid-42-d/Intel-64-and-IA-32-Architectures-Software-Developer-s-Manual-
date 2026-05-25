from dotenv import load_dotenv
import os
from pathlib import Path
import gradio as gr
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
assert groq_key, "GROQ_API_KEY not set in environment"

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

Settings.llm = Groq(
    model="llama-3.1-8b-instant",
    api_key=groq_key,
)

PERSIST_DIR = "storage"

if Path(PERSIST_DIR).exists():
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
else:
    documents = SimpleDirectoryReader("Data", recursive=True).load_data()
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir=PERSIST_DIR)

query_engine = index.as_query_engine()

def ask(message: str, history: list[tuple[str, str]]):
    question = message.strip()
    if not question:
        return "Please enter a question."
    resp = query_engine.query(question)
    return str(resp)

demo = gr.ChatInterface(
    fn=ask,
    title="Intel 64 & IA-32 Architecture Bot",
    description="Ask questions based on the Intel Software Developer's Manual.",
)

if __name__ == "__main__":
    demo.launch()
