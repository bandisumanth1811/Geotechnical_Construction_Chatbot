"""
ingest_content.py
-----------------
One-time script to ingest all PDFs in the 'content/' folder
into the existing FAISS vectorstore (merges, does NOT replace).

Usage:
    python ingest_content.py
"""

import os
import json
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────
CONTENT_DIR     = os.path.join(os.path.dirname(__file__), "content")
VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "vectorstore")
METADATA_PATH   = os.path.join(VECTORSTORE_DIR, "metadata.json")

EMBED_MODEL   = "text-embedding-3-small"
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200

# ── Helpers ──────────────────────────────────────────────────────────────────
def load_pdfs_from_dir(directory: str):
    """Read all PDFs in a directory and return LangChain Documents."""
    docs = []
    pdf_files = [f for f in sorted(os.listdir(directory)) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"⚠️  No PDFs found in {directory}")
        return docs, []

    for filename in pdf_files:
        path = os.path.join(directory, filename)
        print(f"📄 Reading: {filename} ...", end=" ", flush=True)
        try:
            reader = PdfReader(path)
            pages_added = 0
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": filename, "page": i + 1}
                    ))
                    pages_added += 1
            print(f"✅ {pages_added} pages loaded")
        except Exception as e:
            print(f"❌ Error: {e}")

    return docs, pdf_files


def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)


def load_existing_metadata():
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"built_at": None, "pdf_files": []}


def save_metadata(existing_pdfs: list, new_pdfs: list):
    combined = sorted(set(existing_pdfs + new_pdfs))
    meta = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "pdf_files": combined
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\n📋 Metadata updated: {len(combined)} total PDFs indexed.")
    return combined


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("❌ OPENAI_API_KEY not found. Check your .env file.")
        return

    print(f"\n{'='*60}")
    print("  Geotechnical RAG — Knowledge Base Ingestion")
    print(f"{'='*60}\n")

    # 1. Load embeddings
    print("🔗 Connecting to OpenAI Embeddings ...")
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=api_key)

    # 2. Load existing FAISS index
    if not os.path.isdir(VECTORSTORE_DIR):
        print(f"❌ Vectorstore not found at '{VECTORSTORE_DIR}'. Aborting.")
        return

    print("💾 Loading existing FAISS vectorstore ...")
    try:
        existing_vs = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
    except TypeError:
        existing_vs = FAISS.load_local(VECTORSTORE_DIR, embeddings)
    print("✅ Existing vectorstore loaded.")

    # 3. Read new PDFs from content/
    print(f"\n📂 Scanning '{CONTENT_DIR}' for PDFs ...\n")
    new_docs, new_pdf_files = load_pdfs_from_dir(CONTENT_DIR)

    if not new_docs:
        print("\n⚠️  No new documents to add. Exiting.")
        return

    # 4. Split into chunks
    print(f"\n✂️  Splitting {len(new_docs)} pages into chunks ...")
    chunks = split_docs(new_docs)
    print(f"✅ {len(chunks)} chunks created.")

    # 5. Embed & build new vectorstore from new chunks
    print(f"\n🔢 Embedding {len(chunks)} chunks via OpenAI (this may take a few minutes) ...")
    new_vs = FAISS.from_documents(chunks, embeddings)
    print("✅ New chunks embedded.")

    # 6. Merge into existing
    print("\n🔀 Merging new vectors into existing vectorstore ...")
    existing_vs.merge_from(new_vs)
    print("✅ Merge complete.")

    # 7. Save back to disk
    print(f"\n💾 Saving updated vectorstore to '{VECTORSTORE_DIR}' ...")
    existing_vs.save_local(VECTORSTORE_DIR)
    print("✅ Vectorstore saved.")

    # 8. Update metadata
    old_meta = load_existing_metadata()
    save_metadata(old_meta.get("pdf_files", []), new_pdf_files)

    print(f"\n{'='*60}")
    print("  ✅ Ingestion complete! The chatbot now knows about:")
    for pdf in sorted(new_pdf_files):
        print(f"     • {pdf}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
