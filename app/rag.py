#!/usr/bin/env python
"""
rag.py — Script RAG minimal
Utilise Mistral + FAISS + LangChain
"""

from __future__ import annotations
import os, json, textwrap
from pathlib import Path

import faiss, numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from tqdm.auto import tqdm
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI

load_dotenv()

# ─────────────────────────── Config ──────────────────────────────
DOCS_DIR      = Path("docs")
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150
TOP_K         = 4
FAISS_INDEX   = "faiss.index"
FAISS_META    = "faiss.index.meta.json"

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

SYSTEM_PROMPT = (
    "Tu es un assistant médical précis et concis spécialisé dans le diabète, "
    "l'hypertension et le cholestérol. "
    "Réponds UNIQUEMENT à partir du contexte fourni. "
    "Si la réponse est absente, dis 'Je ne sais pas.'"
)

# ─────────────────────────────────────────────────────────────────

# 1️⃣ Chargement & découpage
def read_pdf_text(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        pages_text = []
        for page in reader.pages:
            try:
                text = page.extract_text()
                if text:
                    pages_text.append(text)
            except Exception:
                pass  # ✅ ignore les pages corrompues
        return "\n".join(pages_text)
    except Exception as e:
        print(f"⚠️ PDF ignoré ({path.name}) : {e}")
        return ""

def load_and_split() -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks: list[str] = []
    for path in DOCS_DIR.rglob("*"):
        if path.suffix.lower() == ".txt":
            text = path.read_text(encoding="utf-8", errors="ignore")
        elif path.suffix.lower() == ".pdf":
            text = read_pdf_text(path)
        else:
            continue
        if text.strip():
            chunks.extend(splitter.split_text(text))
    if not chunks:
        raise RuntimeError(f"Aucun fichier .txt ou .pdf trouvé dans {DOCS_DIR}")
    print(f"✅ {len(chunks)} chunks créés")
    return chunks

# 2️⃣ Embeddings Mistral
def embed(texts: list[str]) -> list[list[float]]:
    embedder = MistralAIEmbeddings(
        api_key=MISTRAL_API_KEY,
        model="mistral-embed"
    )
    return embedder.embed_documents(texts)

# 3️⃣ FAISS store
def get_faiss_store(chunks: list[str]):
    if Path(FAISS_INDEX).exists() and Path(FAISS_META).exists():
        print("✅ FAISS store chargé depuis le disque")
        index = faiss.read_index(FAISS_INDEX)
        chunks = json.loads(Path(FAISS_META).read_text())
        return index, chunks

    print("⏳ Construction du vector store...")
    all_vectors = []
    for i in tqdm(range(0, len(chunks), 32), unit="batch"):
        all_vectors.extend(embed(chunks[i:i+32]))
    mat = np.asarray(all_vectors, dtype=np.float32)

    index = faiss.IndexFlatL2(mat.shape[1])
    index.add(mat)
    faiss.write_index(index, FAISS_INDEX)
    Path(FAISS_META).write_text(json.dumps(chunks))
    print("✅ FAISS store créé et sauvegardé")
    return index, chunks

# 4️⃣ Récupération
def retrieve(query: str, index, chunks, k: int = TOP_K) -> list[str]:
    embedder = MistralAIEmbeddings(
        api_key=MISTRAL_API_KEY,
        model="mistral-embed"
    )
    q_vec = np.asarray(
        embedder.embed_query(query), dtype=np.float32
    ).reshape(1, -1)
    _, idxs = index.search(q_vec, k)
    return [chunks[i] for i in idxs[0]]

# 5️⃣ Prompt
def build_user_prompt(question: str, ctx_chunks: list[str]) -> str:
    context_block = "\n\n".join(
        f"[Doc {i+1}]\n{chunk}" for i, chunk in enumerate(ctx_chunks)
    )
    return (
        f"Contexte :\n{context_block}\n\n"
        f"Question : {question}\nRéponse :"
    )

# 6️⃣ Boucle de chat
def chat_loop(index, chunks):
    llm = ChatMistralAI(
        api_key=MISTRAL_API_KEY,
        model="mistral-small-latest",
        temperature=0.2
    )
    history: list[dict] = []

    while True:
        try:
            q = input("\n💬 Question (Ctrl-C pour quitter) : ")
        except KeyboardInterrupt:
            print("\nAu revoir !")
            break

        ctx = retrieve(q, index, chunks)
        user_prompt = build_user_prompt(q, ctx)

        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + history
            + [{"role": "user", "content": user_prompt}]
        )

        print("\n🔍 Contexte récupéré :")
        print("─" * 60)
        for i, c in enumerate(ctx, 1):
            print(textwrap.indent(textwrap.fill(c, width=88), f"[Doc {i}] "))
        print("─" * 60)

        response = llm.invoke(messages)
        answer = response.content

        print("\n🤖 Réponse :\n")
        print(textwrap.fill(answer, width=88))

        history.extend([
            {"role": "user", "content": q},
            {"role": "assistant", "content": answer},
        ])

# ─── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    chunks = load_and_split()
    index, chunks = get_faiss_store(chunks)
    chat_loop(index, chunks)