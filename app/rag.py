import os
import json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import faiss
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
DOCS_DIR = Path("docs")
INDEX_FILE = "faiss.index"
META_FILE = "faiss.index.meta.json"

client = MistralClient(api_key=MISTRAL_API_KEY)

# ─── 1. Lecture des PDFs ───────────────────────────────────────
def load_texts():
    texts = []
    for pdf in DOCS_DIR.rglob("*.pdf"):
        try:
            loader = PyPDFLoader(str(pdf))
            pages = loader.load()
            for page in pages:
                if page.page_content.strip():
                    texts.append({
                        "text": page.page_content.strip(),
                        "source": pdf.name
                    })
        except Exception as e:
            print(f"❌ {pdf.name} — {e}")
    print(f"✅ {len(texts)} pages chargées")
    return texts

# ─── 2. Embeddings ─────────────────────────────────────────────
def get_embedding(text: str):
    response = client.embeddings(
        model="mistral-embed",
        input=[text]
    )
    return response.data[0].embedding

# ─── 3. Index FAISS ────────────────────────────────────────────
def build_index(texts):
    if Path(INDEX_FILE).exists():
        print("✅ Index FAISS chargé")
        index = faiss.read_index(INDEX_FILE)
        with open(META_FILE, "r") as f:
            metadata = json.load(f)
        return index, metadata

    print("⏳ Création des embeddings...")
    embeddings = []
    metadata = []
    for i, item in enumerate(texts):
        emb = get_embedding(item["text"])
        embeddings.append(emb)
        metadata.append({
            "text": item["text"],
            "source": item["source"]
        })
        if i % 10 == 0:
            print(f"  {i}/{len(texts)} chunks traités")

    vectors = np.array(embeddings, dtype="float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "w") as f:
        json.dump(metadata, f, ensure_ascii=False)
    print("✅ Index FAISS sauvegardé")
    return index, metadata

# ─── 4. Réponse ────────────────────────────────────────────────
def ask(question: str, index, metadata):
    query_emb = np.array([get_embedding(question)], dtype="float32")
    _, indices = index.search(query_emb, k=3)

    context = ""
    sources = []
    for i in indices[0]:
        if i < len(metadata):
            context += metadata[i]["text"] + "\n\n"
            sources.append(metadata[i]["source"])

    response = client.chat(
        model="mistral-small-latest",
        messages=[
            ChatMessage(
                role="system",
                content="Tu es un assistant médical. Réponds uniquement en te basant sur le contexte fourni."
            ),
            ChatMessage(
                role="user",
                content=f"Contexte:\n{context}\n\nQuestion: {question}"
            )
        ]
    )

    answer = response.choices[0].message.content
    citations = "\n\n📚 Sources : " + ", ".join(set(sources))
    return answer + citations

# ─── 5. Boucle CLI ─────────────────────────────────────────────
if __name__ == "__main__":
    texts = load_texts()
    index, metadata = build_index(texts)

    print("\n💬 Assistant RAG minimal prêt ! (tape 'exit' pour quitter)\n")
    while True:
        question = input("Vous : ")
        if question.lower() == "exit":
            break
        response = ask(question, index, metadata)
        print(f"\nAssistant : {response}\n")