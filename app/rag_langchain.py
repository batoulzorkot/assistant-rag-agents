import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
DOCS_DIR = Path("docs")
FAISS_DIR = "faiss_store"

# ─── 1. Chargement des documents ───────────────────────────────
def load_documents():
    docs = []
    for pdf in DOCS_DIR.rglob("*.pdf"):
        try:
            loader = PyPDFLoader(str(pdf))
            pages = loader.load()
            docs.extend(pages)
            print(f"✅ {pdf.name} — {len(pages)} pages")
        except Exception as e:
            print(f"❌ {pdf.name} — ERREUR: {e}")
    print(f"\n📊 Total: {len(docs)} pages chargées")
    return docs

# ─── 2. Découpage en chunks ────────────────────────────────────
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ {len(chunks)} chunks créés")
    return chunks

# ─── 3. Création ou chargement du FAISS ───────────────────────
def get_vectorstore(chunks=None):
    embeddings = MistralAIEmbeddings(
        api_key=MISTRAL_API_KEY,
        model="mistral-embed"
    )
    # ✅ Si FAISS existe déjà → chargement instantané
    if Path(FAISS_DIR).exists():
        print("✅ FAISS store chargé depuis le disque")
        return FAISS.load_local(
            FAISS_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
    # ✅ Sinon → création depuis les chunks
    if chunks is None:
        print("⚠️ FAISS inexistant, chargement des documents...")
        docs = load_documents()
        chunks = split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_DIR)
    print("✅ FAISS store créé et sauvegardé")
    return vectorstore

# ─── 4. Chaîne RAG avec mémoire conversationnelle ─────────────
def get_rag_chain(vectorstore):
    llm = ChatMistralAI(
        api_key=MISTRAL_API_KEY,
        model="mistral-small-latest",
        temperature=0.2
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True
    )
    return chain

# ─── 5. Réponse avec citations intelligentes ──────────────────
def ask_rag(question: str, chain) -> str:
    result = chain.invoke({"question": question})
    answer = result["answer"]
    sources = result["source_documents"]

    phrases_hors_sujet = [
        "je ne sais pas",
        "pas dans le contexte",
        "pas mentionné",
        "n'est pas mentionné",
        "d'après les informations fournies",
        "le contexte ne",
        "pas d'information",
        "je n'ai pas",
        "aucune information"
    ]

    if not sources or any(phrase in answer.lower() for phrase in phrases_hors_sujet):
        return answer

    # ✅ Déduplication des sources
    seen = set()
    unique_sources = []
    for doc in sources:
        source_name = Path(doc.metadata.get("source", "Inconnu")).name
        page = doc.metadata.get("page", "?")
        key = f"{source_name}-{page}"
        if key not in seen:
            seen.add(key)
            unique_sources.append(doc)

    citations = "\n\n📚 Sources :\n"
    for i, doc in enumerate(unique_sources):
        source_name = Path(doc.metadata.get("source", "Inconnu")).name
        page = doc.metadata.get("page", "?")
        citations += f"  [{i+1}] {source_name} — page {page}\n"

    return answer + citations

# ─── 6. Boucle CLI de test ─────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Chargement des documents...")
    vectorstore = get_vectorstore()
    chain = get_rag_chain(vectorstore)

    print("\n💬 Assistant médical RAG prêt ! (tape 'exit' pour quitter)\n")
    while True:
        question = input("Vous : ")
        if question.lower() == "exit":
            break
        response = ask_rag(question, chain)
        print(f"\nAssistant : {response}\n")