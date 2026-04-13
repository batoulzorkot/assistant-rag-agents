import os
import sys
sys.path.append("app")

from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai.chat_models import ChatMistralAI
from langchain.memory import ConversationBufferMemory
from rag_langchain import load_documents, split_documents, get_vectorstore, get_rag_chain, ask_rag
from agents import ask_agent

# ─── Initialisation LLM ───────────────────────────────────────
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

llm = ChatMistralAI(
    model="mistral-small-latest",
    mistral_api_key=MISTRAL_API_KEY,
    temperature=0.2
)

# ─── Mémoire conversationnelle globale ────────────────────────
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# ─── Chargement RAG au démarrage ──────────────────────────────
print("🚀 Chargement du pipeline RAG...")
docs = load_documents()
chunks = split_documents(docs)
vectorstore = get_vectorstore(chunks)
rag_chain = get_rag_chain(vectorstore)
print("✅ Pipeline RAG prêt !")

# ─── Mots clés routeur ────────────────────────────────────────
RAG_KEYWORDS = [
    "diabète", "diabete", "glycémie", "glycemie", "insuline",
    "hypertension", "tension", "pression artérielle",
    "cholestérol", "cholesterol", "ldl", "hdl", "triglycérides",
    "cardiovasculaire", "traitement", "symptôme", "symptome",
    "prévention", "prevention", "document", "selon", "obésité"
]

AGENT_KEYWORDS = [
    "imc", "calories", "calcule", "convertis", "météo", "meteo",
    "temps", "température", "actualité", "actualite", "récent",
    "recent", "dernières", "dernieres", "nouvelles", "recherche",
    "mmol", "mg/dl", "poids", "taille", "kg", "km"
]

# ─── Routeur intelligent ──────────────────────────────────────
def router(question: str) -> str:
    question_lower = question.lower()

    # 1 — Question sur les documents → RAG
    if any(kw in question_lower for kw in RAG_KEYWORDS):
        print("📚 Routage → RAG")
        return ask_rag(question, rag_chain)

    # 2 — Question pour un outil → Agent
    if any(kw in question_lower for kw in AGENT_KEYWORDS):
        print("🔧 Routage → Agent")
        return ask_agent(question)

    # 3 — Conversation normale → LLM direct
    print("💬 Routage → LLM direct")
    memory.chat_memory.add_user_message(question)
    response = llm.invoke(question)
    answer = response.content
    memory.chat_memory.add_ai_message(answer)
    return answer


# ─── Boucle CLI de test ───────────────────────────────────────
if __name__ == "__main__":
    print("\n💬 Assistant médical complet prêt ! (tape 'exit' pour quitter)\n")
    while True:
        question = input("Vous : ")
        if question.lower() == "exit":
            break
        response = router(question)
        print(f"\nAssistant : {response}\n")