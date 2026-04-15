import os
import sys
sys.path.append("app")

from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai.chat_models import ChatMistralAI
from langchain.memory import ConversationBufferMemory
from rag_langchain import load_documents, split_documents, get_vectorstore, get_rag_chain, ask_rag
from agents import ask_agent
from pathlib import Path

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

llm = ChatMistralAI(
    model="mistral-small-latest",
    mistral_api_key=MISTRAL_API_KEY,
    temperature=0.2
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# ─── Chargement RAG au démarrage ──────────────────────────────
print("🚀 Chargement du pipeline RAG...")
# ✅ Charge depuis le disque si FAISS existe, sinon recrée
if Path("faiss_store").exists():
    vectorstore = get_vectorstore()
else:
    docs = load_documents()
    chunks = split_documents(docs)
    vectorstore = get_vectorstore(chunks)
rag_chain = get_rag_chain(vectorstore)
print("✅ Pipeline RAG prêt !")

# ✅ Garde en mémoire le dernier sujet RAG
last_rag_topic = {"question": ""}

RAG_KEYWORDS = [
    "diabète", "diabete", "glycémie", "glycemie", "insuline",
    "hypertension", "tension", "pression artérielle",
    "cholestérol", "cholesterol", "ldl", "hdl", "triglycérides",
    "cardiovasculaire", "traitement", "symptôme", "symptome",
    "prévention", "prevention", "obésité", "obese",
    "document", "selon", "d'après", "oms",
    "causes", "cause", "risque", "risques", "facteur",
    "complication", "complications", "diagnostic"
]

AGENT_KEYWORDS = [
    "imc", "calories", "calcule", "convertis", "météo", "meteo",
    "temps qu'il fait", "température extérieure", "actualité",
    "actualite", "récent", "recent", "dernières", "dernieres",
    "nouvelles", "recherche web", "mmol", "mg/dl",
    "poids", "taille", "kg", "cm"
]

CONVERSATION_KEYWORDS = [
    "bonjour", "merci", "comment vas", "résume", "resume",
    "qu'est-ce que", "c'est quoi", "explique moi", "dis moi",
    "aide moi", "tu peux", "tu es", "qui es tu", "que fais tu",
    "depuis le début", "on a dit", "on a parlé", "notre conversation",
    "conversation précédente", "precedente"
]

SUIVI_PREFIXES = [
    "et quelles", "et quel", "et comment", "et pourquoi",
    "et les", "et le", "et la", "et quoi", "et est",
    "et y a", "et qu", "et c'est", "et ça",
    "suite", "continue", "précise", "détaille",
    "plus de détail", "dis m'en plus", "approfondis"
]

# ─── Chargement historique dans mémoire ───────────────────────
def load_history_into_memory(history: list):
    for item in history[-5:]:
        memory.chat_memory.add_user_message(item["question"])
        memory.chat_memory.add_ai_message(item["response"])
    print(f"✅ {min(len(history), 5)} conversations chargées en mémoire")

# ─── Routeur intelligent ──────────────────────────────────────
def router(question: str) -> str:
    question_lower = question.lower().strip()

    # 0 — Question de suivi → enrichir avec le dernier sujet RAG
    if any(question_lower.startswith(prefix) for prefix in SUIVI_PREFIXES):
        print("📚 Routage → RAG (question de suivi)")
        if last_rag_topic["question"]:
            enriched = f"{question} (contexte : {last_rag_topic['question']})"
            print(f"🔗 Question enrichie : {enriched}")
        else:
            enriched = question
        return ask_rag(enriched, rag_chain)

    # 1 — Conversation générale → LLM direct
    if any(kw in question_lower for kw in CONVERSATION_KEYWORDS):
        print("💬 Routage → LLM direct")
        memory.chat_memory.add_user_message(question)
        response = llm.invoke(question)
        answer = response.content
        memory.chat_memory.add_ai_message(answer)
        return answer

    # 2 — Outil → Agent
    if any(kw in question_lower for kw in AGENT_KEYWORDS):
        print("🔧 Routage → Agent")
        return ask_agent(question)

    # 3 — Documents → RAG
    if any(kw in question_lower for kw in RAG_KEYWORDS):
        print("📚 Routage → RAG")
        last_rag_topic["question"] = question
        return ask_rag(question, rag_chain)

    # 4 — Défaut → LLM direct
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