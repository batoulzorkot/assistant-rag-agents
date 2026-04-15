import sys
import json
import chainlit as cl
from pathlib import Path

sys.path.append("app")
from main import router, load_history_into_memory

HISTORY_FILE = "conversations_history.json"

# ─── Gestion de l'historique ──────────────────────────────────
def load_history():
    if Path(HISTORY_FILE).exists():
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# ─── Démarrage de la conversation ─────────────────────────────
@cl.on_chat_start
async def start():
    history = load_history()
    cl.user_session.set("history", history)

    await cl.Message(
        content="🏥 **Assistant Médical RAG** démarré !\n\nJe peux répondre à vos questions sur le **diabète**, l'**hypertension** et le **cholestérol** en me basant sur des documents officiels de l'OMS.\n\nPostez votre question !"
    ).send()

    await cl.Message(content="⏳ Chargement des documents en cours...").send()

    from rag_langchain import get_vectorstore, get_rag_chain
    # ✅ Charge directement depuis le FAISS existant sur disque
    vectorstore = get_vectorstore()
    chain = get_rag_chain(vectorstore)
    cl.user_session.set("chain", chain)

    if history:
        load_history_into_memory(history)

    await cl.Message(content="✅ Documents chargés ! Posez votre question.").send()


# ─── Traitement des messages ──────────────────────────────────
@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("history")

    async with cl.Step(name="🔍 Traitement en cours..."):
        response = router(message.content)

    history.append({
        "question": message.content,
        "response": response
    })
    save_history(history)
    cl.user_session.set("history", history)

    await cl.Message(
        content=response + "\n\n---\n*⚕️ Les LLM peuvent être trompeurs. Consultez toujours un professionnel de santé.*"
    ).send()