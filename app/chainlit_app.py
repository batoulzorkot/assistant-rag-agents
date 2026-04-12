import sys
import chainlit as cl

sys.path.append("app")
from rag_langchain import load_documents, split_documents, get_vectorstore, get_rag_chain, ask_rag

@cl.on_chat_start
async def start():
    await cl.Message(
        content="🏥 **Assistant Médical RAG** démarré !\n\nJe peux répondre à vos questions sur le **diabète**, l'**hypertension** et le **cholestérol** en me basant sur des documents officiels de l'OMS.\n\nPostez votre question !"
    ).send()

    await cl.Message(content="⏳ Chargement des documents en cours...").send()

    docs = load_documents()
    chunks = split_documents(docs)
    vectorstore = get_vectorstore(chunks)
    chain = get_rag_chain(vectorstore)

    cl.user_session.set("chain", chain)

    await cl.Message(content="✅ Documents chargés ! Posez votre question.").send()


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")

    async with cl.Step(name="🔍 Recherche dans les documents..."):
        response = ask_rag(message.content, chain)

    await cl.Message(content=response).send()
