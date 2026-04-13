"""
ui.py — Interface Chainlit pour l'assistant médical intelligent
Combine le pipeline RAG (rag_pipeline.py) et l'agent (agents.py)

Logique de routage :
  - Question sur les documents (diabète, hypertension, cholestérol) → RAG
  - Question nécessitant un outil (calcul, météo, recherche web)    → Agent
  - Conversation simple                                              → LLM direct
"""

import chainlit as cl
from dotenv import load_dotenv
from agents import ask_agent

load_dotenv()

# Mots-clés qui indiquent une question pour l'agent (outils)
AGENT_KEYWORDS = [
    "imc", "calcule", "calcul", "convertis", "conversion",
    "glycémie", "mmol", "mg/dl", "calories", "cholestérol",
    "météo", "temps", "température", "ville",
    "actualités", "nouvelles", "récent", "étude", "recherche",
    "traitement", "recommandation", "2024", "2025",
]

# Mots-clés qui indiquent une question pour le RAG (documents OMS)
RAG_KEYWORDS = [
    "selon", "document", "oms", "fichier", "pdf",
    "définition", "qu'est-ce que", "c'est quoi",
    "symptôme", "cause", "facteur de risque",
    "prévention", "complication", "diagnostic",
]


def route_question(question: str) -> str:
    """
    Détermine si la question doit être traitée par l'agent ou le RAG.
    Retourne 'agent' ou 'rag'.
    """
    question_lower = question.lower()

    # Vérifie d'abord les mots-clés agent (priorité aux outils)
    for keyword in AGENT_KEYWORDS:
        if keyword in question_lower:
            return "agent"

    # Ensuite vérifie les mots-clés RAG
    for keyword in RAG_KEYWORDS:
        if keyword in question_lower:
            return "rag"

    # Par défaut : agent (conversation + outils)
    return "agent"


@cl.on_chat_start
async def on_chat_start():
    """Message de bienvenue au démarrage du chat."""
    await cl.Message(
        content=(
            "👋 Bonjour ! Je suis votre assistant médical intelligent, "
            "spécialisé dans le **diabète**, l'**hypertension artérielle** "
            "et le **cholestérol**.\n\n"
            "Je peux vous aider à :\n"
            "- 📊 **Calculer** votre IMC, convertir votre glycémie ou cholestérol\n"
            "- 🌤️ **Analyser** l'impact de la météo sur votre tension artérielle\n"
            "- 🔍 **Rechercher** les dernières actualités médicales\n"
            "- 📄 **Consulter** les documents OMS sur ces pathologies\n\n"
            "⚠️ *Je ne remplace pas un médecin. "
            "Consultez toujours un professionnel de santé pour un diagnostic.*\n\n"
            "**Comment puis-je vous aider aujourd'hui ?**"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Traite chaque message de l'utilisateur."""
    question = message.content.strip()

    if not question:
        await cl.Message(content="Veuillez poser une question.").send()
        return

    # Affiche un indicateur de chargement
    async with cl.Step(name="Traitement en cours...") as step:

        route = route_question(question)

        if route == "rag":
            step.output = "Recherche dans les documents OMS..."
            try:
                from app.rag_pipeline import ask_rag
                response = ask_rag(question)
            except ImportError:
                response = ask_agent(question)
            except Exception as e:
                response = f"⚠️ Erreur RAG : {e}\n\n" + ask_agent(question)

        else:
            step.output = "Analyse de votre question..."
            response = ask_agent(question)

    await cl.Message(content=response).send()
