"""
agents.py — Agent médical intelligent avec outils spécialisés
Thèmes : diabète, hypertension, cholestérol

Outils intégrés :
  1. calculatrice_medicale — IMC, glycémie, calories, conversion mmol/L ↔ mg/dL
  2. meteo              — OpenWeatherMap (impact sur la tension artérielle)
  3. recherche_web      — Tavily pour les actualités médicales récentes

L'agent choisit automatiquement quel outil utiliser selon la question.
Fonction principale : ask_agent(question: str) -> str
"""

import os
import math

from dotenv import load_dotenv
load_dotenv()

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import requests

# ---------------------------------------------------------------------------
# Initialisation du LLM Mistral
# ---------------------------------------------------------------------------

llm = ChatMistralAI(
    model="mistral-large-latest",
    mistral_api_key=os.environ.get("MISTRAL_API_KEY"),
    temperature=0.2,
)

# ---------------------------------------------------------------------------
# Mémoire conversationnelle partagée
# ---------------------------------------------------------------------------

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)

# ---------------------------------------------------------------------------
# OUTIL 1 — Calculatrice médicale
# ---------------------------------------------------------------------------

def calculatrice_medicale(query: str) -> str:
    """
    Effectue des calculs médicaux courants liés au diabète, à l'hypertension
    et au cholestérol.

    Commandes reconnues (les paramètres sont séparés par des virgules) :
      imc <poids_kg>, <taille_m>
          → calcule l'IMC et sa catégorie OMS
      glycemie_mmol <valeur_mmol>
          → convertit mmol/L → mg/dL
      glycemie_mg <valeur_mg>
          → convertit mg/dL → mmol/L
      calories <proteines_g>, <glucides_g>, <lipides_g>
          → estime les calories totales
      cholesterol_mmol <valeur_mmol>
          → convertit mmol/L → mg/dL (cholestérol)
      cholesterol_mg <valeur_mg>
          → convertit mg/dL → mmol/L (cholestérol)
    """
    query = query.strip().lower()
    parts = [p.strip() for p in query.split(",")]

    try:
        # ── IMC ──────────────────────────────────────────────────────────────
        if parts[0].startswith("imc"):
            # Ex. : "imc 75, 1.75"
            tokens = parts[0].split()
            poids = float(tokens[1]) if len(tokens) > 1 else float(parts[1])
            taille = float(parts[1]) if len(tokens) > 1 else float(parts[2])
            imc = poids / (taille ** 2)

            if imc < 18.5:
                cat = "Insuffisance pondérale (< 18.5)"
            elif imc < 25:
                cat = "Poids normal (18.5 – 24.9)"
            elif imc < 30:
                cat = "Surpoids (25 – 29.9)"
            elif imc < 35:
                cat = "Obésité modérée (30 – 34.9)"
            elif imc < 40:
                cat = "Obésité sévère (35 – 39.9)"
            else:
                cat = "Obésité morbide (≥ 40)"

            risque = ""
            if imc >= 25:
                risque = " ⚠️ Un IMC élevé est un facteur de risque pour le diabète de type 2 et l'hypertension."

            return (
                f"📊 IMC = {imc:.1f} kg/m²\n"
                f"Catégorie OMS : {cat}{risque}\n"
                f"(Poids : {poids} kg, Taille : {taille} m)"
            )

        # ── Glycémie mmol/L → mg/dL ──────────────────────────────────────────
        elif parts[0].startswith("glycemie_mmol"):
            tokens = parts[0].split()
            val = float(tokens[1]) if len(tokens) > 1 else float(parts[1])
            mg = val * 18.0182

            etat = ""
            if val < 3.9:
                etat = "⚠️ Hypoglycémie"
            elif val <= 5.5:
                etat = "✅ Glycémie normale à jeun"
            elif val <= 6.9:
                etat = "⚠️ Pré-diabète (OMS)"
            else:
                etat = "🚨 Diabète probable (OMS : ≥ 7.0 mmol/L à jeun)"

            return (
                f"🩸 {val} mmol/L = {mg:.1f} mg/dL\n"
                f"Interprétation : {etat}"
            )

        # ── Glycémie mg/dL → mmol/L ──────────────────────────────────────────
        elif parts[0].startswith("glycemie_mg"):
            tokens = parts[0].split()
            val = float(tokens[1]) if len(tokens) > 1 else float(parts[1])
            mmol = val / 18.0182

            etat = ""
            if mmol < 3.9:
                etat = "⚠️ Hypoglycémie"
            elif mmol <= 5.5:
                etat = "✅ Glycémie normale à jeun"
            elif mmol <= 6.9:
                etat = "⚠️ Pré-diabète (OMS)"
            else:
                etat = "🚨 Diabète probable (OMS : ≥ 7.0 mmol/L à jeun)"

            return (
                f"🩸 {val} mg/dL = {mmol:.2f} mmol/L\n"
                f"Interprétation : {etat}"
            )

        # ── Calories ─────────────────────────────────────────────────────────
        elif parts[0].startswith("calories"):
            tokens = parts[0].split()
            proteines = float(tokens[1]) if len(tokens) > 1 else float(parts[1])
            glucides = float(parts[1]) if len(tokens) > 1 else float(parts[2])
            lipides = float(parts[2]) if len(tokens) > 1 else float(parts[3])

            cal = proteines * 4 + glucides * 4 + lipides * 9
            return (
                f"🍽️ Apport calorique estimé : {cal:.0f} kcal\n"
                f"  • Protéines : {proteines}g × 4 = {proteines*4:.0f} kcal\n"
                f"  • Glucides  : {glucides}g × 4 = {glucides*4:.0f} kcal\n"
                f"  • Lipides   : {lipides}g × 9 = {lipides*9:.0f} kcal\n"
                "Conseil : Un suivi calorique aide à gérer le diabète et le poids."
            )

        # ── Cholestérol mmol/L → mg/dL ───────────────────────────────────────
        elif parts[0].startswith("cholesterol_mmol"):
            tokens = parts[0].split()
            val = float(tokens[1]) if len(tokens) > 1 else float(parts[1])
            mg = val * 38.67

            etat = ""
            if val < 5.2:
                etat = "✅ Cholestérol total normal (< 5.2 mmol/L)"
            elif val < 6.2:
                etat = "⚠️ Limite haute (5.2 – 6.1 mmol/L)"
            else:
                etat = "🚨 Hypercholestérolémie (≥ 6.2 mmol/L)"

            return (
                f"🫀 {val} mmol/L = {mg:.1f} mg/dL\n"
                f"Interprétation : {etat}"
            )

        # ── Cholestérol mg/dL → mmol/L ───────────────────────────────────────
        elif parts[0].startswith("cholesterol_mg"):
            tokens = parts[0].split()
            val = float(tokens[1]) if len(tokens) > 1 else float(parts[1])
            mmol = val / 38.67

            etat = ""
            if mmol < 5.2:
                etat = "✅ Cholestérol total normal (< 5.2 mmol/L)"
            elif mmol < 6.2:
                etat = "⚠️ Limite haute (5.2 – 6.1 mmol/L)"
            else:
                etat = "🚨 Hypercholestérolémie (≥ 6.2 mmol/L)"

            return (
                f"🫀 {val} mg/dL = {mmol:.2f} mmol/L\n"
                f"Interprétation : {etat}"
            )

        else:
            return (
                "❓ Commande non reconnue. Commandes disponibles :\n"
                "  • imc <poids_kg>, <taille_m>\n"
                "  • glycemie_mmol <valeur> ou glycemie_mg <valeur>\n"
                "  • calories <proteines_g>, <glucides_g>, <lipides_g>\n"
                "  • cholesterol_mmol <valeur> ou cholesterol_mg <valeur>"
            )

    except (IndexError, ValueError) as e:
        return f"⚠️ Erreur de paramètres : {e}. Vérifiez le format de votre requête."


# ---------------------------------------------------------------------------
# OUTIL 2 — Météo (OpenWeatherMap)
# ---------------------------------------------------------------------------

def get_weather(city: str) -> str:
    """
    Récupère la météo actuelle via OpenWeatherMap et l'interprète
    dans un contexte médical (impact sur la tension artérielle).
    """
    api_key = os.environ.get("OPENWEATHERMAP_API_KEY")
    if not api_key:
        return (
            "⚠️ Clé API OpenWeatherMap manquante.\n"
            "Ajoutez OPENWEATHERMAP_API_KEY dans vos variables d'environnement."
        )

    city = city.strip()
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric",
        "lang": "fr",
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        pressure = data["main"]["pressure"]
        description = data["weather"][0]["description"].capitalize()
        wind_speed = data["wind"]["speed"]

        # ── Conseils médicaux contextuels ────────────────────────────────────
        conseils = []

        if temp <= 5:
            conseils.append(
                "🌡️ Froid intense : le froid provoque une vasoconstriction qui peut "
                "augmenter la tension artérielle. Couvrez-vous bien."
            )
        elif temp >= 30:
            conseils.append(
                "☀️ Chaleur élevée : la déshydratation et la chaleur peuvent affecter "
                "la glycémie et la pression artérielle. Hydratez-vous régulièrement."
            )

        if humidity >= 80:
            conseils.append(
                "💧 Humidité élevée : peut amplifier la sensation de chaleur et "
                "augmenter le risque de déshydratation chez les diabétiques."
            )

        if pressure < 1000:
            conseils.append(
                "🌀 Pression atmosphérique basse : certains patients hypertendus "
                "rapportent des maux de tête lors de baisses barométriques."
            )

        conseils_str = "\n".join(conseils) if conseils else "✅ Conditions météo sans risque particulier signalé."

        return (
            f"🌍 Météo à {city.title()} :\n"
            f"  • Conditions   : {description}\n"
            f"  • Température  : {temp}°C (ressenti {feels_like}°C)\n"
            f"  • Humidité     : {humidity}%\n"
            f"  • Pression     : {pressure} hPa\n"
            f"  • Vent         : {wind_speed} m/s\n\n"
            f"💊 Impact médical :\n{conseils_str}"
        )

    except requests.exceptions.HTTPError as e:
        if resp.status_code == 404:
            return f"❌ Ville '{city}' introuvable. Vérifiez l'orthographe."
        return f"❌ Erreur API météo : {e}"
    except requests.exceptions.RequestException as e:
        return f"❌ Impossible de contacter l'API météo : {e}"


# ---------------------------------------------------------------------------
# OUTIL 3 — Recherche web médicale (Tavily)
# ---------------------------------------------------------------------------

def recherche_web_medicale(query: str) -> str:
    """
    Recherche des actualités médicales récentes sur le diabète,
    l'hypertension et le cholestérol via l'API Tavily.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return (
            "⚠️ Clé API Tavily manquante.\n"
            "Ajoutez TAVILY_API_KEY dans vos variables d'environnement."
        )

    # Enrichir la requête avec le contexte médical si nécessaire
    medical_keywords = ["diabète", "diabetes", "hypertension", "cholestérol",
                        "cholesterol", "glycémie", "tension", "cardiovasculaire"]
    query_lower = query.lower()
    enriched = query
    if not any(kw in query_lower for kw in medical_keywords):
        enriched = f"{query} diabète hypertension cholestérol"

    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": enriched,
        "search_depth": "basic",
        "include_answer": True,
        "max_results": 4,
        "include_domains": [
            "who.int", "has-sante.fr", "ameli.fr",
            "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov",
            "lemonde.fr", "lequipe.fr", "sante.fr",
        ],
    }

    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        output_parts = [f"🔍 Résultats pour : « {query} »\n"]

        # Réponse synthétique de Tavily (si disponible)
        if data.get("answer"):
            output_parts.append(f"📝 Synthèse :\n{data['answer']}\n")

        # Résultats individuels
        results = data.get("results", [])
        if results:
            output_parts.append("📰 Sources récentes :")
            for i, r in enumerate(results[:4], 1):
                title = r.get("title", "Sans titre")
                url_r = r.get("url", "")
                snippet = r.get("content", "")[:200].replace("\n", " ")
                output_parts.append(
                    f"\n{i}. {title}\n   {snippet}…\n   🔗 {url_r}"
                )
        else:
            output_parts.append("Aucun résultat trouvé.")

        output_parts.append(
            "\n⚕️ Rappel : Ces informations sont à titre éducatif. "
            "Consultez un professionnel de santé pour tout diagnostic."
        )

        return "\n".join(output_parts)

    except requests.exceptions.HTTPError as e:
        if resp.status_code == 401:
            return "❌ Clé Tavily invalide. Vérifiez TAVILY_API_KEY."
        return f"❌ Erreur API Tavily : {e}"
    except requests.exceptions.RequestException as e:
        return f"❌ Impossible de contacter Tavily : {e}"


# ---------------------------------------------------------------------------
# Définition des outils LangChain
# ---------------------------------------------------------------------------

tools = [
    Tool(
        name="calculatrice_medicale",
        func=calculatrice_medicale,
        description=(
            "Calculatrice médicale spécialisée. Utilise cet outil pour :\n"
            "- Calculer l'IMC (Indice de Masse Corporelle)\n"
            "- Convertir la glycémie entre mmol/L et mg/dL\n"
            "- Convertir le cholestérol entre mmol/L et mg/dL\n"
            "- Estimer les calories d'un repas (protéines, glucides, lipides)\n"
            "Exemples d'entrée : 'imc 75, 1.75' | 'glycemie_mmol 6.5' | "
            "'cholesterol_mg 220' | 'calories 30, 50, 20'"
        ),
    ),
    Tool(
        name="meteo",
        func=get_weather,
        description=(
            "Récupère la météo actuelle d'une ville et analyse son impact "
            "médical sur la tension artérielle, la glycémie et les maladies "
            "cardiovasculaires. Fournis uniquement le nom de la ville en entrée. "
            "Exemples : 'Paris', 'Lyon', 'Marseille', 'New York'."
        ),
    ),
    Tool(
        name="recherche_web_medicale",
        func=recherche_web_medicale,
        description=(
            "Recherche des actualités médicales récentes sur le diabète, "
            "l'hypertension artérielle et le cholestérol. "
            "Utilise cet outil pour des questions sur les dernières études, "
            "recommandations OMS, traitements récents ou nouvelles médicales. "
            "Exemples : 'nouveaux traitements diabète type 2 2024', "
            "'recommandations OMS hypertension', 'études cholestérol LDL'."
        ),
    ),
]

# ---------------------------------------------------------------------------
# Prompt ReAct pour l'agent
# ---------------------------------------------------------------------------

REACT_PROMPT_TEMPLATE = """Tu es un assistant médical intelligent spécialisé dans le diabète, 
l'hypertension artérielle et le cholestérol. Tu aides les utilisateurs à comprendre 
leur état de santé et tu fournis des informations médicales fiables.

Tu as accès aux outils suivants :
{tools}

Pour utiliser un outil, respecte EXACTEMENT ce format :
Thought: [réflexion sur ce qu'il faut faire]
Action: [nom de l'outil parmi {tool_names}]
Action Input: [paramètre d'entrée de l'outil]
Observation: [résultat de l'outil]

Quand tu as suffisamment d'informations pour répondre :
Thought: Je connais maintenant la réponse finale.
Final Answer: [ta réponse complète et bienveillante en français]

Règles importantes :
- Réponds toujours en français
- Adopte un ton professionnel mais bienveillant
- Rappelle toujours de consulter un médecin pour tout diagnostic
- Utilise les outils pertinents selon la question
- Si aucun outil n'est nécessaire, réponds directement

Historique de conversation :
{chat_history}

Question : {input}

{agent_scratchpad}"""

react_prompt = PromptTemplate(
    input_variables=["tools", "tool_names", "chat_history", "input", "agent_scratchpad"],
    template=REACT_PROMPT_TEMPLATE,
)

# ---------------------------------------------------------------------------
# Création de l'agent ReAct
# ---------------------------------------------------------------------------

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,           # Affiche le raisonnement étape par étape
    handle_parsing_errors=True,
    max_iterations=6,       # Évite les boucles infinies
    return_intermediate_steps=False,
)

# ---------------------------------------------------------------------------
# Fonction principale — appelée par main.py et chainlit_app.py
# ---------------------------------------------------------------------------

def ask_agent(question: str) -> str:
    """
    Point d'entrée principal de l'agent.
    
    Args:
        question: La question posée par l'utilisateur (en langage naturel).
    
    Returns:
        La réponse de l'agent sous forme de chaîne de caractères.
    
    Exemple d'utilisation :
        >>> from app.agents import ask_agent
        >>> print(ask_agent("Quel est mon IMC si je pèse 80kg pour 1m75 ?"))
        >>> print(ask_agent("Quelle est la météo à Paris ?"))
        >>> print(ask_agent("Quelles sont les dernières recommandations sur le diabète ?"))
    """
    if not question or not question.strip():
        return "Bonjour ! Comment puis-je vous aider aujourd'hui ?"

    try:
        result = agent_executor.invoke({"input": question.strip()})
        return result.get("output", "Je n'ai pas pu générer de réponse.")
    except Exception as e:
        return (
            f"⚠️ Une erreur s'est produite lors du traitement de votre question : {e}\n"
            "Veuillez reformuler ou réessayer."
        )


# ---------------------------------------------------------------------------
# Test rapide en mode standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Test de l'agent médical")
    print("=" * 60)

    questions_test = [
        "Bonjour, que peux-tu faire pour moi ?",
        "Calcule mon IMC : je pèse 78 kg et je mesure 1m72",
        "Convertis une glycémie de 7.2 mmol/L en mg/dL",
        "Quelle est la météo à Paris et son impact sur l'hypertension ?",
        "Quelles sont les dernières nouvelles sur le traitement du diabète de type 2 ?",
    ]

    for q in questions_test:
        print(f"\n🙋 Question : {q}")
        print(f"🤖 Réponse  : {ask_agent(q)}")
        print("-" * 60)
