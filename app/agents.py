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
import time
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
    model="mistral-small-latest",
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
    query = query.strip().lower()
    parts = [p.strip() for p in query.split(",")]

    try:
        if parts[0].startswith("imc"):
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

        elif parts[0].startswith("glycemie_mmol"):
            tokens = parts[0].split()
            val = float(tokens[1]) if len(tokens) > 1 else float(parts[1])
            mg = val * 18.0182

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

        elif parts[0].startswith("glycemie_mg"):
            tokens = parts[0].split()
            val = float(tokens[1]) if len(tokens) > 1 else float(parts[1])
            mmol = val / 18.0182

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

        elif parts[0].startswith("cholesterol_mmol"):
            tokens = parts[0].split()
            val = float(tokens[1]) if len(tokens) > 1 else float(parts[1])
            mg = val * 38.67

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

        elif parts[0].startswith("cholesterol_mg"):
            tokens = parts[0].split()
            val = float(tokens[1]) if len(tokens) > 1 else float(parts[1])
            mmol = val / 38.67

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
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return (
            "⚠️ Clé API Tavily manquante.\n"
            "Ajoutez TAVILY_API_KEY dans vos variables d'environnement."
        )

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
            "lemonde.fr", "sante.fr",
        ],
    }

    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        output_parts = [f"🔍 Résultats pour : « {query} »\n"]

        if data.get("answer"):
            output_parts.append(f"📝 Synthèse :\n{data['answer']}\n")

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

Règles IMPORTANTES :
- Réponds toujours en français
- Adopte un ton professionnel mais bienveillant
- Rappelle toujours de consulter un médecin pour tout diagnostic
- Utilise les outils pertinents selon la question
- Si tu as la réponse sans avoir besoin d'un outil, écris DIRECTEMENT :
  Thought: Je connais la réponse.
  Final Answer: [ta réponse]
- Ne génère JAMAIS une Observation toi-même, attends toujours le résultat réel de l'outil
- Après chaque Observation, écris soit une nouvelle Action, soit la Final Answer

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
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,
    max_execution_time=30,
    return_intermediate_steps=False,
    # ✅ early_stopping_method supprimé — non supporté par cette version
)

# ---------------------------------------------------------------------------
# Fonction principale — avec retry automatique sur rate limit
# ---------------------------------------------------------------------------

def ask_agent(question: str) -> str:
    if not question or not question.strip():
        return "Bonjour ! Comment puis-je vous aider aujourd'hui ?"

    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = agent_executor.invoke({"input": question.strip()})
            return result.get("output", "Je n'ai pas pu générer de réponse.")
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate_limited" in error_str:
                wait_time = (attempt + 1) * 5
                print(f"⏳ Rate limit atteint, attente {wait_time}s...")
                time.sleep(wait_time)
                continue
            return (
                f"⚠️ Une erreur s'est produite : {e}\n"
                "Veuillez reformuler ou réessayer."
            )

    return "⚠️ Limite de requêtes atteinte. Attendez quelques secondes et réessayez."


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