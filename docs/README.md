# Assistant Médical Intelligent — RAG + Agents (LangChain + Chainlit)

Ce projet est un assistant médical intelligent combinant **RAG (Retrieval-Augmented Generation)** et des **agents avec outils**, spécialisé sur le diabète, l'hypertension et le cholestérol.

---

## Contenu du projet

```
assistant-rag-agents/
├── app/
│   ├── rag_langchain.py       # Pipeline RAG avec mémoire conversationnelle (LangChain + FAISS)
│   ├── agents.py              # Agent ReAct avec 3 outils (calculatrice, météo, recherche web)
│   ├── main.py                # Routeur intelligent RAG / Agent / LLM direct
│   ├── chainlit_app.py        # Interface web Chainlit
│   └── ui.py                  # Interface CLI de test
├── docs/                      # Documents PDF sources (OMS)
│   ├── diabete/
│   ├── hypertension/
│   └── cholesterol/
├── faiss_store/               # Index vectoriel FAISS (généré automatiquement)
├── .env.example               # Exemple de configuration des clés API
├── .chainlit/config.toml      # Configuration Chainlit
├── requirements.txt           # Dépendances Python
└── README.md
```

---

## Fonctionnalités

### Partie 1 — RAG
- Ingestion de documents PDF via `PyPDFLoader`
- Découpage en chunks et vectorisation avec `MistralAIEmbeddings`
- Stockage et recherche via **FAISS**
- Réponses générées par **Mistral** via LangChain avec **citations de sources**

### Partie 2 — Agents & Outils
L'agent choisit automatiquement l'outil adapté selon la question :

| Outil | Description |
|-------|-------------|
| `calculatrice_medicale` | IMC, glycémie (mmol/L ↔ mg/dL), cholestérol, calories |
| `meteo` | Météo en temps réel (OpenWeatherMap) + impact médical |
| `recherche_web_medicale` | Actualités médicales récentes (Tavily) |

### Partie 3 — Routeur intelligent
```
Question médicale sur les documents  →  RAG (avec citations)
Question nécessitant un calcul/outil →  Agent
Bonjour / conversation simple        →  LLM direct
```

### Partie 4 — Finition
- Mémoire conversationnelle (suivi du contexte)
- Historique sauvegardé dans `conversations_history.json`
- Interface web **Chainlit**

---

## Prérequis

- Python 3.10+
- Clés API : Mistral, OpenWeatherMap, Tavily

---

## Installation

### 1. Cloner le repo

```bash
git clone https://github.com/batoulzorkot/assistant-rag-agents.git
cd assistant-rag-agents
```

### 2. Créer et activer un environnement virtuel

**macOS / Linux :**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows PowerShell :**
```bash
python -m venv venv
.\.venv\Scripts\Activate.ps1
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer les clés API

Créez un fichier `.env` à la racine du projet :

```
MISTRAL_API_KEY=votre_clé_mistral
OPENWEATHERMAP_API_KEY=votre_clé_openweather
TAVILY_API_KEY=votre_clé_tavily
```

> Voir `.env.example` pour le format exact.

---

## Ajouter vos documents

Placez vos fichiers PDF dans le dossier `docs/` (sous-dossiers possibles) :

```
docs/
├── diabete/
│   └── rapport_diabete_oms.pdf
├── hypertension/
│   └── fiche_hypertension.pdf
└── cholesterol/
    └── guide_cholesterol.pdf
```

> Le pipeline charge automatiquement tous les `.pdf` dans `docs/` et ses sous-dossiers.

---

## Lancer l'application

### Interface Chainlit (recommandée)

```bash
chainlit run app/chainlit_app.py
```

Ouvrez ensuite : [http://localhost:8000](http://localhost:8000)

### CLI de test (terminal)

```bash
python app/rag_langchain.py
```

---

## Exemples de questions

**RAG — documents :**
```
Quels sont les symptômes du diabète ?
Comment prévenir l'hypertension ?
Quels sont les traitements du cholestérol élevé ?
```

**Agent — outils :**
```
Calcule mon IMC : je pèse 75kg et je mesure 1m70
Convertis une glycémie de 7.2 mmol/L en mg/dL
Quelle est la météo à Paris et son impact sur la tension ?
Quelles sont les dernières recommandations OMS sur le diabète ?
```

**LLM direct :**
```
Bonjour !
Merci pour ton aide
```

---

## Notes techniques

- Le FAISS store est créé dans `faiss_store/` au premier lancement, puis rechargé automatiquement.
- L'historique des conversations est sauvegardé dans `conversations_history.json`.
- En cas d'erreur **rate limit Mistral**, l'agent réessaie automatiquement (jusqu'à 3 fois).
- La clé OpenWeatherMap peut mettre jusqu'à 2h à s'activer après création du compte.

---

## Dépendances principales

```
chainlit
faiss-cpu
langchain
langchain-community
langchain-mistralai
langchain-text-splitters
pypdf
python-dotenv
requests
```
