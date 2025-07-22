# pipeline/judge.py

import openai
import os
from dotenv import load_dotenv

# Load credentials from .env or directly set them here
load_dotenv()

openai.api_key = os.getenv("AZURE_API_KEY") or 'your-api-key-here'
openai.azure_endpoint = os.getenv("AZURE_API_BASE") or 'https://your-azure-endpoint.com'
openai.api_version = '2023-06-01-preview'  # This matches your working version

MODEL_NAME = 'azure.gpt-4o'  # Or another alias from your subscription

def judge_chunk(chunk: str) -> str:
    """
    Use GPT-4o via Azure to judge if a chunk should be used for fine-tuning,
    vector storage, or discarded.
    """

    prompt = f"""
Tu es un évaluateur strict de données d'entraînement pour un assistant LLM financier.

Ta tâche est de lire un extrait de texte et de choisir l'une des trois options suivantes :

- fine_tune : si le texte est structuré, pertinent et idéal pour entraîner un modèle supervisé.
- vector_only : si le texte est utile pour la recherche ou le contexte, mais pas pour l'entraînement direct.
- discard : si le texte est hors-sujet, générique ou inutilisable.

Voici le texte :

\"\"\"{chunk}\"\"\"

Réponds uniquement par fine_tune, vector_only ou discard.
"""

    response = openai.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Tu es un juge spécialisé en données pour LLM de finance."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=10,
        temperature=0.0
    )

    result = response.choices[0].message.content.strip().lower()
    
    # Normalize response
    if "fine_tune" in result:
        return "fine_tune"
    elif "vector_only" in result or "vector" in result:
        return "vector_only"
    else:
        return "discard"
