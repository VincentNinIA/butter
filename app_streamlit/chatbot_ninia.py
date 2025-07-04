"""
Chatbot Butterfly Packaging - Version Modulaire NINIA
====================================================

Interface de chat pour interagir avec le système de RAG (Retrieval Augmented Generation).
Le chatbot utilise la nouvelle architecture modulaire NINIA avec LLM intégré.

Le chatbot permet aux utilisateurs de :
- Poser des questions sur les produits
- Commander des produits en spécifiant une quantité
- Obtenir des informations sur les stocks
- Recevoir des suggestions d'alternatives si un produit est en rupture
- Discuter de tout et de rien avec l'agent IA

Exemple d'utilisation :
    "Commande 10 caisses américaines double cannelure"
    "Quelles sont les caractéristiques des caisses Galia ?"
    "Je cherche une alternative aux étuis fourreau mousse"
    "Quel jour sommes-nous ?"
    "Comment vas-tu aujourd'hui ?"
"""

import sys, os
# ajoute la racine du projet dans le chemin des modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from ninia.agent import NiniaAgent  # Import de la nouvelle architecture modulaire
import re
from rag.retrieval import _inventory_df  # Pour accéder à la liste des produits
import unidecode  # Pour gérer les accents
from datetime import datetime, timedelta
from rag.inventory_watcher import start_inventory_watcher
import threading
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()

# Configuration de la page Streamlit
st.set_page_config(
    page_title="NINIA - Assistant IA Butterfly Packaging",
    page_icon="🦋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre et description
st.title("🦋 NINIA - Assistant IA Butterfly Packaging")
st.markdown("""
Assistant spécialisé dans l'analyse des commandes et la gestion d'inventaire.
Posez vos questions sur les produits, commandez en ligne, ou demandez des alternatives !
""")

# Initialisation de l'agent avec la nouvelle architecture
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("⚠️ La clé API OpenAI n'est pas configurée. Veuillez définir OPENAI_API_KEY dans le fichier .env")
    st.stop()

try:
    agent = NiniaAgent(api_key=api_key)
    st.success("✅ Agent NINIA initialisé avec succès !")
except Exception as e:
    st.error(f"❌ Erreur lors de l'initialisation de l'agent : {str(e)}")
    st.error("Veuillez vérifier votre connexion internet et la validité de votre clé API.")
    st.stop()

# Gestion du thread de surveillance d'inventaire
if not hasattr(st.session_state, 'inventory_watcher_started'):
    try:
        inventory_watcher_thread = threading.Thread(target=start_inventory_watcher, daemon=True)
        inventory_watcher_thread.start()
        st.session_state.inventory_watcher_started = True
        st.session_state.inventory_watcher_thread = inventory_watcher_thread
        print("Surveillance de l'inventaire démarrée (première initialisation)")
    except Exception as e:
        st.error(f"Erreur lors du démarrage de la surveillance de l'inventaire : {str(e)}")
        print(f"Erreur détaillée : {str(e)}")
        st.session_state.inventory_watcher_started = False

# Initialisation de l'historique de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar avec informations
with st.sidebar:
    st.header("ℹ️ Informations")
    st.markdown("""
    **Fonctionnalités disponibles :**
    - 📦 **Commandes** : "Je veux commander 10 caisses américaines"
    - 📊 **Stock** : "Quel est le stock des films étirables ?"
    - 🔄 **Alternatives** : "Alternative aux caisses Galia"
    - 💬 **Questions générales** : "Comment vas-tu ?"
    
    **Exemples de commandes :**
    - "Commande 5 rouleaux de film étirable 17 µm à 8€"
    - "Je voudrais 20 caisses américaines double cannelure"
    - "Besoin de 15 étuis fourreau mousse pour le 15/12"
    """)
    
    # Affichage du nombre de produits disponibles
    if _inventory_df is not None:
        st.metric("📦 Produits disponibles", len(_inventory_df))
    
    # Bouton pour effacer l'historique
    if st.button("🗑️ Effacer l'historique"):
        st.session_state.messages = []
        st.rerun()

# Affichage de l'historique des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Zone de saisie du message
if prompt := st.chat_input("Tapez votre message ici..."):
    # Ajout du message utilisateur à l'historique
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Affichage du message utilisateur
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Affichage du message assistant avec spinner
    with st.chat_message("assistant"):
        with st.spinner("🤔 NINIA réfléchit..."):
            try:
                # Conversion de l'historique pour le format attendu par l'agent
                chat_history = []
                for msg in st.session_state.messages[:-1]:  # Exclure le dernier message (celui qu'on traite)
                    if msg["role"] == "user":
                        chat_history.append({"role": "user", "content": msg["content"]})
                    elif msg["role"] == "assistant":
                        chat_history.append({"role": "assistant", "content": msg["content"]})
                
                # Traitement du message par l'agent
                response = agent.process_message(prompt, chat_history)
                
                # Affichage de la réponse
                st.markdown(response)
                
                # Ajout de la réponse à l'historique
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_message = f"❌ Désolé, une erreur s'est produite : {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🦋 <strong>NINIA</strong> - Assistant IA Butterfly Packaging</p>
    <p>Version modulaire avec LLM intégré</p>
</div>
""", unsafe_allow_html=True) 