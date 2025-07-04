"""
Chatbot Butterfly Packaging
==========================

Interface de chat pour interagir avec le système de RAG (Retrieval Augmented Generation).
Le chatbot permet aux utilisateurs de :
- Poser des questions sur les produits
- Commander des produits en spécifiant une quantité
- Obtenir des informations sur les stocks
- Recevoir des suggestions d'alternatives si un produit est en rupture
- Discuter de tout et de rien avec l'agent IA

Le chatbot utilise Streamlit pour l'interface et s'appuie sur le module RAG pour :
- La détection des produits mentionnés
- L'extraction des quantités
- La recherche d'informations pertinentes
- La génération de réponses contextualisées
- La gestion des conversations générales

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
from agent import NINIAAgent  # Import de l'agent
import re
from rag.retrieval import _inventory_df  # Pour accéder à la liste des produits
import unidecode  # Pour gérer les accents
from datetime import datetime, timedelta
from rag.inventory_watcher import start_inventory_watcher
import threading
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()

# Initialisation de l'agent
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("La clé API OpenAI n'est pas configurée. Veuillez définir OPENAI_API_KEY dans le fichier .env")
    st.stop()

try:
    agent = NINIAAgent(api_key=api_key)
except Exception as e:
    st.error(f"Erreur lors de l'initialisation de l'agent : {str(e)}")
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

# Le thread étant daemon=True, il s'arrêtera automatiquement à la fin du programme
# Pas besoin de fonction de nettoyage explicite

def normalize_text(text):
    """
    Normalise le texte en retirant les accents et en mettant en minuscules.
    
    Args:
        text (str): Texte à normaliser
        
    Returns:
        str: Texte normalisé (sans accents, en minuscules)
    """
    return unidecode.unidecode(text.lower())

def extract_delivery_date(text):
    """
    Extrait la date de livraison souhaitée du message.
    
    La fonction recherche dans le texte :
    1. Les dates au format JJ/MM/AAAA ou JJ-MM-AAAA
    2. Les expressions comme "dans X jours" ou "pour le JJ/MM"
    
    Args:
        text (str): Message de l'utilisateur
        
    Returns:
        datetime: Date de livraison souhaitée ou None si non spécifiée
    """
    # Recherche des dates au format JJ/MM/AAAA ou JJ-MM-AAAA
    date_pattern = r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b'
    matches = re.findall(date_pattern, text)
    if matches:
        day, month, year = map(int, matches[0])
        return datetime(year, month, day)
    
    # Recherche des expressions "dans X jours"
    days_pattern = r'dans (\d+) jours?'
    days_match = re.search(days_pattern, text)
    if days_match:
        days = int(days_match.group(1))
        return datetime.now() + timedelta(days=days)
    
    # Recherche des expressions "pour le JJ/MM"
    month_day_pattern = r'pour le (\d{1,2})[/-](\d{1,2})'
    month_day_match = re.search(month_day_pattern, text)
    if month_day_match:
        day, month = map(int, month_day_match.groups())
        year = datetime.now().year
        if datetime(year, month, day) < datetime.now():
            year += 1
        return datetime(year, month, day)
    
    return None

def extract_product_and_quantity(text):
    """
    Extrait le nom du produit, la quantité, le prix et la date de livraison du message.
    
    La fonction recherche dans le texte :
    1. Le plus long nom de produit correspondant (insensible à la casse et aux accents)
    2. Le premier nombre mentionné comme quantité
    3. Le prix mentionné (précédé de € ou suivi de euros/€)
    4. La date de livraison souhaitée
    
    Args:
        text (str): Message de l'utilisateur
        
    Returns:
        tuple: (nom_produit, quantité, prix, date_livraison)
            - nom_produit (str|None): Nom du produit trouvé ou None si aucun produit détecté
            - quantité (int): Quantité demandée (0 si non spécifiée)
            - prix (float|None): Prix spécifié (None si non spécifié)
            - date_livraison (datetime|None): Date de livraison souhaitée ou None si non spécifiée
    """
    # Liste des produits connus
    products = _inventory_df['nom'].tolist()
    
    # Debug: afficher tous les produits normalisés
    print("=== PRODUITS DISPONIBLES ===")
    for p in products:
        print(f"Original: {p} -> Normalisé: {normalize_text(p)}")
    print("===========================")
    
    # Normalisation du texte d'entrée
    normalized_text = normalize_text(text)
    print(f"\n=== RECHERCHE ===")
    print(f"Texte original: {text}")
    print(f"Texte normalisé: {normalized_text}")
    
    # Recherche du produit dans le texte
    found_product = None
    max_match_length = 0
    max_match_score = 0
    
    # Préparation des mots du texte normalisé
    text_words = set(normalized_text.split())
    
    for product in products:
        normalized_product = normalize_text(product)
        product_words = set(normalized_product.split())
        
        # Calcul du score de correspondance
        # 1. Correspondance exacte
        if normalized_product in normalized_text:
            score = len(normalized_product)
            print(f"Match exact trouvé: {product} ({normalized_product})")
        else:
            # 2. Correspondance partielle basée sur les mots communs
            common_words = text_words.intersection(product_words)
            if common_words:
                # Calcul du score basé sur le nombre de mots communs et leur longueur
                score = sum(len(word) for word in common_words)
                print(f"Match partiel trouvé: {product} ({normalized_product})")
                print(f"Mots communs: {common_words}")
            else:
                score = 0
                continue
        
        # Si c'est la meilleure correspondance trouvée jusqu'ici
        if score > max_match_score:
            found_product = product
            max_match_score = score
            max_match_length = len(normalized_product)
            print(f"  -> Nouveau meilleur match (score: {score}, longueur: {max_match_length})")
    
    # Recherche de la quantité (nombres)
    quantities = re.findall(r'\b(\d+)\b', text)
    quantity = int(quantities[0]) if quantities else 0
    
    # Recherche du prix
    # Différents patterns possibles: 12€, 12 €, 12 euros, à 12€, etc.
    price_patterns = [
        r'(\d+[.,]?\d*)[ ]*€',               # 12€, 12.5€, etc.
        r'(\d+[.,]?\d*)[ ]*euros?',          # 12 euros, 12.5 euro
        r'à[ ]*(\d+[.,]?\d*)[ ]*€',          # à 12€, à 12.5€
        r'à[ ]*(\d+[.,]?\d*)[ ]*euros?',     # à 12 euros
        r'pour[ ]*(\d+[.,]?\d*)[ ]*€',       # pour 12€
        r'pour[ ]*(\d+[.,]?\d*)[ ]*euros?',  # pour 12 euros
    ]
    price = None
    for pattern in price_patterns:
        price_match = re.search(pattern, normalized_text)
        if price_match:
            # Remplace la virgule par un point pour la conversion en float
            price_str = price_match.group(1).replace(',', '.')
            price = float(price_str)
            print(f"Prix trouvé: {price}€ (pattern: {pattern})")
            break
    
    # Si aucun prix trouvé, essayer avec des expressions plus simples
    if price is None:
        # Expression pour trouver les nombres après "à" ou avant "€"
        simple_price_match = re.search(r'a (\d+)[^\d]', normalized_text)
        if simple_price_match:
            price = float(simple_price_match.group(1))
            print(f"Prix trouvé (expression simple): {price}€")
    
    # Recherche de la date de livraison
    delivery_date = extract_delivery_date(text)
    
    print(f"\n=== RÉSULTAT ===")
    print(f"Produit trouvé: {found_product}")
    print(f"Quantité trouvée: {quantity}")
    print(f"Prix spécifié: {price}€" if price else "Prix non spécifié")
    print(f"Date de livraison souhaitée: {delivery_date}")
    print("================")
    
    return found_product, quantity, price, delivery_date

def test_marge_insuffisante():
    """
    Test du système avec une marge insuffisante.
    """
    try:
        # Test avec une commande de caisses américaines double cannelure
        test_query = "Je voudrais commander 10 caisses américaines double cannelure à 12€"
        print("\n=== TEST MARGE INSUFFISANTE ===")
        print(f"Requête : {test_query}")
        
        # Utilisation de l'agent pour traiter le message
        response = agent.process_message(test_query)
        print("\nRéponse du système :")
        print(response)
        print("==============================\n")
        return response
    except Exception as e:
        print(f"Erreur lors du test : {str(e)}")
        return f"Erreur lors du test : {str(e)}"

# Configuration de la page Streamlit
st.title("Chatbot pour Butterfly Packaging")

# Ajout d'un bouton de test
if st.button("Tester le système avec une marge insuffisante"):
    print("Bouton de test cliqué")  # Debug log
    test_result = test_marge_insuffisante()
    st.markdown("### Résultat du test :")
    st.markdown(test_result)

# Initialisation de l'historique des messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage de l'historique des messages
for m in st.session_state.messages:
    role = "user" if isinstance(m, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(m.content)

# Champ de saisie pour le message utilisateur
prompt = st.chat_input("Quel est le besoin de votre client ?")

if prompt:
    # Affichage du message utilisateur
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append(HumanMessage(prompt))

    try:
        # Utilisation de l'agent pour traiter le message
        chat_history = [
            {"role": m.type, "content": m.content}
            for m in st.session_state.messages[:-1]  # Exclure le dernier message qui vient d'être ajouté
        ]
        
        # Traitement du message par l'agent
        response = agent.process_message(prompt, chat_history=chat_history)
        
        # Affichage de la réponse
        with st.chat_message("assistant"):
            st.markdown(response, unsafe_allow_html=True)
        st.session_state.messages.append(AIMessage(response))
        
    except Exception as e:
        error_message = f"Désolé, une erreur s'est produite : {str(e)}"
        with st.chat_message("assistant"):
            st.markdown(error_message)
        st.session_state.messages.append(AIMessage(error_message))