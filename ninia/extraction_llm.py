#!/usr/bin/env python3
"""
Extraction d'informations de commande basée sur LLM
"""

import json
import logging
from typing import Optional, Tuple, Dict, Any, List
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# Modèle officiel du projet : GPT-4.1 (nom exact : 'gpt-4.1')
MODEL_NAME = "gpt-4.1"

def extract_multiple_orders_with_llm(text: str, llm_client) -> List[Dict[str, Any]]:
    """
    Extrait TOUS les produits d'un message contenant potentiellement plusieurs commandes.
    
    Args:
        text (str): Le message complet à analyser
        llm_client: Client LLM configuré
        
    Returns:
        List[Dict]: Liste des produits extraits avec leurs informations
    """
    prompt = f"""
    Tu es un assistant spécialisé dans l'extraction d'informations de commande.
    
    Analyse ce message et extrait TOUS les produits demandés au format JSON strict.
    
    Format attendu :
    {{
        "products": [
            {{
                "product_name": "nom du produit",
                "quantity": nombre_entier,
                "proposed_price": nombre_decimal
            }},
            {{
                "product_name": "autre produit",
                "quantity": nombre_entier,
                "proposed_price": nombre_decimal
            }}
        ]
    }}
    
    Règles importantes :
    - Extraire TOUS les produits du message, même s'il y en a plusieurs
    - Si une information n'est pas trouvée, utilise null
    - Le prix doit être en euros (nombre décimal)
    - La quantité doit être un nombre entier
    - Ignore les formules de politesse (Bonjour, Merci, etc.)
    - Concentre-toi uniquement sur les parties commande
    - Si le message contient des listes (avec -, •, 1., etc.), traite chaque élément
    
    Exemples :
    
    Message : "Bonjour, j'aimerais commander : - 10 caisses américaines simple cannelure à 14€ l'unité - 60 caisses américaines double cannelure à 14€ l'unité - 5 films étirable standard 15 µm à 12€ l'unité"
    
    Réponse : {{
        "products": [
            {{"product_name": "caisses américaines simple cannelure", "quantity": 10, "proposed_price": 14.0}},
            {{"product_name": "caisses américaines double cannelure", "quantity": 60, "proposed_price": 14.0}},
            {{"product_name": "films étirable standard 15 µm", "quantity": 5, "proposed_price": 12.0}}
        ]
    }}
    
    Message à analyser : "{text}"
    
    IMPORTANT : Réponds UNIQUEMENT avec le JSON, sans texte supplémentaire, sans explications.
    """
    
    try:
        response = llm_client.invoke(prompt)
        response_text = response.content.strip()
        logger.info(f"LLM response pour extraction multiple: {response_text}")
        
        # Nettoyer la réponse pour enlever tout texte avant le JSON
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Essayer de parser le JSON
        data = json.loads(response_text)
        products = data.get('products', [])
        
        logger.info(f"Extraction LLM multiple - {len(products)} produits trouvés")
        for i, product in enumerate(products):
            logger.info(f"  Produit {i+1}: {product.get('product_name')} x{product.get('quantity')} à {product.get('proposed_price')}€")
        
        return products
        
    except json.JSONDecodeError as e:
        logger.error(f"Erreur JSON lors de l'extraction LLM multiple: {e}")
        logger.error(f"Réponse LLM problématique: {response_text}")
        return []
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction LLM multiple: {e}")
        return []

def extract_order_info_with_llm(text: str, api_key: str) -> Tuple[Optional[str], Optional[int], Optional[float]]:
    """
    Extrait les informations de commande (produit, quantité, prix) en utilisant GPT-4.1 via LangChain.
    """
    llm_client = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0,
        api_key=api_key
    )
    prompt = f"""
    Analyse cette commande et extrait les informations suivantes au format JSON :
    - product_name: nom du produit (string, null si non trouvé)
    - quantity: quantité demandée (integer, null si non trouvée)
    - proposed_price: prix proposé en euros (float, null si non trouvé)
    
    Exemples de formats à reconnaître :
    - "Je veux commander 5 rouleaux de film étirable standard 17 µm à 8€"
    - "Commande de 200 Film machine sans groupe Polytech 9 µm"
    - "10 Caisses Galia"
    - "Film étirable standard 15 µm, 30 unités"
    - "combien pour 5 Etui fourreau mousse"
    
    Commande à analyser : "{text}"
    
    Réponds uniquement avec le JSON, sans texte supplémentaire.
    """
    try:
        response = llm_client.invoke(prompt)
        response_text = response.content.strip()
        logger.info(f"LLM response: {response_text}")
        data = json.loads(response_text)
        product_name = data.get('product_name')
        quantity = data.get('quantity')
        proposed_price = data.get('proposed_price')
        logger.info(f"Extraction LLM - Produit: {product_name}, Quantité: {quantity}, Prix: {proposed_price}")
        return product_name, quantity, proposed_price
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction LLM: {e}")
        return None, None, None

def extract_order_info_with_llm_fallback(text: str, llm_client, inventory_df) -> Tuple[Optional[str], Optional[int], Optional[float]]:
    """
    Version avec fallback : utilise le LLM fourni puis valide avec l'inventaire.
    """
    product_name, quantity, proposed_price = extract_order_info_with_llm_client(text, llm_client)
    if product_name:
        matched_product = find_best_product_match(product_name, inventory_df)
        if matched_product:
            return matched_product, quantity, proposed_price
        else:
            logger.warning(f"Produit extrait par LLM '{product_name}' non trouvé dans l'inventaire")
    return None, quantity, proposed_price

def extract_order_info_with_llm_client(text: str, llm_client) -> Tuple[Optional[str], Optional[int], Optional[float]]:
    """
    Extrait les informations de commande (produit, quantité, prix) en utilisant un client LLM fourni.
    """
    prompt = f"""
    Tu es un assistant spécialisé dans l'extraction d'informations de commande.
    
    Analyse cette commande et extrait les informations suivantes au format JSON strict :
    {{
        "product_name": "nom du produit",
        "quantity": nombre_entier,
        "proposed_price": nombre_decimal
    }}
    
    Règles importantes :
    - Si une information n'est pas trouvée, utilise null
    - Le prix doit être en euros (nombre décimal)
    - La quantité doit être un nombre entier
    - Ignore tout texte avant la commande (comme "Client X :")
    - Concentre-toi uniquement sur la partie commande
    
    Exemples de formats à reconnaître :
    - "Client Emeline : commande 100 caisses americaines double cannelure à 10€" → {{"product_name": "caisses americaines double cannelure", "quantity": 100, "proposed_price": 10.0}}
    - "Je veux commander 5 rouleaux de film étirable standard 17 µm à 8€" → {{"product_name": "film étirable standard 17 µm", "quantity": 5, "proposed_price": 8.0}}
    - "Commande de 200 Film machine sans groupe Polytech 9 µm" → {{"product_name": "Film machine sans groupe Polytech 9 µm", "quantity": 200, "proposed_price": null}}
    - "10 Caisses Galia" → {{"product_name": "Caisses Galia", "quantity": 10, "proposed_price": null}}
    
    Commande à analyser : "{text}"
    
    IMPORTANT : Réponds UNIQUEMENT avec le JSON, sans texte supplémentaire, sans explications.
    """
    try:
        response = llm_client.invoke(prompt)
        response_text = response.content.strip()
        logger.info(f"LLM response: {response_text}")
        
        # Nettoyer la réponse pour enlever tout texte avant le JSON
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Essayer de parser le JSON
        data = json.loads(response_text)
        product_name = data.get('product_name')
        quantity = data.get('quantity')
        proposed_price = data.get('proposed_price')
        
        logger.info(f"Extraction LLM - Produit: {product_name}, Quantité: {quantity}, Prix: {proposed_price}")
        return product_name, quantity, proposed_price
        
    except json.JSONDecodeError as e:
        logger.error(f"Erreur JSON lors de l'extraction LLM: {e}")
        logger.error(f"Réponse LLM problématique: {response_text}")
        return None, None, None
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction LLM: {e}")
        return None, None, None

def find_best_product_match(product_name: str, inventory_df) -> Optional[str]:
    """
    Trouve le meilleur match pour un nom de produit dans l'inventaire.
    """
    import unidecode
    import re
    
    # Déterminer le nom de la colonne (peut être 'nom' ou 'Nom')
    nom_column = 'Nom' if 'Nom' in inventory_df.columns else 'nom'
    
    # Normaliser le nom du produit recherché (pluriel vers singulier + nettoyage)
    product_normalized = unidecode.unidecode(product_name.lower().strip())
    
    # Conversion pluriel → singulier pour les termes clés
    product_normalized = re.sub(r'\bfilms\b', 'film', product_normalized)
    product_normalized = re.sub(r'\bcaisses\b', 'caisse', product_normalized)
    product_normalized = re.sub(r'\betuis\b', 'etui', product_normalized)
    
    # Chercher d'abord avec la colonne nom_normalise si elle existe
    if 'nom_normalise' in inventory_df.columns:
        for _, row in inventory_df.iterrows():
            inventory_normalized = str(row['nom_normalise']).lower()
            
            # Match exact normalisé
            if product_normalized == inventory_normalized:
                return row[nom_column]
            
            # Pour les produits avec spécifications techniques (µm, mm, etc.)
            # Vérifier que les spécifications correspondent exactement
            product_specs = re.findall(r'\d+\s*(?:[µμ]?m|um)', product_normalized)
            inventory_specs = re.findall(r'\d+\s*(?:[µμ]?m|um)', inventory_normalized)
            
            if product_specs and inventory_specs:
                # Si il y a des spécifications, elles doivent correspondre exactement
                if product_specs == inventory_specs:
                    # Vérifier aussi que les mots-clés principaux correspondent
                    product_base = re.sub(r'\d+\s*(?:[µμ]?m|um)', '', product_normalized).strip()
                    inventory_base = re.sub(r'\d+\s*(?:[µμ]?m|um)', '', inventory_normalized).strip()
                    if product_base in inventory_base or inventory_base in product_base:
                        return row[nom_column]
            else:
                # Pas de spécifications techniques, utiliser le matching classique
                # Match partiel flexible (contenant ou contenu)
                if product_normalized in inventory_normalized or inventory_normalized in product_normalized:
                    return row[nom_column]
                
                # Match par mots-clés principaux
                product_words = set(product_normalized.split())
                inventory_words = set(inventory_normalized.split())
                common_words = product_words.intersection(inventory_words)
                
                # Si au moins 70% des mots clés correspondent
                if len(common_words) >= max(1, int(0.7 * len(product_words))):
                    return row[nom_column]
    
    # Fallback sur la colonne nom avec normalisation à la volée
    for _, row in inventory_df.iterrows():
        inventory_name = str(row[nom_column])
        inventory_normalized = unidecode.unidecode(inventory_name.lower().strip())
        
        # Match exact normalisé
        if product_normalized == inventory_normalized:
            return inventory_name
        
        # Match partiel flexible
        if product_normalized in inventory_normalized or inventory_normalized in product_normalized:
            return inventory_name
        
        # Match par mots-clés principaux
        product_words = set(product_normalized.split())
        inventory_words = set(inventory_normalized.split())
        common_words = product_words.intersection(inventory_words)
        
        # Si au moins 70% des mots clés correspondent
        if len(common_words) >= max(1, int(0.7 * len(product_words))):
            return inventory_name
    
    return None 