"""
Module Core du système RAG
=========================

Ce module gère la logique principale du système de Retrieval Augmented Generation (RAG).
Il est responsable de :
- La génération des réponses en utilisant le contexte récupéré
- L'analyse et la sélection des alternatives en cas de rupture de stock
- Le formatage des informations produits pour le LLM

Le module utilise LangChain pour :
- L'interaction avec le modèle GPT-4
- La gestion des prompts système et utilisateur
- Le formatage des messages

Architecture :
- Le module récupère les informations via retrieval.py
- Il formate ces informations pour le LLM
- Il gère la logique de sélection des alternatives
- Il génère des réponses contextualisées
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from .retrieval import fetch_docs, get_stock, _inventory_df, ProductInfo
from .prompt import SYSTEM_PROMPT
from .settings import settings
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import re

_llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=1,
    api_key=settings.openai_api_key,
)

def check_margin(product: ProductInfo, quantity: int, prix_propose: Optional[float] = None) -> tuple[bool, float]:
    """
    Vérifie si la marge est suffisante pour la quantité demandée.
    
    Args:
        product (ProductInfo): Informations du produit
        quantity (int): Quantité demandée
        prix_propose (Optional[float]): Prix proposé par l'utilisateur, si spécifié
        
    Returns:
        tuple[bool, float]: (marge_suffisante, marge_actuelle)
    """
    if product['prix_achat'] == 0:
        return True, 0.0
        
    # Si un prix est proposé, on l'utilise pour calculer la marge réelle
    prix_vente = prix_propose if prix_propose is not None else product['prix_vente_conseille']
    
    if prix_vente == 0:
        return True, 0.0
        
    marge_actuelle = prix_vente - product['prix_achat']
    return marge_actuelle >= product['marge_minimum'], marge_actuelle

def format_product_context(product: ProductInfo, is_alternative: bool = False) -> str:
    """
    Formate les informations d'un produit pour le contexte du LLM.
    
    Args:
        product (ProductInfo): Informations du produit à formater
        is_alternative (bool): Indique s'il s'agit d'une alternative
        
    Returns:
        str: Texte formaté avec les informations du produit
    """
    prefix = "ALTERNATIVE :" if is_alternative else "PRODUIT DEMANDÉ :"
    context = (
        f"{prefix}\n"
        f"Nom : {product['name']}\n"
        f"Stock initial : {product['stock_initial']}\n"
        f"Commandes à livrer : {product['commandes']}\n"
        f"Stock disponible : {product['stock_disponible']}\n"
        f"Délai de réapprovisionnement : {product['delai']}\n"
    )
    
    # Ajout des informations de prix si disponibles
    if product['prix_achat'] > 0 and product['prix_vente_conseille'] > 0:
        context += (
            f"Prix d'achat : {product['prix_achat']}€\n"
            f"Prix de vente conseillé : {product['prix_vente_conseille']}€\n"
            f"Marge minimum requise : {product['marge_minimum']}€\n"
        )
    
    if product.get('description'):
        context += f"\nDescription détaillée :\n{product['description']}\n"
    return context

def analyze_alternatives(product: ProductInfo, alternatives: List[ProductInfo], required_qty: int) -> Optional[ProductInfo]:
    """
    Analyse les alternatives disponibles et sélectionne la meilleure.
    
    Le choix se fait selon :
    1. La similarité avec le produit demandé
    2. La disponibilité en stock
    3. Le délai de réapprovisionnement
    4. Les caractéristiques techniques
    
    Args:
        product (ProductInfo): Produit initialement demandé
        alternatives (List[ProductInfo]): Liste des alternatives disponibles
        required_qty (int): Quantité demandée
        
    Returns:
        Optional[ProductInfo]: Meilleure alternative ou None si aucune alternative
    """
    if not alternatives:
        return None
        
    # Création d'un prompt pour l'analyse
    analysis_prompt = (
        "En tant qu'expert en emballages, analyse le produit demandé et ses alternatives pour "
        "sélectionner la meilleure option de remplacement.\n\n"
        f"{format_product_context(product)}\n"
        f"Quantité demandée : {required_qty}\n\n"
        "ALTERNATIVES DISPONIBLES :\n"
    )
    
    for alt in alternatives:
        analysis_prompt += f"\n{format_product_context(alt, is_alternative=True)}"
    
    analysis_prompt += (
        "\nSélectionne la meilleure alternative en considérant :\n"
        "1. La similarité avec le produit demandé\n"
        "2. Le stock disponible\n"
        "3. Le délai de réapprovisionnement\n"
        "4. Les caractéristiques techniques\n\n"
        "Réponds uniquement avec le nom de la meilleure alternative."
    )
    
    # Demande au LLM de choisir la meilleure alternative
    system = SystemMessage(content="Tu es un expert en emballages. Réponds uniquement avec le nom du produit choisi.")
    user = HumanMessage(content=analysis_prompt)
    best_alternative_name = _llm.invoke([system, user]).content.strip()
    
    # Retrouve l'alternative sélectionnée
    for alt in alternatives:
        if alt['name'].lower() in best_alternative_name.lower():
            return alt
            
    return alternatives[0]  # Par défaut, retourne la première alternative si pas de correspondance

def check_delivery_time(product: ProductInfo, delivery_date: Optional[datetime] = None) -> bool:
    """
    Vérifie si le délai de réapprovisionnement du produit est compatible avec la date souhaitée.
    
    Args:
        product (ProductInfo): Informations du produit
        delivery_date (Optional[datetime]): Date de réapprovisionnement souhaitée
        
    Returns:
        bool: True si le délai est compatible, False sinon
    """
    if not delivery_date:
        return True
        
    # Extraction du nombre de jours du délai de réapprovisionnement
    delai_str = product['delai']
    jours_match = re.search(r'(\d+)\s*jours?', delai_str)
    if not jours_match:
        return True  # Si on ne peut pas extraire le délai, on considère que c'est compatible
        
    delai_jours = int(jours_match.group(1))
    date_rapprovisionnement_estimee = datetime.now() + timedelta(days=delai_jours)
    
    return date_rapprovisionnement_estimee <= delivery_date

def answer(question: str, product_id: str = None, required_qty: int = 0, 
          delivery_date: Optional[datetime] = None, prix_propose: Optional[float] = None) -> str:
    """
    Génère une réponse contextualisée à la question de l'utilisateur.
    
    Le processus est le suivant :
    1. Récupération des informations produit et alternatives
    2. Vérification de la disponibilité en stock
    3. Vérification de la compatibilité du délai de réapprovisionnement
    4. Vérification de la marge (en utilisant le prix proposé si disponible)
    5. Sélection d'alternatives si nécessaire
    6. Génération d'une réponse avec le contexte approprié
    
    Args:
        question (str): Question ou demande de l'utilisateur
        product_id (str, optional): Identifiant du produit demandé
        required_qty (int, optional): Quantité demandée
        delivery_date (Optional[datetime], optional): Date de réapprovisionnement souhaitée
        prix_propose (Optional[float], optional): Prix proposé par l'utilisateur
        
    Returns:
        str: Réponse générée par le LLM
        
    Raises:
        ValueError: Si aucune information pertinente n'est trouvée
    """
    try:
        # Passer le prix proposé directement au RAG pour qu'il cherche des alternatives si marge insuffisante
        result = fetch_docs(question, product_id=product_id, required_qty=required_qty, prix_propose=prix_propose)
    except ValueError:
        return ("Je n'ai pas trouvé d'informations pertinentes dans la base documentaire. "
                "Pouvez‑vous reformuler ou préciser votre demande ?")

    if not result["produit"]:
        return "Je n'ai pas trouvé d'informations sur ce produit."

    produit = result["produit"]
    alternatives = result["alternatives"]
    
    # Construction du contexte
    context_parts = [format_product_context(produit)]
    
    # Si un prix est proposé, l'ajouter au contexte
    if prix_propose is not None:
        context_parts.append(f"\nPrix proposé par le client : {prix_propose}€ (prix conseillé : {produit['prix_vente_conseille']}€)")
    
    # Vérification du délai de réapprovisionnement
    delai_compatible = check_delivery_time(produit, delivery_date)
    if not delai_compatible:
        context_parts.append("\nATTENTION : Le délai de réapprovisionnement n'est pas compatible avec la date souhaitée.")
    
    # Vérification de la marge avec le prix proposé si disponible
    marge_suffisante, marge_actuelle = check_margin(produit, required_qty, prix_propose)
    if not marge_suffisante:
        # Avertissement de marge insuffisante dans le contexte pour le LLM
        prix_utilise = prix_propose if prix_propose is not None else produit['prix_vente_conseille']
        context_parts.append(
            f"\nATTENTION : La marge actuelle ({marge_actuelle}€) est inférieure à la marge minimum requise ({produit['marge_minimum']}€)."
        )
        # Debug : message de recherche d'alternatives
        print(f"\nMarge insuffisante ({marge_actuelle}€) pour le produit demandé. Recherche d'alternatives avec une meilleure marge...")
        print(f"Prix d'achat: {produit['prix_achat']}€, Prix utilisé: {prix_utilise}€, Marge minimum: {produit['marge_minimum']}€")
    
    # Si stock insuffisant, délai incompatible ou marge insuffisante, on ajoute les alternatives au contexte
    if (produit["stock_disponible"] < required_qty or not delai_compatible or not marge_suffisante):
        if alternatives:
            context_parts.append("\nINFORMATIONS SUR LES ALTERNATIVES :")
            
            # Tableau récapitulatif des alternatives pour faciliter la comparaison
            tableau_alternatives = "Tableau comparatif des alternatives :\n"
            tableau_alternatives += "Nom | Stock disponible | Délai | Prix achat | Prix vente | Marge min | Marge avec prix proposé | Marge suffisante\n"
            tableau_alternatives += "----|-----------------|-------|------------|-----------|-----------|---------------------|----------------\n"
            
            for alt in alternatives:
                # On vérifie que la marge de l'alternative est bien suffisante avant de l'inclure
                alt_marge_suffisante, alt_marge_actuelle = check_margin(alt, required_qty, prix_propose)
                
                # Ligne du tableau récapitulatif
                marge_ok = "Oui" if alt_marge_suffisante else "Non"
                tableau_alternatives += f"{alt['name']} | {alt['stock_disponible']} | {alt['delai']} | {alt['prix_achat']}€ | {alt['prix_vente_conseille']}€ | {alt['marge_minimum']}€ | {alt_marge_actuelle}€ | {marge_ok}\n"
                
                # Préparation du contexte détaillé de l'alternative
                alt_context = format_product_context(alt, is_alternative=True)
                
                # Ajout des informations spécifiques sur la marge
                if prix_propose is not None:
                    alt_context += (
                        f"Prix proposé par le client : {prix_propose}€\n"
                        f"Marge avec prix proposé : {alt_marge_actuelle}€\n"
                        f"Marge minimum requise : {alt['marge_minimum']}€\n"
                        f"Marge suffisante : {'Oui' if alt_marge_suffisante else 'Non'}\n"
                    )
                
                # Ajouter des informations sur les caractéristiques techniques si disponibles
                if alt.get('similarite_technique') is not None:
                    alt_context += f"Similarité technique : {alt.get('similarite_technique', 0):.2%}\n"
                
                # Ajout des informations détaillées du produit
                if alt.get('description'):
                    try:
                        # On essaie de parser le JSON de la description
                        import json
                        desc_json = json.loads(alt['description'])
                        
                        # Ajout des informations techniques
                        if 'technical_details' in desc_json:
                            tech = desc_json['technical_details']
                            alt_context += "\nCaractéristiques techniques :\n"
                            if 'conception' in tech:
                                alt_context += f"Conception : {tech['conception']}\n"
                            if 'types_cannelure' in tech:
                                alt_context += "Types de cannelure :\n"
                                for cannelure in tech['types_cannelure']:
                                    alt_context += f"- Type : {cannelure.get('type', 'N/A')}\n"
                                    alt_context += f"  Force : {cannelure.get('force', 'N/A')}\n"
                            if 'avantages' in tech:
                                alt_context += "Avantages :\n"
                                for avantage in tech['avantages']:
                                    alt_context += f"- {avantage}\n"
                        
                        # Ajout des informations d'usage
                        if 'usage_and_applications' in desc_json:
                            usage = desc_json['usage_and_applications']
                            alt_context += "\nApplications :\n"
                            if 'secteurs_concernes' in usage:
                                alt_context += "Secteurs concernés :\n"
                                for secteur in usage['secteurs_concernes']:
                                    alt_context += f"- {secteur}\n"
                            if 'applications' in usage:
                                alt_context += "Utilisations :\n"
                                for app in usage['applications']:
                                    alt_context += f"- {app}\n"
                        
                        # Ajout des dimensions disponibles
                        if 'variations' in desc_json:
                            alt_context += "\nDimensions disponibles :\n"
                            for var in desc_json['variations']:
                                if 'dimensions' in var:
                                    alt_context += f"- {var['dimensions']}\n"
                    
                    except Exception as e:
                        print(f"Erreur lors du parsing de la description : {str(e)}")
                        # Si le parsing échoue, on ajoute la description brute
                        alt_context += f"\nDescription détaillée :\n{alt['description']}\n"
                
                context_parts.append(alt_context)
                
                print(f"\nAlternative: {alt['name']}")
                print(f"Prix d'achat: {alt['prix_achat']}€")
                print(f"Prix de vente conseillé: {alt['prix_vente_conseille']}€")
                print(f"Marge minimum: {alt['marge_minimum']}€")
                print(f"Prix proposé: {prix_propose if prix_propose is not None else 'Non spécifié'}€")
                print(f"Marge actuelle avec prix proposé: {alt_marge_actuelle}€")
                print(f"Marge suffisante: {'Oui' if alt_marge_suffisante else 'Non'}")
            
            # Ajout du tableau récapitulatif
            context_parts.append(tableau_alternatives)
        else:
            context_parts.append("\nAucune alternative disponible.")

    # Génération de la réponse finale
    context = "\n".join(context_parts)
    print("\n=== FULL PROMPT ===")
    print(context)
    print("\n=== END PROMPT ===\n")
    
    response = _llm.invoke(SYSTEM_PROMPT.format(context=context))
    return response.content