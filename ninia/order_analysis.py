import logging
from rag.retrieval import fetch_docs, enrich_alternatives_for_llm
from .extraction_llm import extract_order_info_with_llm_fallback
from .technical_selection import TechnicalSelector, select_technical_alternative
import unidecode

logger = logging.getLogger(__name__)

def normalize_name(name: str) -> str:
    """Normalise un nom de produit pour la recherche."""
    if not isinstance(name, str):
        name = str(name)
    return unidecode.unidecode(name.lower().strip())

def analyser_commande(user_query_for_order: str, inventory_df, llm_client=None):
    """
    Analyse une commande complète en vérifiant le stock, la marge et les alternatives.
    Args:
        user_query_for_order (str): La requête brute de l'utilisateur pour la commande.
        inventory_df: DataFrame d'inventaire (obligatoire)
        llm_client: Client LLM pour l'extraction (optionnel, utilise regex si None)
    Returns:
        Dict[str, Any]: Résultat de l'analyse avec recommandations.
    """
    logger.info(f"DEBUG - analyser_commande appelé avec: {user_query_for_order}")
    
    # S'assurer que l'inventaire a la colonne nom_normalise
    if 'nom_normalise' not in inventory_df.columns:
        inventory_df['nom_normalise'] = inventory_df['nom'].apply(normalize_name)
    
    # Extraction des informations avec LLM si disponible, sinon fallback regex
    if llm_client:
        logger.info("DEBUG - Utilisation de l'extraction LLM")
        product_id, quantite, prix_propose = extract_order_info_with_llm_fallback(
            user_query_for_order, llm_client, inventory_df
        )
        
        # Si l'extraction LLM échoue complètement, utiliser le fallback regex
        if product_id is None and quantite is None and prix_propose is None:
            logger.warning("DEBUG - Extraction LLM échouée, utilisation du fallback regex")
            from .extraction import _extract_product_and_quantity_from_string
            product_id, quantite, prix_propose = _extract_product_and_quantity_from_string(
                user_query_for_order, inventory_df
            )
    else:
        logger.info("DEBUG - Utilisation de l'extraction regex (fallback)")
        from .extraction import _extract_product_and_quantity_from_string
        product_id, quantite, prix_propose = _extract_product_and_quantity_from_string(
            user_query_for_order, inventory_df
        )
    
    logger.info(f"DEBUG - Extraction: produit={product_id}, quantité={quantite}, prix_propose={prix_propose}")
    
    if not product_id:
        return {
            "status": "ERROR",
            "message": "Impossible d'identifier le produit demandé.",
            "produit": None,
            "analyse": {
                "stock_suffisant": False,
                "marge_suffisante": False,
                "delai_compatible": False
            }
        }
    
    # Appel à fetch_docs avec le prix proposé
    logger.info(f"DEBUG - fetch_docs appelé avec prix_propose={prix_propose}")
    result = fetch_docs(
        query=f"Commande de {quantite} {product_id}" + (f" à {prix_propose} euros" if prix_propose is not None else ""),
        product_id=product_id,
        required_qty=quantite,
        prix_propose=prix_propose
    )
    
    if not result or not result.get("produit"):
        return {
            "status": "ERROR",
            "message": f"Produit '{product_id}' non trouvé dans l'inventaire.",
            "produit": None,
            "analyse": {
                "stock_suffisant": False,
                "marge_suffisante": False,
                "delai_compatible": False
            }
        }
    
    produit_info = result["produit"]
    
    # Récupération des données de marge calculées par fetch_docs
    marge_actuelle = produit_info.get('marge_actuelle', 0.0)
    marge_suffisante = produit_info.get('marge_suffisante', True)
    prix_propose_retenu = produit_info.get('prix_propose_retenu')
    stock_disponible = produit_info.get('stock_disponible', 0)
    
    logger.info(f"DEBUG - Données de marge de fetch_docs: marge_actuelle={marge_actuelle}, marge_suffisante={marge_suffisante}")
    logger.info(f"DEBUG - Stock disponible: {stock_disponible}, Quantité demandée: {quantite}")
    
    # Analyse des résultats
    stock_suffisant = stock_disponible >= quantite
    
    # Construction de l'analyse détaillée
    analyse = {
        "quantite_demandee": quantite,
        "stock_disponible": stock_disponible,
        "stock_suffisant": stock_suffisant,
        "marge_actuelle": marge_actuelle,
        "marge_suffisante": marge_suffisante,
        "prix_propose_retenu": prix_propose_retenu,
        "delai_compatible": True,  # Par défaut, à affiner si nécessaire
        "delai_demande": "standard",  # À affiner si nécessaire
        "delai_dispo": produit_info.get('delai', 'standard')
    }
    
    logger.info(f"DEBUG - Analyse finale: stock_suffisant={stock_suffisant}, marge_suffisante={marge_suffisante}")
    
    # Gestion des alternatives avec sélection LLM
    alternatives = result.get("alternatives", [])
    selection_llm = None
    produit_final = produit_info
    analyse_finale = analyse
    
    # Si il y a des problèmes ET des alternatives disponibles → Utiliser la sélection LLM
    if (not stock_suffisant or not marge_suffisante) and alternatives:
        logger.info(f"DEBUG - Problème détecté avec {len(alternatives)} alternatives disponibles, lancement sélection LLM")
        
        try:
            # Enrichissement des alternatives pour l'analyse LLM
            alternatives_enrichies = enrich_alternatives_for_llm(
                alternatives, produit_info, prix_propose
            )
            
            # Sélection LLM (avec fallback automatique si LLM indisponible)
            selection_llm = select_technical_alternative(
                produit_demande=produit_info,
                alternatives=alternatives_enrichies,
                llm_client=llm_client,  # Utilisera le fallback si None
                criteres_prioritaires=["stock disponible", "marge", "similarité technique", "délai"]
            )
            
            # S'assurer que selection_llm est toujours un dictionnaire
            if not isinstance(selection_llm, dict):
                logger.warning(f"DEBUG - select_technical_alternative a retourné un type inattendu: {type(selection_llm)}, valeur: {selection_llm}")
                selection_llm = {
                    "choix_optimal": None,
                    "analyse_comparative": [],
                    "recommandation_finale": "Erreur inattendue dans la sélection LLM",
                    "erreur": f"Type retourné inattendu: {type(selection_llm)}"
                }
            
            if selection_llm:
                logger.info(f"DEBUG - Sélection LLM terminée: {selection_llm.get('choix_optimal', {}).get('nom') if selection_llm.get('choix_optimal') else 'Aucun choix'}")
            else:
                logger.info("DEBUG - Sélection LLM a retourné None")
            
            # Si le LLM a choisi une alternative valide
            choix_optimal = selection_llm.get('choix_optimal') if selection_llm else None
            if choix_optimal and isinstance(choix_optimal, dict) and choix_optimal.get('nom'):
                nom_choisi = choix_optimal['nom']
                
                # Trouver l'alternative correspondante
                alternative_choisie = None
                for alt in alternatives_enrichies:
                    if alt.get('name') == nom_choisi:
                        alternative_choisie = alt
                        break
                
                if alternative_choisie:
                    logger.info(f"DEBUG - Alternative choisie par LLM: {nom_choisi}")
                    
                    # Vérifier que l'alternative résout les problèmes
                    alt_stock_ok = alternative_choisie.get('stock_disponible', 0) >= quantite
                    alt_marge_ok = alternative_choisie.get('marge', 0) >= alternative_choisie.get('marge_minimum', 0)
                    
                    if alt_stock_ok and alt_marge_ok:
                        # Utiliser l'alternative comme produit final
                        produit_final = alternative_choisie
                        
                        # Recalculer l'analyse avec l'alternative
                        analyse_finale = {
                            "quantite_demandee": quantite,
                            "stock_disponible": alternative_choisie.get('stock_disponible', 0),
                            "stock_suffisant": alt_stock_ok,
                            "marge_actuelle": alternative_choisie.get('marge', 0),
                            "marge_suffisante": alt_marge_ok,
                            "prix_propose_retenu": prix_propose,
                            "delai_compatible": True,
                            "delai_demande": "standard",
                            "delai_dispo": alternative_choisie.get('delai', 'standard'),
                            "produit_substitue": True,
                            "produit_original": product_id,
                            "raison_substitution": choix_optimal.get('raison', 'Sélection automatique par LLM')
                        }
                        
                        stock_suffisant = alt_stock_ok
                        marge_suffisante = alt_marge_ok
                        
                        logger.info(f"DEBUG - Alternative validée: stock_ok={alt_stock_ok}, marge_ok={alt_marge_ok}")
                    else:
                        logger.warning(f"DEBUG - Alternative choisie ne résout pas les problèmes: stock_ok={alt_stock_ok}, marge_ok={alt_marge_ok}")
                else:
                    logger.warning(f"DEBUG - Alternative choisie '{nom_choisi}' non trouvée dans la liste")
            else:
                logger.info("DEBUG - LLM n'a pas choisi d'alternative valide")
                
        except Exception as e:
            logger.error(f"DEBUG - Erreur lors de la sélection LLM: {str(e)}")
            # En cas d'erreur, continuer avec la logique normale
    
    # Détermination du statut global final - NOUVELLE LOGIQUE
    # On utilise toujours le produit ORIGINAL dans l'analyse, pas l'alternative
    produit_final = produit_info  # Toujours le produit demandé
    analyse_finale = analyse      # Toujours l'analyse du produit demandé
    
    # Logique de statut : on signale TOUS les problèmes
    if not stock_suffisant and not marge_suffisante:
        status = "REFUSED"
        message = f"❌ {product_id} : REFUSÉ - Stock insuffisant ({stock_disponible}/{quantite}) ET Marge insuffisante ({marge_actuelle}€ < {produit_info.get('marge_minimum', 'N/A')}€)"
    elif not stock_suffisant:
        status = "ATTENTION"
        message = f"⚠️ {product_id} : STOCK INSUFFISANT - Disponible: {stock_disponible}, demandé: {quantite}"
    elif not marge_suffisante:
        status = "REFUSED"
        message = f"❌ {product_id} : REFUSÉ - Marge insuffisante (actuelle: {marge_actuelle}€, minimum requise: {produit_info.get('marge_minimum', 'N/A')}€)"
    else:
        status = "OK"
        message = f"✅ {product_id} : OK - Quantité: {quantite}, Prix: {prix_propose_retenu}€"
    
    return {
        "status": status,
        "message": message,
        "produit": produit_final,
        "produit_original": produit_info if analyse_finale.get('produit_substitue') else None,
        "analyse": analyse_finale,
        "alternatives": alternatives,
        "selection_llm": selection_llm,
        "substitution_effectuee": analyse_finale.get('produit_substitue', False)
    }

def analyse_commande(produit, quantite, prix=None):
    """
    Analyse la commande (stock, marge, etc.)
    Retourne une chaîne ou un dict avec l'analyse.
    """
    # À implémenter
    return f"Analyse de la commande : {quantite} x {produit} (prix: {prix})" 