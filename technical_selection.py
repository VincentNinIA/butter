"""
Module de sélection technique par LLM
Responsabilité : Présenter les alternatives au LLM et récupérer sa sélection
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain_openai import ChatOpenAI
import json

logger = logging.getLogger(__name__)

class TechnicalSelector:
    """Classe pour la sélection technique d'alternatives par LLM"""
    
    def __init__(self, llm_client: Optional[ChatOpenAI] = None):
        """
        Initialise le sélecteur technique
        
        Args:
            llm_client: Client LLM optionnel. Si None, un client par défaut sera créé
        """
        self.llm_client = llm_client
        
    def _create_selection_prompt(self, 
                                produit_demande: Dict[str, Any], 
                                alternatives: List[Dict[str, Any]], 
                                criteres_prioritaires: List[str] = None) -> str:
        """
        Crée un prompt spécialisé pour la sélection technique
        
        Args:
            produit_demande: Informations du produit demandé originalement
            alternatives: Liste des alternatives disponibles
            criteres_prioritaires: Critères de sélection prioritaires
            
        Returns:
            str: Prompt formaté pour le LLM
        """
        if criteres_prioritaires is None:
            criteres_prioritaires = ["similarité technique", "stock disponible", "marge", "délai"]
        
        prompt = f"""Vous êtes un expert en sélection de produits techniques. Votre mission est d'analyser les alternatives disponibles et de choisir la meilleure option pour remplacer le produit demandé.

**PRODUIT DEMANDÉ ORIGINALEMENT :**
- Nom : {produit_demande.get('name', 'N/A')}
- Fiche technique complète : {produit_demande.get('description', 'N/A')}
- Stock disponible : {produit_demande.get('stock_disponible', 'N/A')}
- Marge actuelle : {produit_demande.get('marge_actuelle', 'N/A')}€
- Marge minimum requise : {produit_demande.get('marge_minimum', 'N/A')}€
- Délai de livraison : {produit_demande.get('delai', 'N/A')}

**PROBLÈME IDENTIFIÉ :**
"""
        
        # Identifier le problème principal
        if not produit_demande.get('marge_suffisante', True):
            prompt += "- Marge insuffisante\n"
        if produit_demande.get('stock_disponible', 0) < produit_demande.get('quantite_demandee', 0):
            prompt += "- Stock insuffisant\n"
        
        prompt += f"""
**ALTERNATIVES DISPONIBLES :**
"""
        
        for i, alt in enumerate(alternatives, 1):
            # Formatage sécurisé de la similarité technique
            similarite_tech = alt.get('similarite_technique')
            if similarite_tech is not None:
                similarite_display = f"{similarite_tech:.0%}"
            else:
                similarite_display = "À analyser par le LLM"
            
            prompt += f"""
{i}. **{alt.get('name', 'Produit inconnu')}**
   • Similarité technique : {similarite_display}
   • Stock disponible : {alt.get('stock_disponible', 'N/A')}
   • Marge : {alt.get('marge', 'N/A')}€ (minimum : {alt.get('marge_minimum', 'N/A')}€)
   • Délai de livraison : {alt.get('delai', 'N/A')}
   • Prix d'achat : {alt.get('prix_achat', 'N/A')}€
   • Prix de vente conseillé : {alt.get('prix_vente_conseille', 'N/A')}€
   • Fiche technique complète : {alt.get('description', 'N/A')}
"""

        prompt += f"""
**CRITÈRES DE SÉLECTION (par ordre de priorité) :**
{chr(10).join(f"{i+1}. {critere}" for i, critere in enumerate(criteres_prioritaires))}

**RÈGLES CRITIQUES À RESPECTER :**
- COMPATIBILITÉ TECHNIQUE OBLIGATOIRE : Les alternatives doivent être de la MÊME FAMILLE de produits
- PRODUITS INCOMPATIBLES : Film étirable ≠ Carton ≠ Mousse ≠ Adhésif ≠ Sangles, etc.
- ACCEPTABLES : Film 17µm → Film 20µm (même famille, épaisseur différente)
- INACCEPTABLES : Film → Etui carton (familles complètement différentes)
- Si AUCUNE alternative n'est de la même famille technique, REJETER TOUTES

**INSTRUCTIONS :**
1. Analysez chaque alternative selon les critères de sélection
2. ÉLIMINEZ d'abord toutes les alternatives avec similarité technique < 30%
3. Parmi les alternatives restantes, choisissez celle avec le meilleur équilibre
4. Si aucune alternative n'est techniquement compatible, REFUSEZ la substitution

**FORMAT DE RÉPONSE OBLIGATOIRE :**
Répondez UNIQUEMENT au format JSON suivant :
{{
    "choix_optimal": {{
        "nom": "Nom exact du produit choisi ou null si aucun",
        "raison": "Explication détaillée du choix",
        "score_confiance": 0.8
    }},
    "analyse_comparative": [
        {{
            "nom": "Nom du produit",
            "avantages": ["liste des avantages"],
            "inconvenients": ["liste des inconvénients"],
            "score_adequation": 0.7
        }}
    ],
    "recommandation_finale": "Accepter/Rejeter avec explication"
}}

Analysez maintenant et donnez votre recommandation :"""
        
        return prompt
    
    def select_best_alternative(self, 
                              produit_demande: Dict[str, Any], 
                              alternatives: List[Dict[str, Any]],
                              criteres_prioritaires: List[str] = None) -> Dict[str, Any]:
        """
        Utilise le LLM pour sélectionner la meilleure alternative
        
        Args:
            produit_demande: Informations du produit demandé
            alternatives: Liste des alternatives disponibles
            criteres_prioritaires: Critères de sélection prioritaires
            
        Returns:
            Dict contenant la sélection du LLM et l'analyse
        """
        if not alternatives:
            logger.warning("Aucune alternative fournie pour la sélection LLM")
            return {
                "choix_optimal": None,
                "analyse_comparative": [],
                "recommandation_finale": "Aucune alternative disponible",
                "erreur": "Aucune alternative fournie"
            }
        
        if not self.llm_client:
            logger.warning("Aucun client LLM disponible pour la sélection")
            return self._fallback_selection(produit_demande, alternatives)
        
        try:
            # Créer le prompt de sélection
            prompt = self._create_selection_prompt(produit_demande, alternatives, criteres_prioritaires)
            
            logger.info(f"Envoi du prompt de sélection au LLM pour {len(alternatives)} alternatives")
            
            # Appel au LLM
            response = self.llm_client.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            logger.debug(f"Réponse brute du LLM : {response_text[:200]}...")
            
            # Parser la réponse JSON
            try:
                # Extraire le JSON de la réponse
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start == -1 or json_end == 0:
                    raise ValueError("Aucun JSON trouvé dans la réponse")
                
                json_str = response_text[json_start:json_end]
                selection_result = json.loads(json_str)
                
                logger.info(f"Sélection LLM réussie : {selection_result.get('choix_optimal', {}).get('nom', 'Aucun choix')}")
                
                return selection_result
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Erreur de parsing JSON : {str(e)}")
                logger.debug(f"Réponse non parsable : {response_text}")
                return self._parse_fallback_response(response_text, alternatives)
                
        except Exception as e:
            logger.error(f"Erreur lors de la sélection LLM : {str(e)}")
            return self._fallback_selection(produit_demande, alternatives)
    
    def _fallback_selection(self, 
                          produit_demande: Dict[str, Any], 
                          alternatives: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Sélection de fallback basée sur des règles simples
        
        Args:
            produit_demande: Produit demandé
            alternatives: Alternatives disponibles
            
        Returns:
            Dict: Résultat de sélection fallback
        """
        logger.info("Utilisation de la sélection fallback (règles simples)")
        
        if not alternatives:
            return {
                "choix_optimal": None,
                "analyse_comparative": [],
                "recommandation_finale": "Aucune alternative disponible",
                "methode": "fallback"
            }
        
        # Filtrage basique par famille de produits (noms similaires)
        produit_demande_nom = produit_demande.get('name', '').lower()
        alternatives_compatibles = []
        
        for alt in alternatives:
            alt_nom = alt.get('name', '').lower()
            # Vérification basique de famille de produits
            mots_produit = set(produit_demande_nom.split())
            mots_alt = set(alt_nom.split())
            mots_techniques = {'film', 'carton', 'mousse', 'etui', 'adhesif', 'sangle', 'palette', 'boite'}
            
            # Cherche si les mots techniques correspondent
            tech_produit = mots_produit.intersection(mots_techniques)
            tech_alt = mots_alt.intersection(mots_techniques)
            
            if tech_produit and tech_alt and tech_produit.intersection(tech_alt):
                alternatives_compatibles.append(alt)
            elif not tech_produit and not tech_alt:
                # Si aucun mot technique spécifique, garder pour analyse
                alternatives_compatibles.append(alt)
        
        if not alternatives_compatibles:
            logger.warning("Aucune alternative de la même famille technique trouvée")
            return {
                "choix_optimal": None,
                "analyse_comparative": [],
                "recommandation_finale": "Aucune alternative de la même famille technique disponible",
                "methode": "fallback",
                "raison_rejet": "incompatibilité_technique"
            }
        
        logger.info(f"Filtrage par famille technique: {len(alternatives)} → {len(alternatives_compatibles)} alternatives compatibles")
        
        # Tri par marge puis par stock
        alternatives_sorted = sorted(
            alternatives_compatibles, 
            key=lambda x: (
                x.get('marge', 0) - x.get('marge_minimum', 0),
                x.get('stock_disponible', 0)
            ), 
            reverse=True
        )
        
        best_alternative = alternatives_sorted[0]
        
        return {
            "choix_optimal": {
                "nom": best_alternative.get('name'),
                "raison": f"Sélection automatique basée sur la famille technique et la marge disponible",
                "score_confiance": 0.6
            },
            "analyse_comparative": [
                {
                    "nom": alt.get('name'),
                    "avantages": [f"Même famille technique", f"Marge: {alt.get('marge', 0) - alt.get('marge_minimum', 0):.2f}€"],
                    "inconvenients": [],
                    "score_adequation": 0.7
                } for alt in alternatives_sorted[:3]
            ],
            "recommandation_finale": f"Recommandation automatique : {best_alternative.get('name')} (même famille technique)",
            "methode": "fallback"
        }
    
    def _parse_fallback_response(self, response_text: str, alternatives: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parse une réponse LLM non-JSON en utilisant des heuristiques
        
        Args:
            response_text: Texte de réponse du LLM
            alternatives: Alternatives disponibles
            
        Returns:
            Dict: Résultat parsé
        """
        logger.info("Tentative de parsing fallback de la réponse LLM")
        
        # Chercher des noms de produits dans la réponse
        choix_trouve = None
        for alt in alternatives:
            nom_produit = alt.get('name', '')
            if nom_produit.lower() in response_text.lower():
                choix_trouve = nom_produit
                break
        
        return {
            "choix_optimal": {
                "nom": choix_trouve,
                "raison": "Extraction du choix à partir de la réponse textuelle du LLM",
                "score_confiance": 0.4
            },
            "analyse_comparative": [],
            "recommandation_finale": f"Choix extrait : {choix_trouve}" if choix_trouve else "Aucun choix identifiable",
            "methode": "parsing_fallback",
            "reponse_originale": response_text[:500]
        }

# Fonction utilitaire pour faciliter l'utilisation
def select_technical_alternative(produit_demande: Dict[str, Any], 
                                alternatives: List[Dict[str, Any]], 
                                llm_client: Optional[ChatOpenAI] = None,
                                criteres_prioritaires: List[str] = None) -> Dict[str, Any]:
    """
    Fonction utilitaire pour la sélection technique d'alternatives
    
    Args:
        produit_demande: Produit demandé originalement
        alternatives: Liste des alternatives disponibles
        llm_client: Client LLM optionnel
        criteres_prioritaires: Critères de sélection prioritaires
        
    Returns:
        Dict: Résultat de la sélection
    """
    selector = TechnicalSelector(llm_client)
    return selector.select_best_alternative(produit_demande, alternatives, criteres_prioritaires) 