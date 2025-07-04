from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import re
import unidecode
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel, Field
from rag.core import answer
from rag.retrieval import get_stock, fetch_docs, _inventory_df
import pandas as pd

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Définition du modèle Pydantic pour les arguments de analyser_commande
class AnalyserCommandeArgs(BaseModel):
    product_id: str = Field(description="Identifiant ou nom EXACT du produit à commander (ne pas modifier l'entrée utilisateur)")
    quantite: int = Field(description="La QUANTITÉ du produit à commander. Ce champ est OBLIGATOIRE.")
    prix_propose: Optional[float] = Field(default=None, description="Prix proposé par le client (optionnel)")
    delai: Optional[datetime] = Field(default=None, description="Délai de livraison souhaité (optionnel)")

# Fonction d'assistance pour normaliser et extraire produit/quantité
def _normalize_text_for_extraction(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return unidecode.unidecode(text.lower().strip())

def _strip_command_keywords(text: str, keywords: List[str]) -> str:
    stripped_text = text
    for keyword in keywords:
        # S'assurer que le mot-clé est au début et suivi d'un espace ou est toute la chaîne
        if stripped_text.startswith(keyword + " "):
            stripped_text = stripped_text[len(keyword)+1:].lstrip()
        elif stripped_text == keyword: # Au cas où le produit serait juste le mot-clé (peu probable)
            stripped_text = ""
            break
    return stripped_text

def _extract_product_and_quantity_from_string(text: str, inventory_df: pd.DataFrame) -> tuple[Optional[str], Optional[int], Optional[float]]:
    normalized_text = _normalize_text_for_extraction(text)
    
    quantity = None
    product_text_part = normalized_text
    prix_propose = None

    # Regex pour extraire le prix (ex: "pour 15 euros", "a 12.50€", "prix: 10")
    # (?:...) est un groupe non-capturant
    # (\\d+(?:[.,]\\d{1,2})?) capture un nombre entier ou décimal (avec . ou ,)
    # (?:\\s*(?:euros?|eur|€))? capture optionnellement "euro(s)", "eur" ou "€"
    price_patterns = [
        r"(?:a|pour|de)\s*(\d+(?:[.,]\d{1,2})?)\s*(?:euros?|eur|€)",
        r"(\d+(?:[.,]\d{1,2})?)\s*(?:euros?|eur|€)", # Si le prix est juste avant "euros"
        r"prix(?:[\s]*est[\s]*de|[\s]*de|[\s]*:\s*|[\s]+)?(\d+(?:[.,]\d{1,2})?)"
    ]

    original_text_for_price_search = normalized_text # Garder une copie pour la recherche de prix

    for pattern in price_patterns:
        match_price = re.search(pattern, normalized_text)
        if match_price:
            try:
                prix_str = match_price.group(1).replace(',', '.') # Remplace virgule par point pour float()
                prix_propose = float(prix_str)
                logger.info(f"_extract: Prix proposé trouvé: {prix_propose}€ à partir de '{match_price.group(0)}'")
                # Enlever l'expression du prix du texte pour ne pas interférer avec l'extraction produit/quantité
                # On modifie normalized_text ici, ce qui affectera l'extraction de produit/quantité.
                # product_text_part sera mis à jour plus tard si une quantité est trouvée.
                normalized_text = normalized_text.replace(match_price.group(0), "", 1).strip()
                product_text_part = normalized_text # Mettre à jour product_text_part également
                break 
            except ValueError:
                logger.warning(f"_extract: Impossible de convertir le prix trouvé '{match_price.group(1)}' en float.")

    # Listes d'unités et mots-clés de commande
    command_units = [
        "unite", "unites", "rouleau", "rouleaux", "exemplaire", "exemplaires",
        "piece", "pieces", "caisse", "caisses", "boite", "boites", "palette", "palettes",
        "sac", "sacs", "carton", "cartons"
    ]
    technical_units = ["um", "microns", "mm", "cm", "m", "kg", "g", "l", "ml"]
    # Mots-clés utilisés aussi pour la précédence, mais aussi pour le nettoyage potentiel
    command_keywords_for_stripping = sorted(["commande de", "commande", "acheter", "prendre", "veux", "voudrais", "besoin de", "pour", "donne moi"], key=len, reverse=True)

    # Regex pour trouver des nombres suivis optionnellement d'une unité
    # (nombre) (optionnel: espace) (optionnel: mot)
    # On va chercher tous les nombres et analyser leur contexte.
    
    potential_quantities = [] # Store tuples of (quantity, product_string_if_this_is_qty, original_match_text)

    # 1. Chercher "nombre + unité_de_commande"
    # Pattern: (\d+)\s+(word_unit_de_commande)
    for unit in command_units:
        # Regex: (nombre) espace (unité de commande)\b pour éviter les sur-correspondances (ex: "unites" ne doit pas matcher "uniteSUPER")
        # On cherche la quantité AVANT ou APRES le nom du produit
        # Exemple: "10 pieces de produit X" ou "produit X, 10 pieces"
        # Pattern: (\d+)\s*UNIT\b  ou   UNIT\s*(\d+)  (pas géré pour l'instant, on se concentre sur NOMBRE UNIT)
        
        # Chercher "(\d+)\s+UNIT" (ex: "30 unites")
        # On doit extraire le nombre, l'unité et ce qui reste pour le produit
        # et on veut ce qui est avant le nombre et après l'unité comme produit.
        
        # On itère sur les matchs pour "nombre unité"
        # Ex: "Je veux 10 caisses de produit A et 5 boites de produit B" -> on veut pouvoir les séparer
        # Pour l'instant, on se simplifie la tâche et on prend le dernier trouvé s'il y en a plusieurs.

        # Chercher "(\d+)\s*(unit_keyword)\b"
        # Le \b est important pour ne pas que "piece" matche "pieces" et cause des problèmes de découpage
        pattern_qty_unit = rf"(\d+)\s*({unit})\b"
        
        for match in re.finditer(pattern_qty_unit, normalized_text):
            qty_candidate = int(match.group(1))
            matched_text = match.group(0) # ex: "30 unites"
            
            # Le reste est le produit: tout ce qui n'est pas le "nombre unité"
            # On supprime la première occurrence de ce match pour obtenir le produit.
            # C'est une heuristique, si le même "nombre unité" apparaît plusieurs fois, ça peut être problématique.
            temp_product_text = normalized_text.replace(matched_text, "", 1).strip()
            # Nettoyer les virgules ou connecteurs restants
            temp_product_text = re.sub(r"^\s*[,_\-\s]*", "", temp_product_text)
            temp_product_text = re.sub(r"[,_\-\s]*\s*$", "", temp_product_text)
            potential_quantities.append({'q': qty_candidate, 'p': temp_product_text, 'match': matched_text, 'type': 'qty_unit'})
            logger.info(f"_extract: Candidat (qty_unit) trouvé: Q={qty_candidate}, P='{temp_product_text}' à partir de '{matched_text}'")


    # 2. Si pas de "nombre + unité_de_commande" clair, chercher des nombres qui ne sont PAS suivis d'une unité technique.
    if not potential_quantities:
        # Regex pour capturer un nombre (\d+) et ce qui le suit (pour vérifier l'unité technique)
        # On cherche un nombre, puis on regarde ce qui le suit.
        # On privilégie les nombres qui sont en début de chaîne ou après des mots-clés de commande.
        
        # Cette regex est similaire à l'ancienne, mais on va filtrer après.
        # "(?:commande de|...)?\s*(\d+)\s*(.*)"
        # Le (.*) est trop gourmand.
        
        # On cherche tous les nombres dans la chaîne
        all_numbers_matches = list(re.finditer(r"(\d+)", normalized_text))
        
        for i, match_num in enumerate(all_numbers_matches):
            num_val = int(match_num.group(1))
            start_idx, end_idx = match_num.span()

            # Vérifier si ce nombre est suivi d'une unité technique
            is_technical = False
            # Regarder le mot juste après le nombre
            # Extraire une courte sous-chaîne après le nombre pour chercher des unités techniques
            # ex: "15 um" -> "um" ; "20kg" -> "kg"
            substring_after_num = normalized_text[end_idx:end_idx+10].strip().lower() # +10 pour avoir assez de contexte
            for tech_unit in technical_units:
                if substring_after_num.startswith(tech_unit):
                    is_technical = True
                    logger.info(f"_extract: Nombre {num_val} identifié comme technique (suivi de '{tech_unit}')")
                    break
            
            if not is_technical:
                # Ce nombre n'est pas suivi d'une unité technique. C'est un candidat.
                # Comment déterminer le texte du produit ?
                # Si le nombre est au début: "10 produit X" -> Q=10, P="produit X"
                # Si le nombre est à la fin: "produit X 10" -> Q=10, P="produit X" (plus rare pour commandes)
                # Si au milieu: "produit X 10 pour ..." (plus complexe)

                # Heuristique: si on a un nombre non technique, on essaie de le séparer du reste.
                # On prend le dernier nombre non technique comme quantité (heuristique courante)
                # ou le premier. Testons avec le dernier.
                
                # Si on prend ce nombre, le reste est le produit.
                # On enlève ce nombre de la chaîne.
                # Il faut faire attention à ne pas enlever une partie d'un autre nombre.
                # On utilise les indices start_idx et end_idx pour reconstruire
                temp_product_text = (normalized_text[:start_idx] + normalized_text[end_idx:]).strip()
                temp_product_text = re.sub(r"^\s*[,_\-\s]*", "", temp_product_text)
                temp_product_text = re.sub(r"[,_\-\s]*\s*$", "", temp_product_text)
                
                # On peut aussi essayer de voir si le nombre est précédé de mots-clés de commande
                command_keywords = ["commande de", "commande", "acheter", "prendre", "veux", "voudrais", "besoin de", "pour"]
                preceding_text = normalized_text[:start_idx].strip()
                is_preceded_by_keyword = any(preceding_text.endswith(keyword) for keyword in command_keywords)

                potential_quantities.append({
                    'q': num_val, 
                    'p': temp_product_text, 
                    'match': match_num.group(1), 
                    'type': 'isolated_num',
                    'is_preceded': is_preceded_by_keyword,
                    'pos': start_idx # position du nombre
                })
                logger.info(f"_extract: Candidat (isolated_num) trouvé: Q={num_val}, P='{temp_product_text}', Précédé par mot-clé: {is_preceded_by_keyword}")

    # Stratégie de sélection parmi les candidats :
    if potential_quantities:
        # 1. Priorité aux matchs "nombre + unité_de_commande"
        qty_unit_matches = [p for p in potential_quantities if p['type'] == 'qty_unit']
        if qty_unit_matches:
            chosen_candidate = qty_unit_matches[-1] 
            quantity = chosen_candidate['q']
            product_text_part = _normalize_text_for_extraction(chosen_candidate['p'])
            logger.info(f"_extract: Choix initial (qty_unit): Q={quantity}, P='{product_text_part}'")
        else:
            # 2. Sinon, prendre parmi les "isolated_num"
            isolated_keyword_matches = [p for p in potential_quantities if p['type'] == 'isolated_num' and p['is_preceded']]
            if isolated_keyword_matches:
                chosen_candidate = isolated_keyword_matches[-1]
            else:
                 if potential_quantities: 
                    chosen_candidate = potential_quantities[-1] 
                 else: 
                    chosen_candidate = None
            
            if chosen_candidate:
                quantity = chosen_candidate['q']
                product_text_part = _normalize_text_for_extraction(chosen_candidate['p'])
                logger.info(f"_extract: Choix initial (isolated_num): Q={quantity}, P='{product_text_part}'")
            else: 
                quantity = None
                product_text_part = normalized_text 
                logger.info(f"_extract: Aucun candidat isolé valable après filtrage. Q=None, P='{product_text_part}'")
                
    else: 
        quantity = None
        product_text_part = normalized_text
        logger.info(f"_extract: Aucun nombre trouvé ou tous techniques (pré-nettoyage). Q=None, P='{product_text_part}'")

    # Nettoyage final du product_text_part des mots-clés de commande au début
    # Ce nettoyage est appliqué à product_text_part qu'il vienne d'un match de quantité ou du texte normalisé original.
    if product_text_part:
        product_text_part = _strip_command_keywords(product_text_part, command_keywords_for_stripping)
        logger.info(f"_extract: Product text part APRÈS nettoyage des mots-clés: '{product_text_part}'")

    # Si product_text_part est vide et qu'on a une quantité, c'est une erreur
    if quantity is not None and not product_text_part.strip():
        logger.warning(f"_extract: Quantité {quantity} trouvée mais nom de produit vide. Retour à texte original pour produit.")
        product_text_part = normalized_text # Sécurité: si le produit est vide, on repart du texte complet
        # On pourrait aussi invalider la quantité ici, mais ça dépend de la robustesse voulue.

    # Recherche du produit dans l'inventaire (logique existante)
    # S'assurer que la colonne nom_normalise existe
    if 'nom_normalise' not in inventory_df.columns:
        if 'nom' in inventory_df.columns:
            logger.warning("La colonne 'nom_normalise' n'était pas dans l'inventaire pour l'extraction, création à la volée.")
            inventory_df['nom_normalise'] = inventory_df['nom'].apply(_normalize_text_for_extraction)
        else:
            logger.error("Colonne 'nom' et 'nom_normalise' manquantes dans l'inventaire pour l'extraction.")
            return None, quantity, prix_propose

    best_match_product_name = None
    if product_text_part: # S'assurer qu'on a un texte pour chercher le produit
        exact_match = inventory_df[inventory_df['nom_normalise'] == product_text_part]
        if not exact_match.empty:
            best_match_product_name = exact_match['nom'].iloc[0]
        else:
            # Correspondance partielle améliorée: on cherche le nom de produit normalisé
            # qui a le plus long recouvrement avec product_text_part
            best_overlap = 0
            
            # Tokenize product_text_part pour une meilleure recherche partielle
            # (simple split pour l'instant)
            text_part_tokens = set(product_text_part.split())

            for index, row in inventory_df.iterrows():
                prod_norm = row['nom_normalise']
                original_name = row['nom']
                
                # Option 1: Le nom normalisé du produit est DANS le texte extrait
                if prod_norm in product_text_part:
                    if len(prod_norm) > best_overlap: # On veut le match le plus long
                        best_overlap = len(prod_norm)
                        best_match_product_name = original_name
                
                # Option 2: Le texte extrait est contenu DANS le nom normalisé du produit
                # (moins probable si on a bien extrait la quantité)
                elif product_text_part in prod_norm:
                     if len(product_text_part) > best_overlap:
                        best_overlap = len(product_text_part)
                        best_match_product_name = original_name
                
                # Option 3: Chevauchement de tokens (plus robuste pour des variations)
                else:
                    prod_tokens = set(prod_norm.split())
                    common_tokens = text_part_tokens.intersection(prod_tokens)
                    # Score simple : nombre de tokens communs / nombre total de tokens dans le nom du produit (pour normaliser)
                    # ou juste nombre de tokens communs si on veut privilégier les produits avec plus de mots en commun
                    if prod_tokens: # éviter division par zéro
                        # overlap_score = len(common_tokens) / len(prod_tokens)
                        overlap_score = len(common_tokens) # Simpler: plus de mots communs = mieux
                        if overlap_score > best_overlap:
                             best_overlap = overlap_score
                             best_match_product_name = original_name
            
            if best_match_product_name:
                 logger.info(f"_extract: Final (match partiel chevauchant) - Produit: '{best_match_product_name}', Score: {best_overlap}")
            else:
                logger.info(f"_extract: Aucun match de produit suffisamment bon trouvé pour '{product_text_part}'")
                # Si aucun bon match, on pourrait envisager de ne pas retourner de nom de produit
                # best_match_product_name = None # ou laisser tel quel pour que le LLM gère.
    
    if best_match_product_name is None and product_text_part:
        # Si après toutes les tentatives, on n'a pas de nom de produit précis mais on avait un texte,
        # on peut retourner le texte nettoyé pour que le LLM tente une recherche RAG plus large.
        # Mais attention, cela peut introduire du bruit si le texte est vague.
        # Pour l'instant, on préfère retourner None si aucun match structuré n'est fait.
        # Cependant, si un prix a été extrait, mais pas le produit, le produit est essentiel.
        # Si un produit a été extrait, mais pas la quantité, la quantité est essentielle.
        pass

    logger.info(f"_extract: Final - Produit: '{best_match_product_name}', Quantité: {quantity}, Prix proposé: {prix_propose}")
    return best_match_product_name, quantity, prix_propose

class NINIAAgent:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Une clé API est requise pour initialiser l'agent")
            
        self.llm = ChatOpenAI(
            model="gpt-4.1",
            temperature=0,
            api_key=api_key
        )
        self.inventory_df = _inventory_df
        
        self.tools = [
            Tool(
                name="verifier_stock",
                func=self.verifier_stock,
                description="Vérifie le stock disponible pour un produit donné. L'input est une chaîne (str) contenant le nom ou ID du produit."
            ),
            Tool(
                name="analyser_commande",
                func=self.analyser_commande,
                description=(
                    "Analyse une demande de commande. \
                    L'input DOIT être une chaîne de caractères (str) contenant la description complète de la commande, incluant le nom du produit et la quantité. \
                    Exemple d'input: 'Je veux commander 200 Film machine sans groupe Polytech 9 µm'. \
                    Ne passez que la phrase de commande de l'utilisateur."
                )
            ),
            Tool(
                name="recherche_documents",
                func=answer,
                description="Recherche des informations détaillées dans la base de connaissances. Argument: query (str)"
            )
        ]
        
        self.system_prompt = """Vous êtes NINIA, un assistant IA spécialisé dans l'analyse des commandes et la gestion d'inventaire.
        
        **Instructions importantes :**
        - Lorsque vous appelez un outil, fournissez l'input EXACTEMENT comme décrit par l'outil.
        - Pour `analyser_commande`, passez la phrase de l'utilisateur décrivant la commande (produit et quantité).
        - Pour `verifier_stock`, passez le nom du produit.
        
        Pour chaque demande, vous devez d'abord déterminer le type de requête :
        1. REQUÊTE D'INFORMATION : Pour les questions sur les stocks, caractéristiques, disponibilité. Appelez `verifier_stock` ou `recherche_documents`.
        2. COMMANDE : Pour les demandes d'achat avec un produit et une quantité. Appelez `analyser_commande` en lui passant la phrase de commande.
        
        Format de réponse selon le type de requête :
        
        Pour une REQUÊTE D'INFORMATION :
        📊 État du stock : [Disponible/En rupture]
        📦 Stock actuel : [X] unités
        ⏱️ Délai de réapprovisionnement : [X semaines]
        💡 Caractéristiques techniques : [Si demandées]
        🔄 Alternatives disponibles : [Si en rupture]
        
        Pour une COMMANDE :
        🎯 État : [OK/ATTENTION/ERREUR]
        📊 Analyse :
           • Quantité demandée : [X] unités
           • Stock disponible : [X] unités
           • Marge : [X]€/unité
           • Délai : [X semaines]
        ⚠️ Points d'attention : [Si applicable]
        🔄 Alternatives proposées : [Si nécessaire]
        
        Pour les commandes multiples, traitez chaque produit séparément avec ce format.
        
        Dans tous les cas :
        - Soyez concis et utilisez des émojis pour une meilleure lisibilité
        - Mettez en évidence les points importants
        - Proposez des solutions concrètes en cas de problème
        - Utilisez des listes à puces pour une meilleure structure."""
        
        try:
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            
            self.agent = create_openai_functions_agent(
                llm=self.llm,
                prompt=self.prompt,
                tools=self.tools
            )
            
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True
            )
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de l'agent : {str(e)}")
            raise
    
    def verifier_stock(self, product_name_or_id: str) -> Dict[str, Any]:
        """
        Vérifie le stock disponible pour un produit.
        
        Args:
            product_name_or_id (str): Nom ou ID du produit
            
        Returns:
            Dict[str, Any]: Informations sur le stock
            
        Raises:
            ValueError: Si product_name_or_id est vide ou invalide
        """
        if not product_name_or_id or not isinstance(product_name_or_id, str):
            raise ValueError("L'identifiant du produit doit être une chaîne non vide")
            
        try:
            logger.info(f"Vérification du stock pour le produit : {product_name_or_id}")
            stock = get_stock(product_name_or_id)
            
            if stock is None:
                logger.warning(f"Produit non trouvé : {product_name_or_id}")
                return {
                    "status": "ERROR",
                    "message": f"Produit {product_name_or_id} non trouvé",
                    "stock_disponible": 0
                }
                
            return {
                "product_id": product_name_or_id,
                "stock_disponible": stock,
                "status": "OK" if stock > 0 else "RUPTURE",
                "message": f"Stock disponible : {stock} unités" if stock > 0 else "Rupture de stock"
            }
        except Exception as e:
            logger.error(f"Erreur lors de la vérification du stock : {str(e)}")
            return {
                "status": "ERROR",
                "message": f"Erreur lors de la vérification du stock : {str(e)}",
                "stock_disponible": 0
            }
    
    def analyser_commande(self, 
                         user_query_for_order: str
                         ) -> Dict[str, Any]:
        """
        Analyse une commande complète en vérifiant le stock, la marge et les alternatives.
        
        Args:
            user_query_for_order (str): La requête brute de l'utilisateur pour la commande.
            
        Returns:
            Dict[str, Any]: Résultat de l'analyse avec recommandations.
            
        Raises:
            ValueError: Si les paramètres sont invalides (géré par Pydantic maintenant)
        """
        logger.info(f"analyser_commande appelée avec la chaîne: '{user_query_for_order}'")

        # Utiliser la fonction d'extraction améliorée
        product_id, quantite, prix_propose_extrait = _extract_product_and_quantity_from_string(user_query_for_order, self.inventory_df)

        # Le délai n'est pas extrait par cette fonction pour l'instant.
        delai = None # Peut être géré par une extraction plus avancée ou par le LLM.

        if not product_id:
            logger.warning(f"Impossible d'extraire product_id de '{user_query_for_order}'")
            return {"status": "ERROR", "message": f"Impossible d'identifier le produit dans votre demande: '{user_query_for_order}'. Veuillez reformuler."}
        
        if quantite is None or quantite <= 0:
            logger.warning(f"Impossible d'extraire une quantité valide de '{user_query_for_order}' (quantité trouvée: {quantite})")
            return {"status": "ERROR", "message": f"Impossible d'identifier la quantité dans votre demande: '{user_query_for_order}'. Veuillez préciser la quantité."}

        try:
            logger.info(f"Analyse de la commande (après extraction): {quantite} '{product_id}', Prix proposé: {prix_propose_extrait}")
            
            result = fetch_docs(
                query=f"Commande de {quantite} {product_id}" + (f" à {prix_propose_extrait} euros" if prix_propose_extrait is not None else ""),
                product_id=product_id,
                required_qty=quantite,
                prix_propose=prix_propose_extrait, # Utiliser le prix extrait ici
            )
            
            if not result["produit"]:
                logger.warning(f"Produit '{product_id}' non trouvé par fetch_docs après extraction.")
                return {
                    "status": "ERROR",
                    "message": f"Produit '{product_id}' non trouvé dans notre inventaire.",
                    "alternatives": []
                }
                
            produit_data = result["produit"]
            alternatives = result["alternatives"]
            
            # Analyse complète
            stock_suffisant = produit_data["stock_disponible"] >= quantite
            delai_compatible = True if delai is None else produit_data.get("delai_compatible", True)
            
            # Analyse de la marge
            marge_suffisante = True
            marge_actuelle = 0.0 # Initialiser à 0.0
            
            # La logique de calcul de marge utilise déjà prix_propose (qui est maintenant prix_propose_extrait via l'appel à fetch_docs)
            # mais on peut la rendre plus explicite ici si fetch_docs ne la calcule pas directement
            # ou si on veut la recalculer/vérifier au niveau de l'agent.
            # Pour l'instant, on se fie à ce que fetch_docs retourne dans produit["marge_minimum"], etc.
            # et la propre logique de l'agent ci-dessous.

            # Récupérer les données du produit retourné par fetch_docs
            prix_achat_produit = produit_data.get("prix_achat")
            marge_minimum_produit = produit_data.get("marge_minimum")

            if prix_propose_extrait is not None and prix_achat_produit is not None and marge_minimum_produit is not None:
                marge_actuelle = prix_propose_extrait - prix_achat_produit
                marge_suffisante = marge_actuelle >= marge_minimum_produit
                logger.info(f"Vérification de marge avec prix proposé ({prix_propose_extrait}€): Marge actuelle={marge_actuelle:.2f}€, Marge min={marge_minimum_produit}€, Suffisante={marge_suffisante}")
            elif prix_achat_produit is not None and marge_minimum_produit is not None and produit_data.get("prix_vente_conseille") is not None:
                # Si pas de prix proposé, on vérifie la marge sur le prix de vente conseillé (comportement par défaut)
                # Note: fetch_docs devrait déjà faire cette analyse si prix_propose_extrait est None.
                # Cette partie est plus pour la clarté ou si on voulait surcharger la logique de fetch_docs.
                # Laissons la logique de fetch_docs gérer le cas où prix_propose_extrait est None pour l'instant.
                # On se fie à la marge_suffisante et marge_actuelle calculées en aval si prix_propose_extrait est None.
                pass # La logique existante plus bas devrait gérer cela
            
            # Construction de la réponse détaillée
            response = {
                "status": "OK" if all([stock_suffisant, marge_suffisante, delai_compatible]) else "ATTENTION",
                "produit": produit_data,
                "analyse": {
                    "stock_suffisant": stock_suffisant,
                    "marge_suffisante": marge_suffisante,
                    "delai_compatible": delai_compatible,
                    "stock_disponible": produit_data["stock_disponible"],
                    "quantite_demandee": quantite,
                    "marge_actuelle": marge_actuelle if prix_propose_extrait is not None else produit_data.get("prix_vente_conseille", 0) - prix_achat_produit if prix_achat_produit and produit_data.get("prix_vente_conseille") else None
                }
            }
            
            # Messages détaillés selon l'analyse
            messages = []
            if not stock_suffisant:
                messages.append(f"Stock insuffisant (disponible : {produit_data['stock_disponible']}, demandé : {quantite})")
            if not marge_suffisante and prix_propose_extrait is not None: # Conditionner le message de marge au fait qu'un prix ait été proposé
                messages.append(f"Marge insuffisante avec le prix proposé de {prix_propose_extrait}€ (marge actuelle : {marge_actuelle:.2f}€, minimum requis : {marge_minimum_produit}€)")
            elif not marge_suffisante: # Si la marge est insuffisante SANS prix proposé (basé sur prix conseillé)
                 # Ce cas devrait être couvert par la logique interne de fetch_docs ou si on décidait de la recalculer ici
                 # Pour l'instant, on logue si on arrive ici et que marge_suffisante est False sans prix proposé
                 logger.warning("Marge insuffisante détectée sans prix proposé explicitement par l'utilisateur. Vérifier la logique de calcul de marge par défaut.")
            if not delai_compatible and delai:
                messages.append("Délai de livraison incompatible")
                
            if messages:
                response["message"] = " | ".join(messages)
                if alternatives:
                    response["alternatives"] = alternatives
                    response["message"] += f" | {len(alternatives)} alternatives disponibles"
            else:
                response["message"] = "Commande validée : tous les critères sont satisfaits"
                
            logger.info(f"Analyse terminée : {response['status']} - {response['message']}")
            return response
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de la commande : {str(e)}")
            return {
                "status": "ERROR",
                "message": f"Erreur lors de l'analyse : {str(e)}",
                "alternatives": []
            }
    
    def process_message(self, message: str, chat_history: List[Dict[str, Any]] = None) -> str:
        """
        Traite un message utilisateur et retourne une réponse appropriée.
        
        Args:
            message (str): Message de l'utilisateur
            chat_history (List[Dict[str, Any]], optional): Historique des messages
            
        Returns:
            str: Réponse de l'agent
            
        Raises:
            ValueError: Si le message est vide
        """
        if not message or not isinstance(message, str):
            raise ValueError("Le message doit être une chaîne non vide")
            
        if chat_history is None:
            chat_history = []
            
        try:
            logger.info(f"Traitement du message : {message[:50]}...")
            
            # Détection des demandes d'alternatives
            alternative_keywords = ["alternative", "remplacement", "autre option", "autre solution", "autre produit", "similaire"]
            is_alternative_request = any(keyword in message.lower() for keyword in alternative_keywords)
            
            if is_alternative_request:
                logger.info("Demande d'alternatives détectée, consultation du RAG")
                # Extraction du produit mentionné dans la demande
                product_id, _, _ = _extract_product_and_quantity_from_string(message, self.inventory_df)
                
                if product_id:
                    # Consultation directe du RAG pour les alternatives
                    result = fetch_docs(
                        query=f"Alternative pour {product_id}",
                        product_id=product_id,
                        required_qty=0,  # Quantité non pertinente pour une recherche d'alternatives
                        prix_propose=None  # Prix non pertinent pour une recherche d'alternatives
                    )
                    
                    if result["alternatives"]:
                        # Construction d'une réponse formatée avec les alternatives
                        response = f"Voici les alternatives disponibles pour {product_id} :\n\n"
                        for alt in result["alternatives"]:
                            response += f"- {alt['name']} :\n"
                            
                            # Affichage sécurisé de la similarité technique
                            similarite = alt.get('similarite_technique')
                            if similarite is not None:
                                if isinstance(similarite, (int, float)):
                                    response += f"  * Similarité technique : {similarite:.0%}\n"
                                else:
                                    response += f"  * Similarité technique : {similarite}\n"
                            else:
                                response += f"  * Similarité technique : À analyser par le LLM\n"
                                
                            response += f"  * Stock disponible : {alt['stock_disponible']}\n"
                            response += f"  * Délai de livraison : {alt['delai']}\n"
                            if alt.get('description'):
                                response += f"  * Caractéristiques : {alt['description'][:200]}...\n"
                            response += "\n"
                        return response
                    else:
                        return f"Je n'ai pas trouvé d'alternatives pertinentes pour {product_id}."
                else:
                    return "Je n'ai pas pu identifier le produit pour lequel vous souhaitez des alternatives. Pourriez-vous préciser le nom du produit ?"
            
            # Détection des commandes
            command_keywords = ["commande", "commander", "acheter", "prendre", "je veux", "je voudrais", "j'aimerais", "besoin de"]
            is_command = any(keyword in message.lower() for keyword in command_keywords)
            
            if is_command:
                logger.info("Commande détectée, analyse de la commande")
                
                # Séparation des commandes multiples
                command_separators = [" et ", ", ", " ainsi que ", " plus ", "\n", "\r\n"]
                commands = [message]
                for separator in command_separators:
                    new_commands = []
                    for cmd in commands:
                        if separator in cmd.lower():
                            # Nettoyer les commandes après séparation
                            split_commands = [c.strip() for c in cmd.split(separator) if c.strip()]
                            new_commands.extend(split_commands)
                        else:
                            new_commands.append(cmd)
                    commands = new_commands
                
                # Traitement de chaque commande
                all_responses = []
                for cmd in commands:
                    cmd = cmd.strip()
                    if not cmd:
                        continue
                        
                    # Utiliser directement analyser_commande pour traiter la commande
                    result = self.analyser_commande(cmd)
                    
                    # Construction de la réponse formatée pour cette commande
                    if result['status'] == "OK":
                        produit = result['produit']
                        analyse = result['analyse']
                        response = f"✅ {produit['name']} : OK\n"
                    else:
                        response = f"⚠️ {result.get('produit', {}).get('name', 'Produit non identifié')} : ATTENTION\n"
                        if result.get('alternatives'):
                            response += f"   Alternatives disponibles :\n"
                            for alt in result['alternatives']:
                                response += f"   • {alt['name']} (Stock: {alt['stock_disponible']}, Délai: {alt['delai']})\n"
                    
                    all_responses.append(response)
                
                # Combiner toutes les réponses avec une séparation claire
                return "\n".join(all_responses)
            
            # Si ce n'est pas une demande d'alternatives ni une commande, utiliser le comportement normal
            response = self.agent_executor.invoke({
                "input": message,
                "chat_history": chat_history
            })
            logger.info("Message traité avec succès")
            return response["output"]
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement du message : {str(e)}")
            return f"Désolé, une erreur s'est produite lors du traitement de votre message : {str(e)}"

# Le bloc if __name__ == "__main__": a été déplacé vers test_extraction.py 