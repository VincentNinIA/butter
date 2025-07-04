import logging
from typing import List, Dict, Any, Optional
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from rag.retrieval import fetch_docs, _inventory_df
from rag.core import answer
from .order_analysis import analyser_commande
from .extraction import _extract_product_and_quantity_from_string

logger = logging.getLogger(__name__)

class NiniaAgent:
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
        """
        if not product_name_or_id or not isinstance(product_name_or_id, str):
            raise ValueError("L'identifiant du produit doit être une chaîne non vide")
            
        try:
            logger.info(f"Vérification du stock pour le produit : {product_name_or_id}")
            from rag.retrieval import get_stock
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

    def analyser_commande(self, user_query_for_order: str) -> Dict[str, Any]:
        """
        Analyse une commande complète en vérifiant le stock, la marge et les alternatives.
        
        Args:
            user_query_for_order (str): La requête brute de l'utilisateur pour la commande.
            
        Returns:
            Dict[str, Any]: Résultat de l'analyse avec recommandations.
        """
        return analyser_commande(user_query_for_order, self.inventory_df, self.llm)

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
            
            # Détection des commandes implicites (quantité + produit + prix)
            import re
            quantity_pattern = r'\b\d+\s+'
            price_pattern = r'\s+à\s+\d+[€€]|\s+\d+[€€]'
            has_quantity = bool(re.search(quantity_pattern, message))
            has_price = bool(re.search(price_pattern, message))
            
            # Si on a une quantité ET un prix, c'est probablement une commande
            if has_quantity and has_price:
                is_command = True
                logger.info("Commande implicite détectée (quantité + prix)")
            
            if is_command:
                logger.info("Commande détectée, extraction des produits avec LLM")
                
                # Utiliser l'extraction LLM pour obtenir tous les produits d'un coup
                from .extraction_llm import extract_multiple_orders_with_llm
                products = extract_multiple_orders_with_llm(message, self.llm)
                
                if not products:
                    logger.warning("Aucun produit extrait par le LLM, fallback vers analyser_commande")
                    # Fallback vers l'ancienne méthode si l'extraction LLM échoue
                    result = self.analyser_commande(message)
                    if result is None:
                        return "Désolé, je n'ai pas pu identifier de produits dans votre message."
                    
                    # Gestion du cas où analyser_commande retourne None
                    if result is None:
                        return "Désolé, je n'ai pas pu identifier de produits dans votre message."
                    
                    produit = result.get('produit', {})
                    if not produit.get('name'):
                        return "Désolé, je n'ai pas pu identifier de produits valides dans votre message."
                        
                    # Format simple pour un seul produit
                    analyse = result.get('analyse', {})
                    if analyse.get('marge_suffisante', True) and analyse.get('stock_suffisant', True):
                        return f"✅ {produit.get('name', 'Produit')} : OK"
                    else:
                        return f"❌ {produit.get('name', 'Produit')} : Problèmes détectés"
                
                # Traitement de chaque produit extrait
                all_responses = []
                for product_info in products:
                    product_name = product_info.get('product_name')
                    quantity = product_info.get('quantity')
                    proposed_price = product_info.get('proposed_price')
                    
                    if not product_name:
                        continue
                    
                    # Créer une commande formatée pour analyser_commande
                    cmd_parts = []
                    if quantity:
                        cmd_parts.append(str(quantity))
                    cmd_parts.append(product_name)
                    if proposed_price:
                        cmd_parts.append(f"à {proposed_price}€")
                    
                    formatted_cmd = " ".join(cmd_parts)
                    logger.info(f"Analyse du produit formaté : {formatted_cmd}")
                    
                    # Utiliser analyser_commande pour traiter ce produit spécifique
                    result = self.analyser_commande(formatted_cmd)
                    
                    # Vérifier que result n'est pas None
                    if result is None:
                        logger.error(f"analyser_commande a retourné None pour: {formatted_cmd}")
                        all_responses.append(f"❌ {product_name} : Erreur d'analyse")
                        continue
                    
                    # Debug: afficher les détails du résultat
                    logger.info(f"DEBUG - Résultat analyse: status={result.get('status')}, produit={result.get('produit', {}).get('name')}")
                    logger.info(f"DEBUG - Analyse: stock_suffisant={result.get('analyse', {}).get('stock_suffisant')}, marge_suffisante={result.get('analyse', {}).get('marge_suffisante')}")
                    
                    # Construction de la réponse formatée pour cette commande
                    produit = result.get('produit', {})
                    analyse = result.get('analyse', {})
                    alternatives = result.get('alternatives', [])
                    print(f"DEBUG analyse pour {produit.get('name', 'Produit inconnu')} : {analyse}")  # DEBUG
                    problemes = []

                    # Vérification des critères bloquants
                    if not analyse.get('marge_suffisante', True):
                        marge_min = produit.get('marge_minimum', analyse.get('marge_minimum', 'N/A'))
                        problemes.append(f"Marge insuffisante (actuelle: {analyse.get('marge_actuelle', 'N/A')}€, minimum requise: {marge_min}€)")
                    if not analyse.get('stock_suffisant', True):
                        problemes.append(f"Stock insuffisant (disponible: {analyse.get('stock_disponible', 'N/A')}, demandé: {analyse.get('quantite_demandee', 'N/A')})")
                    if not analyse.get('delai_compatible', True):
                        problemes.append(f"Délai de livraison incompatible (demandé: {analyse.get('delai_demande', 'N/A')}, dispo: {analyse.get('delai_dispo', 'N/A')})")

                    if problemes:
                        response = f"❌ {produit.get('name', 'Produit inconnu')} : REFUSÉ\n" + "\n".join(problemes) + "\n"
                        
                        # Ajout des alternatives si disponibles
                        if alternatives:
                            response += "\n🔄 **Alternatives disponibles :**\n"
                            for i, alt in enumerate(alternatives[:3], 1):
                                response += f"\n{i}. **{alt.get('name', 'Produit inconnu')}**\n"
                                response += f"   • Stock disponible : {alt.get('stock_disponible', 'N/A')}\n"
                                response += f"   • Délai de livraison : {alt.get('delai', 'N/A')}\n"
                                
                                # Affichage de la description complète (non tronquée)
                                description = alt.get('description', '')
                                if description:
                                    if len(description) > 300:
                                        response += f"   • Caractéristiques : {description[:300]}...\n"
                                    else:
                                        response += f"   • Caractéristiques : {description}\n"
                                
                                # Affichage de la similarité technique si disponible
                                similarite = alt.get('similarite_technique')
                                if similarite is not None:
                                    if isinstance(similarite, (int, float)):
                                        response += f"   • Similarité technique : {similarite:.0%}\n"
                                    else:
                                        response += f"   • Similarité technique : {similarite}\n"
                                else:
                                    response += f"   • Similarité technique : À analyser par le LLM\n"
                                
                                response += "\n"
                        else:
                            response += "\n❌ Aucune alternative disponible pour le moment.\n"
                    else:
                        response = f"✅ {produit.get('name', 'Produit inconnu')} : OK\n"
                    
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