import os
from dotenv import load_dotenv
from typing import List, Dict, Union, TypedDict, Optional, Any
import unidecode

from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import pandas as pd

# Fonction pour normaliser les noms
def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    return unidecode.unidecode(name.lower().strip())

# Chargement de l'inventaire depuis data/inventaire_stock.csv
_inventory_df = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "../data/inventaire_stock.csv"),
    sep=';',
    encoding='utf-8'
)

# Nettoyer les noms des colonnes (enlever espaces et accents)
_inventory_df.columns = (
    _inventory_df.columns
    .str.strip()
    .str.lower() # Mettre en minuscule aussi pour la cohérence
    .str.replace('é', 'e')
    .str.replace('è', 'e')
    .str.replace('ê', 'e')
    .str.replace('à', 'a')
    .str.replace('ç', 'c')
    .str.replace('_', '') # Supprimer les underscores potentiels
)

# Debug: afficher les noms des colonnes après nettoyage
print("\nNoms des colonnes après nettoyage et normalisation (lowercase, no accents, no underscore):")
print(_inventory_df.columns.tolist())

# Harmonisation des noms de colonnes (vérifier la correspondance exacte avec les noms nettoyés)
# S'assurer que les clés ici correspondent EXACTEMENT aux noms de colonnes après le nettoyage ci-dessus
_inventory_df.rename(columns={
    'productid': 'product_id', # product_id est le standard
    'nom': 'nom',
    'commandesalivrer': 'commandes_alivrer', # commandes_alivrer est le standard
    'quantitestock': 'quantite_stock', # quantite_stock est le standard
    'delailivraison': 'delai_livraison',
    'prixachat': 'prix_achat',
    'prixventeconseille': 'prix_vente_conseille', # Correction de l'accent
    'margeminimum': 'marge_minimum'
}, inplace=True)

# Debug: afficher les noms des colonnes après renommage
print("\nNoms des colonnes après renommage :")
print(_inventory_df.columns.tolist())

# Créer la colonne normalisée une seule fois
if 'nom' in _inventory_df.columns:
    _inventory_df['nom_normalise'] = _inventory_df['nom'].apply(normalize_name)
    print("\nColonne 'nom_normalise' créée.")
    # print(_inventory_df[['nom', 'nom_normalise']].head()) # Décommenter pour voir les noms normalisés
else:
    print("\nERREUR : La colonne 'nom' n'existe pas après le renommage. Vérifiez le mapping des colonnes.")
    # Créer une colonne vide pour éviter les erreurs, mais c'est un pansement
    _inventory_df['nom_normalise'] = pd.Series(dtype='str')

def find_product_in_inventory(product_identifier: str) -> pd.DataFrame:
    """
    Recherche un produit dans l'inventaire par nom normalisé ou ID.
    Gère les tentatives de pluriel/singulier, en particulier pour le premier mot.
    """
    normalized_search_term = normalize_name(product_identifier)
    print(f"[find_product_in_inventory] Recherche de: '{product_identifier}', normalisé en: '{normalized_search_term}'")

    # 1. Recherche par nom normalisé exact
    result_df = _inventory_df.loc[_inventory_df["nom_normalise"] == normalized_search_term]
    if not result_df.empty:
        print(f"[find_product_in_inventory] Trouvé (nom normalisé exact): {result_df[['nom', 'nom_normalise']].iloc[0].to_dict()}")
        return result_df

    # 2. Logique améliorée pour pluriel/singulier du premier mot si multi-mots
    search_words = normalized_search_term.split(' ')
    if len(search_words) > 0:
        first_word_search = search_words[0]
        rest_of_words_search = ' '.join(search_words[1:]) if len(search_words) > 1 else ''

        # 2a. Essayer de rendre le premier mot du terme de recherche pluriel
        if not first_word_search.endswith('s'):
            potential_plural_first_word = first_word_search + 's'
            full_potential_plural_term = potential_plural_first_word
            if rest_of_words_search:
                full_potential_plural_term += ' ' + rest_of_words_search
            
            # print(f"[find_product_in_inventory] Tentative 2a: Premier mot rendu pluriel '{full_potential_plural_term}'") # Debug
            result_df = _inventory_df.loc[_inventory_df["nom_normalise"] == full_potential_plural_term]
            if not result_df.empty:
                print(f"[find_product_in_inventory] Trouvé (premier mot rendu pluriel): {result_df[['nom', 'nom_normalise']].iloc[0].to_dict()}")
                return result_df

        # 2b. Essayer de rendre le premier mot du terme de recherche singulier
        if first_word_search.endswith('s'):
            potential_singular_first_word = first_word_search[:-1]
            full_potential_singular_term = potential_singular_first_word
            if rest_of_words_search:
                full_potential_singular_term += ' ' + rest_of_words_search
                
            # print(f"[find_product_in_inventory] Tentative 2b: Premier mot rendu singulier '{full_potential_singular_term}'") # Debug
            result_df = _inventory_df.loc[_inventory_df["nom_normalise"] == full_potential_singular_term]
            if not result_df.empty:
                print(f"[find_product_in_inventory] Trouvé (premier mot rendu singulier): {result_df[['nom', 'nom_normalise']].iloc[0].to_dict()}")
                return result_df
    
    # 3. Ancienne logique de gestion du 's' final sur la chaîne complète (comme fallback)
    # print(f"[find_product_in_inventory] Tentative 3: Gestion pluriel/singulier simple sur la chaîne entière") # Debug
    if normalized_search_term.endswith('s'):
        singular_form_full = normalized_search_term[:-1]
        result_df = _inventory_df.loc[_inventory_df["nom_normalise"] == singular_form_full]
        if not result_df.empty:
            print(f"[find_product_in_inventory] Trouvé (chaîne entière rendue singulière): {result_df[['nom', 'nom_normalise']].iloc[0].to_dict()}")
            return result_df
    else:
        plural_form_full = normalized_search_term + 's'
        result_df = _inventory_df.loc[_inventory_df["nom_normalise"] == plural_form_full]
        if not result_df.empty:
            print(f"[find_product_in_inventory] Trouvé (chaîne entière rendue plurielle): {result_df[['nom', 'nom_normalise']].iloc[0].to_dict()}")
            return result_df

    # 4. Recherche par product_id (si product_identifier est un ID)
    # print(f"[find_product_in_inventory] Tentative 4: Recherche par product_id (non normalisé): '{product_identifier}'") # Debug
    # Assurez-vous que la colonne product_id existe et est correctement nommée après le rename.
    if 'product_id' in _inventory_df.columns:
        result_df = _inventory_df.loc[_inventory_df["product_id"] == product_identifier] 
        if not result_df.empty:
            print(f"[find_product_in_inventory] Trouvé (par product_id): {result_df[['nom', 'product_id']].iloc[0].to_dict()}")
            return result_df
    else:
        print("[find_product_in_inventory] ATTENTION: Colonne 'product_id' non trouvée pour la recherche par ID.")
    
    print(f"[find_product_in_inventory] Produit NON TROUVÉ: '{product_identifier}'")
    return pd.DataFrame() # Retourne un DataFrame vide si non trouvé

def get_stock(product_identifier: str) -> int:
    """Retourne la quantité disponible en stock (quantite_stock - commandes_alivrer)."""
    
    _prod_row_df = find_product_in_inventory(product_identifier)
    
    if _prod_row_df.empty:
        print(f"Produit non trouvé dans get_stock : {product_identifier} (normalisé essayé : {normalize_name(product_identifier)})")
        return 0
        
    # S'assurer qu'on a les bonnes colonnes après renommage
    try:
        total = int(_prod_row_df["quantite_stock"].iloc[0])
        pending = int(_prod_row_df["commandes_alivrer"].iloc[0])
        stock_disponible = total - pending
        print(f"Stock trouvé pour '{product_identifier}' (trouvé comme '{_prod_row_df['nom'].iloc[0]}'): {total} - {pending} = {stock_disponible}")
        return stock_disponible
    except KeyError as e:
        print(f"ERREUR CRITIQUE dans get_stock : Clé manquante {e}. Vérifiez le renommage des colonnes.")
        print(f"Colonnes disponibles dans la ligne trouvée : {_prod_row_df.iloc[0].index.tolist()}")
        return 0
    except Exception as e_gen:
        print(f"Erreur inattendue dans get_stock lors du calcul du stock pour {product_identifier}: {e_gen}")
        return 0

from .settings import settings

class NoRelevantDocsError(ValueError):
    """Exception levée lorsqu'aucun document ne dépasse le seuil de similarité."""
    pass

# Charge les variables d'environnement depuis .env
load_dotenv()

# Initialisation du client Pinecone avec clé et région
_pc = Pinecone(
    api_key=settings.pinecone_api_key,
    environment=settings.pinecone_env,
)
# Debug: affiche la liste des index existants
print("Index disponibles :", [info["name"] for info in _pc.list_indexes()])

# Connexion à l'index configuré
_index = _pc.Index(settings.index_name)

# Configuration du modèle d'embeddings et du vector store
_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=settings.openai_api_key,
)
_vector_store = PineconeVectorStore(
    index=_index,
    embedding=_embeddings,
)

class ProductInfo(TypedDict):
    name: str
    stock_initial: int
    commandes: int
    stock_disponible: int
    delai: str
    description: Optional[str]
    prix_achat: float
    prix_vente_conseille: float
    marge_minimum: float

def format_product_info(product_name: str, row: pd.Series, stock_dispo: int, description: str = None) -> ProductInfo:
    """Formate les informations d'un produit de manière structurée."""
    # Debug: afficher les noms des colonnes disponibles
    print("\nColonnes disponibles dans la ligne :")
    print(row.index.tolist())
    
    return {
        "name": product_name,
        "stock_initial": int(row["quantite_stock"]),
        "commandes": int(row["commandes_alivrer"]),
        "stock_disponible": stock_dispo,
        "delai": row["delai_livraison"],
        "description": description,
        "prix_achat": float(row["prix_achat"]) if pd.notna(row["prix_achat"]) else 0.0,
        "prix_vente_conseille": float(row["prix_vente_conseille"]) if pd.notna(row["prix_vente_conseille"]) else 0.0,
        "marge_minimum": float(row["marge_minimum"]) if pd.notna(row["marge_minimum"]) else 0.0
    }

def fetch_docs(query: str, product_id: str = None, required_qty: int = 0, prix_propose: float = None) -> Dict[str, Union[ProductInfo, List[ProductInfo]]]:
    """
    Retourne les informations du produit demandé et ses alternatives possibles.
    """
    print("\n=== DÉBUT RECHERCHE RAG ===")
    print(f"Requête : {query}")
    print(f"Produit demandé : {product_id}")
    print(f"Quantité requise : {required_qty}")
    print(f"Prix proposé : {prix_propose}€")
    
    # Si pas de product_id, recherche simple
    if not product_id:
        print("Pas de product_id, recherche simple")
        # Recherche sémantique dans Pinecone
        docs_and_scores = _vector_store.similarity_search_with_score(
            query,
            k=settings.top_k,
        )
        
        # Si des documents sont trouvés, essayer d'extraire le produit le plus pertinent
        if docs_and_scores:
            best_doc, best_score = docs_and_scores[0]
            print(f"Meilleur document trouvé (score: {best_score})")
            
            # Essayer d'extraire le nom du produit du document
            try:
                # Recherche du nom du produit dans le document
                for product in _inventory_df['nom']:
                    if product.lower() in best_doc.page_content.lower():
                        print(f"Produit trouvé dans le document : {product}")
                        # Utiliser ce produit comme product_id
                        return fetch_docs(query, product_id=product, required_qty=required_qty, prix_propose=prix_propose)
            except Exception as e:
                print(f"Erreur lors de l'extraction du produit : {str(e)}")
        
        print("Aucun produit trouvé dans les documents")
        return {"produit": None, "alternatives": []}

    # Récupération des informations du produit demandé
    print(f"\nRecherche du produit : {product_id}")
    _prod_row = _inventory_df.loc[_inventory_df["nom"] == product_id]
    if _prod_row.empty:
        _prod_row = _inventory_df.loc[_inventory_df["product_id"] == product_id]
    
    if _prod_row.empty:
        print("Produit non trouvé dans l'inventaire")
        return {"produit": None, "alternatives": []}
        
    # Recherche de la fiche technique du produit demandé
    print("\nRecherche de la fiche technique dans Pinecone")
    product_docs = _vector_store.similarity_search_with_score(
        f"fiche technique {product_id}",
        k=1
    )
    
    # Information du produit demandé
    stock = get_stock(product_id)
    description = product_docs[0][0].page_content if product_docs else None
    print(f"Fiche technique trouvée : {'Oui' if description else 'Non'}")
    
    produit_info = format_product_info(
        product_name=_prod_row["nom"].iloc[0],
        row=_prod_row.iloc[0],
        stock_dispo=stock,
        description=description
    )
    
    # Vérification de la marge avec le prix proposé si disponible
    # Ces valeurs seront ajoutées à produit_info
    if prix_propose is not None and produit_info.get('prix_achat') is not None:
        marge_actuelle_calculee = prix_propose - produit_info['prix_achat']
        print(f"Prix proposé: {prix_propose}€, Prix d'achat: {produit_info['prix_achat']}€")
    elif produit_info.get('prix_vente_conseille') is not None and produit_info.get('prix_achat') is not None:
        marge_actuelle_calculee = produit_info['prix_vente_conseille'] - produit_info['prix_achat']
        print(f"Prix conseillé: {produit_info['prix_vente_conseille']}€, Prix d'achat: {produit_info['prix_achat']}€")
    else:
        marge_actuelle_calculee = 0.0 # Cas par défaut si infos manquantes
        print(f"Impossible de calculer la marge actuelle, informations manquantes dans produit_info: {produit_info}")

    marge_minimum_produit = produit_info.get('marge_minimum', 0.0)
    marge_suffisante_calculee = marge_actuelle_calculee >= marge_minimum_produit

    # Enrichir produit_info avec les résultats de l'analyse de marge
    produit_info['marge_actuelle'] = marge_actuelle_calculee
    produit_info['marge_suffisante'] = marge_suffisante_calculee
    produit_info['prix_propose_retenu'] = prix_propose # Garder une trace du prix qui a servi au calcul
    
    print(f"\nVérification des conditions (dans fetch_docs après enrichissement produit_info) :")
    print(f"- Stock disponible : {stock} (requis : {required_qty})")
    print(f"- Marge actuelle : {produit_info['marge_actuelle']:.2f}€ (minimum requis : {produit_info['marge_minimum']}€)")
    print(f"- Marge suffisante : {'Oui' if produit_info['marge_suffisante'] else 'Non'}")
    
    # Si stock suffisant ET marge suffisante (selon produit_info enrichi), pas besoin d'alternatives
    if stock >= required_qty and produit_info['marge_suffisante']:
        print("Stock et marge suffisants, mais recherche d'alternatives quand même pour information")
        # On continue la recherche d'alternatives même si le produit initial est valide
        
    print(f"\nRecherche d'alternatives :")
    print(f"- Stock insuffisant : {stock < required_qty}")
    print(f"- Marge insuffisante : {not produit_info['marge_suffisante']}")
    
    # Configuration pour la recherche d'alternatives améliorée
    max_alternatives = 10  # Augmentation significative du nombre d'alternatives
    min_similarity_score = settings.score_threshold * 0.7  # Seuil plus permissif
    
    # Recherche d'alternatives dans Pinecone
    query_to_use = f"Alternative pour {produit_info['name']} : {query}"
    if not produit_info['marge_suffisante']:
        query_to_use += " avec caractéristiques similaires et meilleure marge"
    
    print(f"\nRequête Pinecone : {query_to_use}")
    docs_and_scores = _vector_store.similarity_search_with_score(
        query_to_use,
        k=max_alternatives,  # Récupération de plus de résultats
    )
    print(f"Nombre de résultats Pinecone : {len(docs_and_scores)}")
    
    # Recherche additionnelle pour enrichir les alternatives
    additional_searches = [
        f"fiche technique similaire {produit_info['name']}",
        f"produit équivalent {produit_info['name']}",
        f"alternative technique {produit_info['name']}",
    ]
    
    for additional_query in additional_searches:
        print(f"\nRecherche supplémentaire : {additional_query}")
        additional_docs = _vector_store.similarity_search_with_score(
            additional_query,
            k=max_alternatives // 2,
        )
        
        # Fusionner les résultats en évitant les doublons
        seen_docs = set(doc.page_content for doc, score in docs_and_scores)
        for doc, score in additional_docs:
            if doc.page_content not in seen_docs and score >= min_similarity_score:
                docs_and_scores.append((doc, score))
                seen_docs.add(doc.page_content)
    
    print(f"Nombre total de résultats après recherches supplémentaires : {len(docs_and_scores)}")
    
    # Tri par score de similarité décroissant
    docs_and_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Filtrage et analyse des alternatives
    alternatives = []
    seen_products = set()  # Pour éviter les doublons
    
    # 1. Analyser d'abord les résultats de Pinecone
    for doc, score in docs_and_scores:
        if score < min_similarity_score:
            print(f"Score trop faible ({score:.3f}) pour un résultat")
            continue
            
        content = doc.page_content
        print(f"\nAnalyse d'un résultat Pinecone (score : {score})")
        print(f"Contenu : {content[:200]}...")
        
        # Recherche de tous les produits mentionnés dans le contenu
        produits_trouves = []
        for product in _inventory_df['nom']:
            if (product.lower() in content.lower() or 
                any(word in product.lower() for word in content.lower().split()) and 
                product != produit_info['name']):
                produits_trouves.append(product)
                seen_products.add(product)  # Marquer comme vu
        
        if not produits_trouves:
            print("Aucun produit trouvé dans le résultat")
            continue
            
        print(f"Produits trouvés dans le résultat : {produits_trouves}")
        
        # Analyser chaque produit trouvé
        for product in produits_trouves:
            alternative_info = analyze_product_alternative(
                product=product,
                required_qty=required_qty,
                prix_propose=prix_propose,
                content=content,
                score=score,
                produit_info=produit_info
            )
            if alternative_info:
                alternatives.append(alternative_info)
    
    # 2. Enrichir avec tous les produits similaires de l'inventaire
    print("\nRecherche de produits similaires dans l'inventaire...")
    for _, row in _inventory_df.iterrows():
        product = row['nom']
        if product != produit_info['name'] and product not in seen_products:
            # Vérifier si le produit est similaire (même catégorie)
            if any(keyword in product.lower() for keyword in ['caisse', 'boite', 'etui']):
                print(f"\nAnalyse du produit de l'inventaire : {product}")
                # Recherche de la fiche technique
                alt_docs = _vector_store.similarity_search_with_score(
                    f"fiche technique {product}",
                    k=1
                )
                content = alt_docs[0][0].page_content if alt_docs else None
                
                alternative_info = analyze_product_alternative(
                    product=product,
                    required_qty=required_qty,
                    prix_propose=prix_propose,
                    content=content,
                    score=0.5,  # Score par défaut pour les produits de l'inventaire
                    produit_info=produit_info
                )
                if alternative_info:
                    alternatives.append(alternative_info)
    
    # 3. La similarité technique sera analysée par le LLM - pas de calcul automatique
    for alt in alternatives:
        alt['similarite_technique'] = None  # Le LLM analysera la compatibilité technique
        print(f"Alternative trouvée : {alt['name']} - Compatibilité technique à analyser par le LLM")
    
    # 4. Tri des alternatives par marge et stock (le LLM analysera la similarité technique)
    alternatives.sort(key=lambda x: (
        x['marge'] - x['marge_minimum'],   # Priorité à la marge
        x.get('stock_disponible', 0)      # Puis au stock disponible
    ), reverse=True)
    
    # 5. Supprimer les doublons en gardant la meilleure version de chaque produit
    seen_names = set()
    unique_alternatives = []
    for alt in alternatives:
        if alt['name'] not in seen_names:
            seen_names.add(alt['name'])
            unique_alternatives.append(alt)
    
    # Garder plus d'alternatives pour l'analyse LLM (jusqu'à 8)
    max_final_alternatives = 8
    alternatives = unique_alternatives[:max_final_alternatives]
    
    print(f"\nNombre d'alternatives trouvées : {len(alternatives)}")
    print("=== FIN RECHERCHE RAG ===\n")

    return {
        "produit": produit_info,
        "alternatives": alternatives
    }

def extract_technical_features(description: str) -> dict:
    """Extrait les caractéristiques techniques d'une description."""
    features = {}
    if description and ("technical_details" in description.lower() or "caracteristiques_techniques" in description.lower()):
        try:
            # Recherche des caractéristiques dans le contenu
            for line in description.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    if key and value:
                        features[key] = value
        except Exception as e:
            print(f"Erreur lors de l'extraction des caractéristiques : {str(e)}")
    return features

def calculate_technical_similarity(features1: dict, features2: dict) -> float:
    """Calcule la similarité entre deux ensembles de caractéristiques techniques."""
    if not features1 or not features2:
        return 0.0
    
    # Compte le nombre de caractéristiques communes
    common_features = 0
    total_features = 0
    
    for key, value1 in features1.items():
        if key in features2:
            total_features += 1
            if features2[key] == value1:
                common_features += 1
    
    # Ajoute les caractéristiques uniques
    total_features += len(set(features1.keys()) - set(features2.keys()))
    total_features += len(set(features2.keys()) - set(features1.keys()))
    
    if total_features == 0:
        return 0.0
    
    return common_features / total_features

def analyze_product_alternative(product: str, required_qty: int, prix_propose: float, 
                              content: str, score: float, produit_info: dict) -> Optional[dict]:
    """Analyse un produit alternatif et retourne ses informations si valide."""
    # Le calcul de similarité technique est maintenant délégué au LLM
    # Plus de seuil automatique ici
    
    # Vérification du stock et de la marge dans le CSV
    product_stock = get_stock(product)
    prod_row = _inventory_df.loc[_inventory_df["nom"] == product].iloc[0]
    
    # Calcul de la marge pour l'alternative
    prix_achat = float(prod_row["prix_achat"]) if pd.notna(prod_row["prix_achat"]) else 0.0
    prix_vente = float(prod_row["prix_vente_conseille"]) if pd.notna(prod_row["prix_vente_conseille"]) else 0.0
    marge_min = float(prod_row["marge_minimum"]) if pd.notna(prod_row["marge_minimum"]) else 0.0
    
    # Calcul de la marge avec le prix proposé
    if prix_propose is not None:
        marge_alt = prix_propose - prix_achat
    else:
        marge_alt = prix_vente - prix_achat
    
    print(f"\nAnalyse de l'alternative : {product}")
    print(f"- Stock : {product_stock} (requis : {required_qty})")
    print(f"- Marge actuelle : {marge_alt}€")
    print(f"- Marge minimum requise : {marge_min}€")
    
    # Vérification de la marge
    marge_suffisante = marge_alt >= marge_min
    
    # Si le problème est la marge, on ne garde que les alternatives avec une marge suffisante
    if not marge_suffisante:
        print("Marge insuffisante, alternative rejetée")
        return None
    
    # La similarité technique sera analysée par le LLM, pas de calcul automatique ici
    print("→ Similarité technique : Sera analysée par le LLM")
    
    # Si le problème est le stock, on vérifie juste le stock
    if product_stock >= required_qty:
        print("Alternative valide, ajoutée à la liste")
        alternative_info = format_product_info(
            product_name=product,
            row=prod_row,
            stock_dispo=product_stock,
            description=content
        )
        # Ajout du score de similarité pour le tri
        alternative_info['score'] = score
        
        # La similarité technique sera calculée par le LLM
        alternative_info['similarite_technique'] = None  # À analyser par le LLM
        
        # Ajout de la marge pour le tri
        alternative_info['marge'] = marge_alt
        alternative_info['marge_minimum'] = marge_min
        
        # Ajout des informations complètes de l'inventaire
        alternative_info['stock_initial'] = int(prod_row["quantite_stock"])
        alternative_info['commandes'] = int(prod_row["commandes_alivrer"])
        alternative_info['delai'] = prod_row["delai_livraison"]
        
        return alternative_info
    else:
        print("Stock insuffisant, alternative rejetée")
        return None

def fetch_docs_for_products(query: str, orders: Dict[str, int], prix_propose: float = None) -> Dict[str, Dict[str, Union[ProductInfo, List[ProductInfo]]]]:
    """Traite plusieurs produits en une seule requête."""
    results = {}
    for pid in orders:
        qty = orders.get(pid, 0)
        results[pid] = fetch_docs(query, product_id=pid, required_qty=qty, prix_propose=prix_propose)
    return results

def enrich_alternatives_for_llm(alternatives: List[Dict[str, Any]], 
                              produit_demande: Dict[str, Any],
                              prix_propose: float = None) -> List[Dict[str, Any]]:
    """
    Enrichit les données des alternatives pour faciliter l'analyse LLM
    
    Args:
        alternatives: Liste des alternatives trouvées
        produit_demande: Produit demandé originalement  
        prix_propose: Prix proposé pour le calcul des marges
        
    Returns:
        List: Alternatives enrichies avec plus d'informations structurées
    """
    alternatives_enrichies = []
    
    for alt in alternatives:
        alt_enrichie = alt.copy()
        
        # Calcul de la marge avec le prix proposé si disponible
        if prix_propose and alt.get('prix_achat'):
            marge_avec_prix_propose = prix_propose - alt['prix_achat']
            alt_enrichie['marge_avec_prix_propose'] = marge_avec_prix_propose
            alt_enrichie['marge_suffisante_prix_propose'] = marge_avec_prix_propose >= alt.get('marge_minimum', 0)
        
        # Comparaison avec le produit demandé
        alt_enrichie['comparaison'] = {
            'similarite_technique': alt.get('similarite_technique', 0),
            'difference_prix_achat': alt.get('prix_achat', 0) - produit_demande.get('prix_achat', 0),
            'difference_marge_minimum': alt.get('marge_minimum', 0) - produit_demande.get('marge_minimum', 0),
            'avantage_stock': alt.get('stock_disponible', 0) > produit_demande.get('stock_disponible', 0)
        }
        
        # Analyse des caractéristiques techniques simplifiées
        if alt.get('description'):
            caracteristiques = extract_technical_features(alt['description'])
            alt_enrichie['caracteristiques_cles'] = list(caracteristiques.keys())[:5]  # Top 5
        
        # Score de recommandation global (pour aider le LLM) - sans similarité technique automatique
        score_global = 0
        score_global += (1 if alt.get('stock_disponible', 0) > 0 else 0) * 0.5  # 50% stock
        score_global += (1 if alt.get('marge', 0) >= alt.get('marge_minimum', 0) else 0) * 0.5  # 50% marge
        
        alt_enrichie['score_recommandation'] = score_global
        
        alternatives_enrichies.append(alt_enrichie)
    
    # Tri par score de recommandation
    alternatives_enrichies.sort(key=lambda x: x.get('score_recommandation', 0), reverse=True)
    
    return alternatives_enrichies