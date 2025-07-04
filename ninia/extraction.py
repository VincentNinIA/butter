import re
import unidecode
import pandas as pd
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

def _normalize_text_for_extraction(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return unidecode.unidecode(text.lower().strip())

def _strip_command_keywords(text: str, keywords: List[str]) -> str:
    stripped_text = text
    for keyword in keywords:
        if stripped_text.startswith(keyword + " "):
            stripped_text = stripped_text[len(keyword)+1:].lstrip()
        elif stripped_text == keyword:
            stripped_text = ""
            break
    return stripped_text

def _extract_product_and_quantity_from_string(text: str, inventory_df: pd.DataFrame) -> tuple[Optional[str], Optional[int], Optional[float]]:
    # Log du texte original pour debug
    logger.info(f"_extract: Texte original pour extraction prix: '{text}' (repr: {repr(text)})")
    # Remplacement de tous les espaces Unicode par un espace standard
    import re as _re
    text_clean = _re.sub(r'\s', ' ', text)
    # Remplacement explicite de tous les 'à' Unicode par 'a'
    text_clean = text_clean.replace('à', 'a')
    prix_propose = None
    price_patterns = [
        r"(?:a|pour|de)\s*([\d]+(?:[.,]\d{1,2})?)\s*(?:euros?|eur|€)",
        r"([\d]+(?:[.,]\d{1,2})?)\s*(?:euros?|eur|€)",
        r"prix(?:[\s]*est[\s]*de|[\s]*de|[\s]*:[\s]*|[\s]+)?([\d]+(?:[.,]\d{1,2})?)"
    ]
    original_text = text_clean
    logger.info(f"_extract: Texte nettoyé pour extraction prix: '{original_text}' (repr: {repr(original_text)})")
    for pattern in price_patterns:
        match_price = re.search(pattern, original_text, re.IGNORECASE)
        if match_price:
            try:
                prix_str = match_price.group(1).replace(',', '.')
                prix_propose = float(prix_str)
                logger.info(f"_extract: Prix proposé trouvé: {prix_propose}€ à partir de '{match_price.group(0)}'")
                original_text = original_text.replace(match_price.group(0), "", 1).strip()
                break
            except ValueError:
                logger.warning(f"_extract: Impossible de convertir le prix trouvé '{match_price.group(1)}' en float.")
    normalized_text = _normalize_text_for_extraction(original_text)
    quantity = None
    product_text_part = normalized_text
    command_units = [
        "unite", "unites", "rouleau", "rouleaux", "exemplaire", "exemplaires",
        "piece", "pieces", "caisse", "caisses", "boite", "boites", "palette", "palettes",
        "sac", "sacs", "carton", "cartons"
    ]
    technical_units = ["um", "microns", "mm", "cm", "m", "kg", "g", "l", "ml"]
    command_keywords_for_stripping = sorted(["commande de", "commande", "acheter", "prendre", "veux", "voudrais", "besoin de", "pour", "donne moi"], key=len, reverse=True)
    potential_quantities = []
    for unit in command_units:
        pattern_qty_unit = rf"(\d+)\s*({unit})\b"
        for match in re.finditer(pattern_qty_unit, normalized_text):
            qty_candidate = int(match.group(1))
            matched_text = match.group(0)
            temp_product_text = normalized_text.replace(matched_text, "", 1).strip()
            temp_product_text = re.sub(r"^\s*[,_\-\s]*", "", temp_product_text)
            temp_product_text = re.sub(r"[,_\-\s]*\s*$", "", temp_product_text)
            potential_quantities.append({'q': qty_candidate, 'p': temp_product_text, 'match': matched_text, 'type': 'qty_unit'})
            logger.info(f"_extract: Candidat (qty_unit) trouvé: Q={qty_candidate}, P='{temp_product_text}' à partir de '{matched_text}'")
    if not potential_quantities:
        all_numbers_matches = list(re.finditer(r"(\d+)", normalized_text))
        for i, match_num in enumerate(all_numbers_matches):
            num_val = int(match_num.group(1))
            start_idx, end_idx = match_num.span()
            is_technical = False
            substring_after_num = normalized_text[end_idx:end_idx+10].strip().lower()
            for tech_unit in technical_units:
                if substring_after_num.startswith(tech_unit):
                    is_technical = True
                    logger.info(f"_extract: Nombre {num_val} identifié comme technique (suivi de '{tech_unit}')")
                    break
            if not is_technical:
                temp_product_text = (normalized_text[:start_idx] + normalized_text[end_idx:]).strip()
                temp_product_text = re.sub(r"^\s*[,_\-\s]*", "", temp_product_text)
                temp_product_text = re.sub(r"[,_\-\s]*\s*$", "", temp_product_text)
                command_keywords = ["commande de", "commande", "acheter", "prendre", "veux", "voudrais", "besoin de", "pour"]
                preceding_text = normalized_text[:start_idx].strip()
                is_preceded_by_keyword = any(preceding_text.endswith(keyword) for keyword in command_keywords)
                potential_quantities.append({
                    'q': num_val,
                    'p': temp_product_text,
                    'match': match_num.group(1),
                    'type': 'isolated_num',
                    'is_preceded': is_preceded_by_keyword,
                    'pos': start_idx
                })
                logger.info(f"_extract: Candidat (isolated_num) trouvé: Q={num_val}, P='{temp_product_text}', Précédé par mot-clé: {is_preceded_by_keyword}")
    if potential_quantities:
        qty_unit_matches = [p for p in potential_quantities if p['type'] == 'qty_unit']
        if qty_unit_matches:
            chosen_candidate = qty_unit_matches[-1]
            quantity = chosen_candidate['q']
            product_text_part = _normalize_text_for_extraction(chosen_candidate['p'])
            logger.info(f"_extract: Choix initial (qty_unit): Q={quantity}, P='{product_text_part}'")
        else:
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
    if product_text_part:
        product_text_part = _strip_command_keywords(product_text_part, command_keywords_for_stripping)
        logger.info(f"_extract: Product text part APRÈS nettoyage des mots-clés: '{product_text_part}'")
    if quantity is not None and not product_text_part.strip():
        logger.warning(f"_extract: Quantité {quantity} trouvée mais nom de produit vide. Retour à texte original pour produit.")
        product_text_part = normalized_text
    if 'nom_normalise' not in inventory_df.columns:
        if 'nom' in inventory_df.columns:
            logger.warning("La colonne 'nom_normalise' n'était pas dans l'inventaire pour l'extraction, création à la volée.")
            inventory_df['nom_normalise'] = inventory_df['nom'].apply(_normalize_text_for_extraction)
        else:
            logger.error("Colonne 'nom' et 'nom_normalise' manquantes dans l'inventaire pour l'extraction.")
            return None, quantity, prix_propose
    best_match_product_name = None
    if product_text_part:
        exact_match = inventory_df[inventory_df['nom_normalise'] == product_text_part]
        if not exact_match.empty:
            best_match_product_name = exact_match['nom'].iloc[0]
        else:
            best_overlap = 0
            text_part_tokens = set(product_text_part.split())
            for index, row in inventory_df.iterrows():
                prod_norm = row['nom_normalise']
                original_name = row['nom']
                
                # 1. Match exact (priorité absolue)
                if prod_norm == product_text_part:
                    best_match_product_name = original_name
                    best_overlap = 1000  # Score maximum
                    break
                
                # 2. Inclusion complète (très haute priorité)
                elif prod_norm in product_text_part:
                    score = len(prod_norm) + 500  # Bonus d'inclusion
                    if score > best_overlap:
                        best_overlap = score
                        best_match_product_name = original_name
                elif product_text_part in prod_norm:
                    score = len(product_text_part) + 400  # Bonus d'inclusion inverse
                    if score > best_overlap:
                        best_overlap = score
                        best_match_product_name = original_name
                
                # 3. Matching par tokens (nouvelle logique améliorée)
                else:
                    prod_tokens = set(prod_norm.split())
                    common_tokens = text_part_tokens.intersection(prod_tokens)
                    
                    if common_tokens and len(text_part_tokens) > 0:
                        # Score basé sur le pourcentage de tokens communs
                        coverage_score = len(common_tokens) / len(text_part_tokens)
                        
                        # Bonus si tous les tokens recherchés sont trouvés
                        if len(common_tokens) == len(text_part_tokens):
                            coverage_score += 1.0  # Bonus de couverture complète
                        
                        # Bonus pour les tokens importants (nombres, unités techniques)
                        important_tokens = {'um', 'mm', 'cm', 'microns'} | {str(i) for i in range(1, 100)}
                        important_common = common_tokens.intersection(important_tokens)
                        if important_common:
                            coverage_score += 0.5 * len(important_common)
                        
                        # Score final pondéré
                        final_score = coverage_score * 100
                        
                        if final_score > best_overlap:
                            best_overlap = final_score
                            best_match_product_name = original_name
                            
            if best_match_product_name:
                logger.info(f"_extract: Final (match partiel chevauchant) - Produit: '{best_match_product_name}', Score: {best_overlap}")
            else:
                logger.info(f"_extract: Aucun match de produit suffisamment bon trouvé pour '{product_text_part}'")
    if best_match_product_name is None and product_text_part:
        pass
    logger.info(f"_extract: Final - Produit: '{best_match_product_name}', Quantité: {quantity}, Prix proposé: {prix_propose}")
    return best_match_product_name, quantity, prix_propose

def extract_entities(message: str):
    """
    Extrait le produit, la quantité et le prix d'un message utilisateur.
    Retourne (produit, quantite, prix) ou (None, None, None) si non trouvé.
    """
    # À implémenter
    return None, None, None 