import os
import time
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path
from typing import Optional, Dict, Any
import hashlib
from .retrieval import _inventory_df, _vector_store
from .settings import settings

class InventoryWatcher(FileSystemEventHandler):
    def __init__(self, inventory_path: str):
        self.inventory_path = inventory_path
        self.last_hash = self._get_file_hash()
        self.last_modified = os.path.getmtime(inventory_path)
        
    def _get_file_hash(self) -> str:
        """Calcule le hash du fichier d'inventaire."""
        with open(self.inventory_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
            
    def _load_inventory(self) -> pd.DataFrame:
        """Charge l'inventaire depuis le fichier CSV."""
        df = pd.read_csv(
            self.inventory_path,
            sep=';',
            encoding='utf-8'
        )
        
        # Nettoyage des noms de colonnes
        df.columns = (
            df.columns
            .str.strip()
            .str.replace('é', 'e')
            .str.replace('è', 'e')
            .str.replace('ê', 'e')
            .str.replace('à', 'a')
            .str.replace('ç', 'c')
        )
        
        return df
        
    def _update_pinecone(self, new_df: pd.DataFrame) -> None:
        """Met à jour la base Pinecone avec les nouveaux produits."""
        # Préparation des documents pour Pinecone
        documents = []
        for _, row in new_df.iterrows():
            # Création du texte descriptif du produit
            product_text = f"""
            Fiche produit : {row['Nom']}
            Caractéristiques techniques :
            - Type : {row['Nom']}
            - Stock initial : {row['quantite_stock']}
            - Commandes à livrer : {row['Commandes_alivrer']}
            - Délai de réapprovisionnement : {row['delai_livraison']}
            - Prix d'achat : {row['prix_achat']}€
            - Prix de vente conseillé : {row['prix_vente_conseillé']}€
            - Marge minimum requise : {row['marge_minimum']}€
            
            Description détaillée :
            Ce produit est une {row['Nom']} avec les caractéristiques suivantes :
            - Stock disponible : {int(row['quantite_stock']) - int(row['Commandes_alivrer'])}
            - Délai de réapprovisionnement : {row['delai_livraison']}
            - Marge minimum : {row['marge_minimum']}€
            """
            
            # Ajout des métadonnées
            metadata = {
                "product_id": row['Product_id'],
                "name": row['Nom'],
                "stock": int(row['quantite_stock']),
                "pending_orders": int(row['Commandes_alivrer']),
                "delivery_time": row['delai_livraison'],
                "purchase_price": float(row['prix_achat']),
                "suggested_price": float(row['prix_vente_conseillé']),
                "minimum_margin": float(row['marge_minimum'])
            }
            
            documents.append({
                "text": product_text,
                "metadata": metadata
            })
        
        # Mise à jour de Pinecone
        if documents:
            _vector_store.add_texts(
                texts=[doc["text"] for doc in documents],
                metadatas=[doc["metadata"] for doc in documents]
            )
            
    def on_modified(self, event):
        """Gère les événements de modification du fichier."""
        if event.src_path == self.inventory_path:
            current_hash = self._get_file_hash()
            current_modified = os.path.getmtime(self.inventory_path)
            
            # Vérifie si le fichier a réellement changé
            if current_hash != self.last_hash and current_modified > self.last_modified:
                print("\n=== DÉTECTION DE MODIFICATIONS DANS L'INVENTAIRE ===")
                print("Mise à jour de l'inventaire en cours...")
                
                try:
                    # Chargement du nouvel inventaire
                    new_df = self._load_inventory()
                    
                    # Mise à jour de l'inventaire en mémoire
                    global _inventory_df
                    _inventory_df = new_df
                    
                    # Mise à jour de Pinecone
                    self._update_pinecone(new_df)
                    
                    print("Mise à jour terminée avec succès !")
                    
                    # Mise à jour des références
                    self.last_hash = current_hash
                    self.last_modified = current_modified
                    
                except Exception as e:
                    print(f"Erreur lors de la mise à jour : {str(e)}")
                print("=== FIN DE LA MISE À JOUR ===\n")

def start_inventory_watcher() -> None:
    """Démarre la surveillance de l'inventaire."""
    inventory_path = os.path.join(os.path.dirname(__file__), "../data/inventaire_stock.csv")
    
    # Création du dossier de surveillance si nécessaire
    Path(os.path.dirname(inventory_path)).mkdir(parents=True, exist_ok=True)
    
    # Initialisation du watcher
    event_handler = InventoryWatcher(inventory_path)
    observer = Observer()
    observer.schedule(event_handler, path=os.path.dirname(inventory_path), recursive=False)
    observer.start()
    
    print(f"\nSurveillance de l'inventaire démarrée sur : {inventory_path}")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\nSurveillance de l'inventaire arrêtée.")
    
    observer.join() 