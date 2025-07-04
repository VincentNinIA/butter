import os
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from rag.settings import settings

def load_inventory_to_pinecone():
    print("=== CHARGEMENT DE L'INVENTAIRE DANS PINECONE ===")
    
    # Chargement de l'inventaire
    inventory_df = pd.read_csv(
        "data/inventaire_stock.csv",
        sep=';',
        encoding='utf-8'
    )
    
    # Nettoyage des noms de colonnes
    inventory_df.columns = (
        inventory_df.columns
        .str.strip()
        .str.replace('é', 'e')
        .str.replace('è', 'e')
        .str.replace('ê', 'e')
        .str.replace('à', 'a')
        .str.replace('ç', 'c')
    )
    
    # Initialisation de Pinecone
    pc = Pinecone(
        api_key=settings.pinecone_api_key,
        environment=settings.pinecone_env,
    )
    
    # Configuration du modèle d'embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=settings.openai_api_key,
    )
    
    # Création du vector store
    vector_store = PineconeVectorStore(
        index=pc.Index(settings.index_name),
        embedding=embeddings,
    )
    
    # Préparation des documents pour Pinecone
    documents = []
    for _, row in inventory_df.iterrows():
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
    
    # Chargement dans Pinecone
    print(f"Chargement de {len(documents)} produits dans Pinecone...")
    vector_store.add_texts(
        texts=[doc["text"] for doc in documents],
        metadatas=[doc["metadata"] for doc in documents]
    )
    print("Chargement terminé !")

if __name__ == "__main__":
    load_inventory_to_pinecone() 