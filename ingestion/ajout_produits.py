import os
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone

# Charger les variables d'environnement
load_dotenv()

# Initialiser Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index("sample-index")

# Initialiser les embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.environ.get("OPENAI_API_KEY")
)

# Créer le vector store
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

def get_next_available_id(vector_store, total_vectors):
    """
    Trouve le prochain ID disponible en analysant les IDs existants.
    """
    existing_ids = set()
    results = vector_store.similarity_search_with_score("", k=total_vectors)
    
    for doc, _ in results:
        try:
            content = doc.page_content.strip()
            # Nettoyer le contenu JSON
            content = content.replace('\n', ' ').replace('\r', '')
            content = ' '.join(content.split())
            # Parser le JSON
            json_content = json.loads(content)
            prod_id = json_content.get("id")
            if prod_id:
                existing_ids.add(prod_id)
        except json.JSONDecodeError as e:
            print(f"Erreur de décodage JSON : {str(e)}")
            continue
        except Exception as e:
            print(f"Erreur inattendue : {str(e)}")
            continue
    
    print("\nIDs existants trouvés :", sorted(list(existing_ids)))
    
    # Trouver le prochain ID disponible
    next_id = 1
    while f"produit_{next_id:03d}" in existing_ids:
        next_id += 1
    
    return f"produit_{next_id:03d}"

def verifier_doublons(nouveaux_produits, vector_store):
    """
    Vérifie les doublons dans la base existante.
    Retourne la liste des produits sans doublons et la liste des doublons trouvés.
    """
    print(f"\nVérification des doublons pour {len(nouveaux_produits)} nouveaux produits...")
    
    # Récupérer tous les documents existants
    stats = index.describe_index_stats()
    total_vectors = stats.total_vector_count
    print(f"Nombre total de vecteurs dans la base : {total_vectors}")
    
    if total_vectors > 0:
        # Créer un ensemble des IDs existants
        existing_ids = set()
        print("\nRécupération des IDs existants...")
        
        # Récupérer tous les documents en une seule fois
        results = vector_store.similarity_search_with_score("", k=total_vectors)
        for doc, _ in results:
            try:
                content = doc.page_content.strip()
                # Nettoyer le contenu JSON
                content = content.replace('\n', ' ').replace('\r', '')
                content = ' '.join(content.split())
                # Parser le JSON
                json_content = json.loads(content)
                prod_id = json_content.get("id")
                if prod_id:
                    existing_ids.add(prod_id)
                    print(f"ID existant trouvé : {prod_id}")
            except json.JSONDecodeError as e:
                print(f"Erreur de décodage JSON : {str(e)}")
                continue
            except Exception as e:
                print(f"Erreur inattendue : {str(e)}")
                continue
        
        print(f"\nNombre d'IDs existants trouvés : {len(existing_ids)}")
        print("Liste des IDs existants :", sorted(list(existing_ids)))
    
        # Vérifier les doublons et mettre à jour les IDs si nécessaire
        produits_uniques = []
        doublons = []
        
        print("\nVérification des nouveaux produits...")
        for produit in nouveaux_produits:
            try:
                content = produit.page_content.strip()
                # Nettoyer le contenu JSON
                content = content.replace('\n', ' ').replace('\r', '')
                content = ' '.join(content.split())
                # Parser le JSON
                json_content = json.loads(content)
                prod_id = json_content.get("id")
                
                if prod_id:
                    if prod_id in existing_ids:
                        print(f"Doublon trouvé : {prod_id}")
                        # Générer un nouvel ID pour ce produit
                        new_id = get_next_available_id(vector_store, total_vectors)
                        print(f"Attribution d'un nouvel ID : {new_id}")
                        # Mettre à jour l'ID dans le contenu JSON
                        json_content["id"] = new_id
                        # Mettre à jour le contenu du document
                        produit.page_content = json.dumps(json_content, ensure_ascii=False, indent=2)
                        produits_uniques.append(produit)
                    else:
                        print(f"Nouveau produit unique : {prod_id}")
                        produits_uniques.append(produit)
                else:
                    print(f"Attention : Pas d'ID trouvé dans le produit")
                    # Générer un nouvel ID pour ce produit
                    new_id = get_next_available_id(vector_store, total_vectors)
                    print(f"Attribution d'un nouvel ID : {new_id}")
                    # Ajouter l'ID au contenu JSON
                    json_content["id"] = new_id
                    # Mettre à jour le contenu du document
                    produit.page_content = json.dumps(json_content, ensure_ascii=False, indent=2)
                    produits_uniques.append(produit)
            except json.JSONDecodeError as e:
                print(f"Erreur de décodage JSON : {str(e)}")
                print(f"Contenu problématique : {content[:200]}...")
                produits_uniques.append(produit)
            except Exception as e:
                print(f"Erreur inattendue : {str(e)}")
                produits_uniques.append(produit)
                
        print(f"\nRésumé de la vérification :")
        print(f"- Produits uniques trouvés : {len(produits_uniques)}")
        print(f"- Doublons trouvés : {len(doublons)}")
        
        return produits_uniques, doublons
    
    print("\nAucun vecteur existant dans la base.")
    return nouveaux_produits, []

# Définir vos nouveaux produits
nouveaux_produits = [
    Document(
        page_content=json.dumps({
  "id": "produit_024",
  "nom": "Film machine avec groupe pré-étirable Power & Super Power",
  "categorie": "Emballages logistiques",
  "type": "Film étirable CAST machine (pré-étirage motorisé)",
  "description": "Films CAST Power (220 %) et Super Power (330 %) pour banderoleuses équipées d’un groupe de pré-étirage à frein motorisé. Leur forte capacité d’étirement permet de réduire la consommation de plastique tout en maintenant une excellente stabilisation des charges.",
  "avantages": [
    "Très haut taux de pré-étirage : 220 % (Power) et 330 % (Super Power)",
    "Économie de matériau : jusqu’à 25 % de réduction des coûts d’emballage",
    "Stabilité de palette optimale avec tension constante",
    "Haute résistance à la déchirure malgré l’épaisseur réduite",
    "Option de personnalisation (couleur, face glissante, UV, antistatique, impression…)",
    "Solution plus écologique : moins de plastique par palette"
  ],
  "utilisations": [
    "Palettisation automatique avec banderoleuse pré-étirage motorisé",
    "Fixation de charges moyennes à lourdes pour transport et stockage",
    "Réduction de l’empreinte plastique tout en garantissant la sécurité des produits"
  ],
  "caracteristiques_techniques": {
    "Film": "CAST",
    "Versions": {
      "Power": {
        "Épaisseur": "12 µm",
        "Laize": "500 mm",
        "Longueur": "2500 m",
        "Taux_étirabilité": "220 %",
        "Taux_étirabilité_rupture": "500 %"
      },
      "Super Power": {
        "Épaisseur": "15 µm",
        "Laize": "500 mm",
        "Longueur": "1800 m",
        "Taux_étirabilité": "330 %",
        "Taux_étirabilité_rupture": "600 %"
      }
    },
    "Poids_bobine": "16 kg (mandrin 1,8 kg Ø 76 mm)",
    "Compatibilité": "Banderoleuse avec groupe de pré-étirage à frein motorisé",
    "Options_personnalisation": [
      "Couleur",
      "Face glissante",
      "Traitement UV",
      "Antistatique",
      "Basses températures",
      "Impression personnalisée"
    ],
    "Équivalences_standard": {
      "Power 12 µm": "Remplace film standard 20 µm",
      "Super Power 15 µm": "Remplace film standard 23 µm"
    }
  },
  "dimensions_disponibles": [
    { "laize": 500, "longueur": 2500, "epaisseur": 12, "unite_laize": "mm", "unite_longueur": "m", "unite_epaisseur": "µm", "version": "Power" },
    { "laize": 500, "longueur": 1800, "epaisseur": 15, "unite_laize": "mm", "unite_longueur": "m", "unite_epaisseur": "µm", "version": "Super Power" }
  ],
  "couleurs_disponibles": ["Transparent", "Blanc"],
  "disponibilite": "En stock – autres variantes sur devis",
  "prix_indicatif": None,
  "images": [
    "https://www.butterflypackaging.com/wp-content/uploads/2023/03/film-machine-power-superpower.png"
  ],
  "mots_cles": [
    "film pré-étirable Power",
    "film Super Power",
    "film CAST haut pré-étirage",
    "palettisation automatique",
    "réduction plastique",
    "film palette haute performance"
  ],

}
, ensure_ascii=False, indent=2),
        metadata={"source": "https://www.butterflypackaging.com/produit/film-machine-power-superpower"}
    )
]

print(f"\nDébut du processus d'ajout de {len(nouveaux_produits)} produits...")

# Vérifier les doublons
produits_a_ajouter, doublons = verifier_doublons(nouveaux_produits, vector_store)

if doublons:
    print(f"\nAttention ! Les produits suivants existent déjà et ne seront pas ajoutés :")
    for doublon in doublons:
        print(f"- {doublon}")

if produits_a_ajouter:
    # Récupérer le nombre total de vecteurs
    stats = index.describe_index_stats()
    total_vectors = stats.total_vector_count
    print(f"Nombre total de vecteurs dans l'index : {total_vectors}")
    
    # Générer des IDs uniques pour chaque produit
    uuids = []
    for produit in produits_a_ajouter:
        try:
            # Nettoyer le contenu JSON
            content = produit.page_content.strip()
            content = content.replace('\n', ' ').replace('\r', '')
            content = ' '.join(content.split())
            # Parser le JSON
            json_content = json.loads(content)
            uuids.append(json_content["id"])
        except json.JSONDecodeError as e:
            print(f"Erreur de décodage JSON lors de la génération d'ID : {str(e)}")
            # Si erreur de décodage, générer un nouvel ID
            new_id = get_next_available_id(vector_store, total_vectors)
            uuids.append(new_id)
            # Mettre à jour le contenu avec le nouvel ID
            try:
                # Recréer le contenu JSON proprement
                json_content = {
                    "id": new_id,
                    "nom": "Film manuel Polytech",
                    "categorie": "Emballages logistiques",
                    "type": "Film étirable CAST rigide manuel",
                    "description": "Film étirable CAST rigide Polytech, multicouches, plus résistant et rigide que les films étirables standards. Il garantit une excellente stabilité des charges avec moins de tours, réduisant ainsi la consommation de plastique. Le Polytech 9 µ remplace par exemple un film standard 20 µ tout en maintenant une forte force de serrage.",
                    "avantages": [
                        "Haute résistance à la déchirure et à la perforation",
                        "Rigidité élevée : meilleure rétention et immobilisation des charges",
                        "Moins de tours nécessaires → économie de film et de temps",
                        "Réduction de la consommation de plastique (ex. 9 µ = standard 20 µ)",
                        "Polyéthylène multicouches transparent (visibilité des marchandises)",
                        "Personnalisable : UV, antistatique, couleur, bord renforcé, impression"
                    ],
                    "utilisations": [
                        "Palettisation manuelle (≤ 10 palettes/jour) de charges lourdes et volumineuses",
                        "Stabilisation de palettes pour transport et stockage longue durée",
                        "Réduction de poids d'emballage tout en conservant la tension de serrage",
                        "Applications logistiques nécessitant une forte rigidité du film"
                    ],
                    "caracteristiques_techniques": {
                        "Film": "CAST rigide Polytech (polyéthylène multicouches)",
                        "Épaisseurs_disponibles": ["9 µm", "12 µm", "15 µm"],
                        "Laize": "450 mm",
                        "Longueur": "270 m",
                        "Poids_bobine": "≈ 2 kg (mandrin 0,180 kg Ø 50 mm)",
                        "Taux_étirabilité": "90 % → 110 %",
                        "Taux_étirabilité_rupture": "≈ 310 %",
                        "Face_collante": "Intérieure uniquement",
                        "Options_personnalisation": [
                            "Face glissante",
                            "Traitement UV (6 / 12 mois)",
                            "Antistatique",
                            "Basses températures",
                            "Impression personnalisée",
                            "Sans mandrin",
                            "Bord renforcé",
                            "Coloré"
                        ],
                        "Équivalences_standard": {
                            "Polytech 9 µm": "Remplace film standard 20 µm",
                            "Polytech 12 µm": "Remplace film standard 23 µm",
                            "Polytech 15 µm": "Remplace film standard 30 µm"
                        },
                        "Recommandation_palette": {
                            "≤ 250 kg": "Polytech 9 µm",
                            "250 – 500 kg": "Polytech 12 µm",
                            "500 – 750 kg": "Polytech 15 µm",
                            "> 750 kg": "Épaisseur sur devis"
                        },
                        "Compatibilité_machine": "Film manuel – non livré avec dévidoir / banderoleuse"
                    },
                    "dimensions_disponibles": [
                        { "laize": 450, "longueur": 270, "unite_laize": "mm", "unite_longueur": "m" }
                    ],
                    "couleurs_disponibles": ["Transparent"],
                    "disponibilite": "En stock – autres variantes sur devis",
                    "prix_indicatif": None,
                    "images": [
                        "https://www.butterflypackaging.com/wp-content/uploads/2023/03/film-manuel-polytech.png"
                    ],
                    "mots_cles": [
                        "film étirable rigide",
                        "film Polytech",
                        "palettisation manuelle",
                        "film CAST",
                        "polyéthylène multicouches",
                        "réduction plastique",
                        "film palette haute résistance"
                    ]
                }
                produit.page_content = json.dumps(json_content, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Erreur lors de la mise à jour du contenu : {str(e)}")
    
    print(f"IDs générés : {uuids}")

    # Ajouter les nouveaux produits à la base
    print(f"\nAjout de {len(produits_a_ajouter)} nouveaux produits...")
    vector_store.add_documents(documents=produits_a_ajouter, ids=uuids)
    print(f"\n{len(produits_a_ajouter)} nouveaux produits ajoutés avec succès !")
    print(f"IDs générés : {uuids}")
else:
    print("\nAucun nouveau produit à ajouter.") 