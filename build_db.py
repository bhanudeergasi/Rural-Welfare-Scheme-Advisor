import os
import json
import chromadb
from chromadb.utils import embedding_functions

CHUNKS_FILE = "data_chunks.json"
DB_DIR = "chroma_db" # This is where our persistent database will live

def build_vector_db():
    if not os.path.exists(CHUNKS_FILE):
        print(f"Error: {CHUNKS_FILE} not found. Please ensure Step 3 is complete.")
        return

    print("Initializing ChromaDB in persistent mode...")
    # This creates a folder on your Windows machine to save the database permanently
    client = chromadb.PersistentClient(path=DB_DIR)
    
    # We use the free, local default embedding model
    sentence_transformer_ef = embedding_functions.DefaultEmbeddingFunction()
    
    collection_name = "rural_schemes"
    
    # Clear out any old versions if you run this script multiple times
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
        
    collection = client.create_collection(
        name=collection_name, 
        embedding_function=sentence_transformer_ef
    )
    
    print("Loading chunks from JSON...")
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
        
    documents = []
    metadatas = []
    ids = []
    
    print(f"Embedding {len(chunks)} chunks. (This may take a minute or two the first time as it downloads the free model)...")
    
    for i, chunk in enumerate(chunks):
        documents.append(chunk["content"])
        
        # ChromaDB requires metadata to be strict strings/numbers (no nulls)
        metadata = {
            "scheme_name": chunk["scheme_name"],
            "section_type": chunk["section_type"],
            "source_url": chunk["source_url"],
            "last_verified_date": chunk["last_verified_date"],
            "state_specific": chunk["state_specific"] if chunk["state_specific"] else "all_india"
        }
        metadatas.append(metadata)
        ids.append(f"{chunk['scheme_name']}_{chunk['section_type']}_{i}")
        
    # Insert everything into the database
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"\nSuccess! Stored {collection.count()} embedded chunks into your local database.")
    print(f"You will now see a new folder named '{DB_DIR}' in your project directory.")

if __name__ == "__main__":
    build_vector_db()