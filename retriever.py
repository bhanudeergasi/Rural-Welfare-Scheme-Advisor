import os
import json
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

# --- Step 1: Configuration & Environment Setup ---
load_dotenv()

# Load Groq API credentials safely from .env file
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL_NAME = "llama-3.1-8b-instant"

if not GROQ_API_KEY:
    raise ValueError("Error: GROQ_API_KEY not found. Please verify your .env file setup.")

# Initialize the Groq Client
llm_client = Groq(api_key=GROQ_API_KEY)

# Local Data Storage Directories
CHUNKS_FILE = "data_chunks.json"
PROFILES_FILE = "scheme_profiles.json"
DB_DIR = "chroma_db"

# Load Core Project Files
with open(PROFILES_FILE, 'r', encoding='utf-8') as f:
    scheme_profiles = json.load(f)

with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
    all_chunks = json.load(f)

# Initialize Persistent Vector Database
db_client = chromadb.PersistentClient(path=DB_DIR)
sentence_transformer_ef = embedding_functions.DefaultEmbeddingFunction()
collection = db_client.get_collection(name="rural_schemes", embedding_function=sentence_transformer_ef)

# --- Step 2: Core Project Retrieval Logic ---

def get_actual_chunk_names(routed_keys, all_chunks):
    """Bridges the gap between short profile keys (mgnrega) and long file names using flexible token matching."""
    unique_chunk_names = list(set(c['scheme_name'] for c in all_chunks))
    actual_targets = []
    
    for key in routed_keys:
        profile_name = scheme_profiles.get(key, {}).get("scheme_name", "").lower()
        key_clean = key.lower().replace("_", "")
        
        for actual in unique_chunk_names:
            actual_clean = actual.lower().replace(" ", "").replace("_", "").replace("-", "")
            
            # Match 1: Abronym/Short key containment
            if key_clean in actual_clean:
                actual_targets.append(actual)
                continue
                
            # Match 2: Shared string token intersections
            profile_tokens = set(profile_name.split())
            actual_tokens = set(actual.lower().split())
            if profile_tokens.intersection(actual_tokens):
                actual_targets.append(actual)
                continue
                
    return list(set(actual_targets)) if actual_targets else unique_chunk_names

def route_query_with_groq(query):
    """Stage 1: Uses Groq (Llama 3) to route the query to specific schemes based on structured profiles."""
    print("\n[Stage 1] Routing query via Groq (Llama 3)...")
    
    prompt = f"""
    You are a highly logical Scheme Routing Assistant for Indian rural welfare.
    Your sole task is to evaluate the user's stated situation against the provided scheme profiles and return the keys of the schemes they might be eligible for.
    
    UNIVERSAL ROUTING RULES:

    1. DEMOGRAPHIC MATCHING: Compare the user's age, gender, and location (if provided) against the target demographic of each scheme. Strictly exclude schemes where the user clearly falls outside the parameters (e.g., age limits or gender-specific schemes).
    2. EDUCATION & SKILL LEVEL: Assess the user's education or profession. Differentiate between schemes meant for unskilled wage labor versus those meant for skilled professionals, graduates, or vocational training. Exclude schemes that contradict the user's education level.
    3. ASSETS & INCOME: Respect strict financial or asset constraints (e.g., land-owning farmers, BPL status). If the user explicitly states facts that disqualify them, exclude those schemes.
    4. ASSETS & OCCUPATION (CRITICAL): If the user explicitly mentions being a "farmer" or owning agricultural "land" / "hectares", you MUST route them to "pm_kisan". 
    5. STRICT EXCLUSION: Do NOT route land-owning farmers to manual wage labor schemes like "mgnrega" unless they explicitly ask for "wage work", "labor", or "employment".
    6. GRACEFUL FALLBACK: If the user provides a very short or vague query, include any scheme they *could* potentially qualify for. It is better to include a scheme if you are unsure than to drop it prematurely.

    Return ONLY a JSON object with a single key "schemes" containing an array of matched scheme keys (e.g., {{"schemes": ["scheme_key_1", "scheme_key_2"]}}).

    User Query: "{query}"

    Scheme Profiles:
    {json.dumps(scheme_profiles, indent=2)}
    """

    try:
        response = llm_client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        result_json = json.loads(response.choices[0].message.content)
        matched_schemes = result_json.get("schemes", [])
        
        valid_keys = [s for s in matched_schemes if s in scheme_profiles.keys()]
        return valid_keys if valid_keys else list(scheme_profiles.keys())
    except Exception as e:
        print(f"Routing failed ({e}). Defaulting to all schemes.")
        return list(scheme_profiles.keys())
    
def hybrid_search(query, target_schemes, top_k=3):
    """Stage 2: Executes Dense + Sparse search, then applies a structural metadata boost before RRF fusion."""
    actual_chunk_names = get_actual_chunk_names(target_schemes, all_chunks)
    print(f"[Stage 2] Executing Section-Prioritized Hybrid Search on: {actual_chunk_names}")

    where_filter = {"scheme_name": {"$in": actual_chunk_names}} if len(actual_chunk_names) > 0 else None

    # --- 1. DENSE RETRIEVAL ---
    dense_results = collection.query(
        query_texts=[query],
        n_results=top_k * 3,  # Pull a slightly wider net for re-ranking
        where=where_filter
    )
    dense_ids = dense_results['ids'][0] if dense_results['ids'] else []

    # --- 2. SPARSE RETRIEVAL ---
    filtered_chunks = []
    chunk_ids = []
    for i, c in enumerate(all_chunks):
        if c['scheme_name'] in actual_chunk_names:
            filtered_chunks.append(c)
            chunk_ids.append(f"{c['scheme_name']}_{c['section_type']}_{i}")

    if not filtered_chunks:
        print("Warning: No matching text chunks found for BM25. Falling back to dense vector lookup only.")
        winning_ids = dense_ids[:top_k]
    else:
        corpus = [c['content'].lower().split() for c in filtered_chunks]
        bm25 = BM25Okapi(corpus)
        tokenized_query = query.lower().split()
        sparse_scores = bm25.get_scores(tokenized_query)

        sparse_ranked = sorted(zip(chunk_ids, sparse_scores), key=lambda x: x[1], reverse=True)
        sparse_ids = [item[0] for item in sparse_ranked[:top_k * 3]]

        # --- 3. SECTION-AWARE RECIPROCAL RANK FUSION (RRF) ---
        rrf_scores = {}
        k_constant = 60 

        # We create a mapping of doc_id to its section type to apply strategic boosts
        id_to_section = {}
        for i, c in enumerate(all_chunks):
            id_to_section[f"{c['scheme_name']}_{c['section_type']}_{i}"] = c['section_type']

        # Determine if the user is asking an explicit FAQ-type question
        faq_keywords = ["faq", "frequently asked", "question", "what if", "how can i transfer", "annexure"]
        is_faq_request = any(kw in query.lower() for kw in faq_keywords)

        # Populate RRF scores from Dense Search
        for rank, doc_id in enumerate(dense_ids):
            section = id_to_section.get(doc_id, "details")
            
            # Apply structural weight modifiers
            boost = 1.0
            if not is_faq_request and section == "frequently_asked_questions":
                boost = 0.4  # De-prioritize FAQs for core structural questions
            elif section in ["eligibility", "details"]:
                boost = 1.5  # Elevate primary informational blocks
                
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (k_constant + rank + 1)) * boost

        # Populate RRF scores from Sparse Search
        for rank, doc_id in enumerate(sparse_ids):
            section = id_to_section.get(doc_id, "details")
            
            boost = 1.0
            if not is_faq_request and section == "frequently_asked_questions":
                boost = 0.4
            elif section in ["eligibility", "details"]:
                boost = 1.5
                
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (k_constant + rank + 1)) * boost

        # Extract top-k winners after penalty/boost adjustments
        winning_ids = [doc_id for doc_id, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]]

    # --- 4. PAYLOAD MAPBACK ---
   
    final_results = []
    if winning_ids:
        chunk_data = collection.get(ids=winning_ids)
        id_to_data = {doc_id: {"content": chunk_data['documents'][idx], "metadata": chunk_data['metadatas'][idx]} 
                      for idx, doc_id in enumerate(chunk_data['ids'])}
                      
        for doc_id in winning_ids:
            final_results.append({
                "id": doc_id,
                "content": id_to_data[doc_id]['content'],
                "metadata": id_to_data[doc_id]['metadata']
            })

    return final_results

# --- Step 3: Interactive Verification Console ---
if __name__ == "__main__":
    print("\n==================================================")
    print("🤖 RURAL ADVISOR HYBRID RETRIEVAL ENVIRONMENT ACTIVE")
    print("==================================================")
    
    while True:
        test_query = input("\nEnter client situation / query (or type 'exit' to quit): ")
        if test_query.lower() in ["exit", "quit", "stop"]:
            print("Shutting down engine. Goodbye!")
            break
            
        if not test_query.strip():
            continue

        # Run Two-Stage Hybrid Processing
        targeted_schemes = route_query_with_groq(test_query)
        best_chunks = hybrid_search(test_query, targeted_schemes, top_k=3)

        print("\n" + "="*50)
        print("🏆 TOP RETRIEVED TEXT CONTEXTS")
        print("="*50)
        for i, chunk in enumerate(best_chunks):
            print(f"\n[Result {i+1}] Scheme Source: {chunk['metadata']['scheme_name']}")
            print(f"Section Component: {chunk['metadata']['section_type']}")
            print("-" * 50)
            print(chunk['content'][:400] + "...\n")