import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

# Import your existing RAG logic from retriever.py
from retriever import route_query_with_groq, hybrid_search

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Critical Error: GROQ_API_KEY environment variable missing.")

# Initialize Groq client
generator_client = Groq(api_key=GROQ_API_KEY)

# Initialize FastAPI and the UI Templates
app = FastAPI(title="Rural Advisor API")
templates = Jinja2Templates(directory="templates")

# Define what data the frontend will send to the backend
class ChatRequest(BaseModel):
    query: str
    history: list = []
    language: str = "English"  # <-- New language parameter from the UI

def translate_to_english(query: str, language: str) -> str:
    """Instantly translates regional queries to English for database searching."""
    if language == "English":
        return query
        
    prompt = f"Translate the following {language} text to English. Return ONLY the English translation, no extra words, notes, or explanations.\n\nText: {query}"
    
    try:
        response = generator_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, # 0.0 for strict translation accuracy
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return query # Fallback to original query if translation fails

def generate_api_response(user_query, retrieved_contexts, chat_history, language):
    """Formats the retrieved data and sends it to the LLM for final synthesis."""
    context_str = ""
    for idx, doc in enumerate(retrieved_contexts):
        context_str += f"\n--- CHUNK {idx+1} ---\nScheme: {doc['metadata']['scheme_name']}\nSection: {doc['metadata']['section_type']}\nContent:\n{doc['content']}\n"

    system_prompt = f"""
    You are an expert, empathetic, and highly logical Rural Welfare Scheme Advisor for the Government of India.
    Your task is to analyze the user's situation and provide actionable advice based ONLY on the verified text chunks provided below.

    CRITICAL EXECUTION RULES:
    1. STRICT LANGUAGE RULE: You MUST write your entire final response strictly in {language}. Do not use English unless the user requested English.
    2. ZERO-NOISE FILTERING: Evaluate the user's demographics against the rules. If they do not qualify, completely IGNORE that scheme. 
    3. FACTUAL STRICTNESS: Ground your answer strictly in the provided text.
    4. OUT OF DOMAIN REJECTION (CRITICAL): If the user asks about a topic (e.g., passports, movies, general knowledge) that is NOT covered in the provided chunks, you MUST refuse to answer. Say exactly: "I am specifically designed to assist with Indian Rural Welfare Schemes. I do not have information regarding that topic in my verified database." NEVER provide external knowledge.
    5. BEST MATCH HIGHLIGHTING: Present matches directly using clean markdown (bolding for numbers, bullet points).
    6. PROFESSIONAL TONE: Speak directly as a human assistant. NEVER mention "database", "retrieved chunks", or "system data".

    Verified Database Context:
    {context_str}
    """

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history[-6:]) # Keep the last 6 messages for token efficiency
    messages.append({"role": "user", "content": user_query})

    try:
        response = generator_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.3,
            max_tokens=1024 # Slightly higher to accommodate longer Hindi/Telugu text
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"System generation error: {str(e)}"

# --- ENDPOINTS ---

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request})

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # 0. The Translation Sandwich: Convert regional query to English
        english_query = translate_to_english(request.query, request.language)
        
        # 1. Stage 1 Router (Uses the English translation)
        targeted_schemes = route_query_with_groq(english_query)
        
        # 2. Stage 2 Hybrid Search (Uses the English translation)
        best_chunks = hybrid_search(english_query, targeted_schemes, top_k=3)
        
        # 3. Stage 3 Generation (Takes English chunks, outputs in user's selected language)
        answer = generate_api_response(request.query, best_chunks, request.history, request.language)
        
        return {"answer": answer}
    except Exception as e:
        return {"answer": f"An error occurred processing your request: {str(e)}"}