import streamlit as st
import os
import requests
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Moodie AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (MODERN DARK UI) ---
st.markdown("""
<style>
    /* Background & Main Colors */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    
    /* Chat Message Bubbles */
    .user-msg {
        background-color: #238636;
        color: white;
        padding: 10px 15px;
        border-radius: 15px 15px 0 15px;
        margin-bottom: 10px;
        text-align: right;
        display: inline-block;
        max-width: 80%;
        float: right;
    }
    .bot-msg {
        background-color: #1F242C;
        border: 1px solid #30363D;
        color: #E6EDF3;
        padding: 10px 15px;
        border-radius: 15px 15px 15px 0;
        margin-bottom: 10px;
        display: inline-block;
        max-width: 80%;
    }
    
    /* Movie Card Styling */
    .movie-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        transition: transform 0.2s;
    }
    .movie-card:hover {
        border-color: #58A6FF;
        transform: translateY(-2px);
    }
    .movie-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #58A6FF;
        margin-bottom: 5px;
    }
    .movie-meta {
        font-size: 0.85rem;
        color: #8B949E;
        margin-bottom: 10px;
    }
    .movie-plot {
        font-size: 0.9rem;
        color: #C9D1D9;
        margin-bottom: 10px;
    }
    
    /* Link Button */
    .stream-btn {
        display: inline-block;
        padding: 5px 10px;
        background-color: #238636;
        color: white !important;
        text-decoration: none;
        border-radius: 5px;
        font-size: 0.8rem;
        margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA MODELS ---
class MoodAnalysis(BaseModel):
    detected_moods: List[str] = Field(description="List of detected emotions")
    summary: str = Field(description="Short empathetic summary")
    recommended_genres: List[str] = Field(description="Genres suitable for mood")
    search_keywords: str = Field(description="English paragraph describing movie PLOT")

# --- BACKEND LOGIC (CACHED) ---
@st.cache_resource
def load_vector_store():
    # Pastikan folder chroma_db ada di root folder
    if not os.path.exists("./chroma_db"):
        st.error("❌ Database tidak ditemukan! Pastikan folder 'chroma_db' sudah di-upload.")
        return None
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_db",
        collection_name="tmdb_movies"
    )
    return vector_store

def get_gemini_chain(api_key):
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest", 
        temperature=0.7, 
        google_api_key=api_key
    )
    parser = JsonOutputParser(pydantic_object=MoodAnalysis)
    prompt = PromptTemplate(
        template="""
        You are an empathetic Movie Therapist named Moodie.
        
        HISTORY: {chat_history}
        USER INPUT: "{user_input}"
        
        Analyze the input and provide a JSON response with:
        1. detected_moods (list of strings)
        2. summary (empathetic response to user)
        3. recommended_genres (list)
        4. search_keywords (a plot description for semantic search)
        
        {format_instructions}
        """,
        input_variables=["user_input", "chat_history"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    return prompt | llm | parser

def get_movie_details(movie_id, tmdb_key):
    """Simple detail fetcher without too much overhead"""
    if not tmdb_key: return {"runtime": "N/A", "poster": None}
    
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={tmdb_key}"
        data = requests.get(url).json()
        poster_path = data.get('poster_path')
        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
        return {"runtime": f"{data.get('runtime', 0)} min", "poster": poster_url, "vote": data.get('vote_average', 0)}
    except:
        return {"runtime": "N/A", "poster": None, "vote": 0}

# --- SIDEBAR UI ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2503/2503508.png", width=50) # Placeholder logo
    st.title("Moodie AI")
    st.markdown("Your Personal Movie Therapist")
    
    st.divider()
    
    # API KEY MANAGEMENT
    # Di Cloud, kita pakai st.secrets. Di lokal, bisa input manual.
    if "GOOGLE_API_KEY" in st.secrets:
        gemini_key = st.secrets["GOOGLE_API_KEY"]
    else:
        gemini_key = st.text_input("Gemini API Key", type="password")
        
    if "TMDB_API_KEY" in st.secrets:
        tmdb_key = st.secrets["TMDB_API_KEY"]
    else:
        tmdb_key = st.text_input("TMDB API Key", type="password")

    st.divider()
    st.markdown("### 🎭 Quick Moods")
    cols = st.columns(3)
    if cols[0].button("😢 Sad"): st.session_state.preset_input = "I'm feeling very sad and need a cry."
    if cols[1].button("😡 Angry"): st.session_state.preset_input = "I am so angry right now!"
    if cols[2].button("😴 Bored"): st.session_state.preset_input = "I'm bored, surprise me."

# --- MAIN UI ---
st.title("Apa yang kamu rasakan hari ini?")
st.markdown("_Ceritakan harimu, dan aku akan pilihkan film yang pas._")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    role_class = "user-msg" if message["role"] == "user" else "bot-msg"
    with st.container():
        if message["role"] == "user":
            st.markdown(f'<div style="text-align: right;"><div class="{role_class}">{message["content"]}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="{role_class}">{message["content"]}</div>', unsafe_allow_html=True)
            # Jika ada rekomendasi film, render cards
            if "movies" in message:
                cols = st.columns(len(message["movies"]))
                for idx, movie in enumerate(message["movies"]):
                    with cols[idx]:
                        st.markdown(f"""
                        <div class="movie-card">
                            <img src="{movie['poster']}" style="width:100%; border-radius:5px; margin-bottom:10px;">
                            <div class="movie-title">{movie['title']}</div>
                            <div class="movie-meta">⭐ {movie['rating']} | ⏳ {movie['runtime']}</div>
                            <div class="movie-plot">{movie['plot'][:100]}...</div>
                            <a href="https://www.google.com/search?q=watch+{movie['title'].replace(' ', '+')}" target="_blank" class="stream-btn">▶ Cari Streaming</a>
                        </div>
                        """, unsafe_allow_html=True)

# Handle Input (Text Input or Button Preset)
user_input = st.chat_input("Ketik perasaanmu di sini...")
if "preset_input" in st.session_state and st.session_state.preset_input:
    user_input = st.session_state.preset_input
    st.session_state.preset_input = None # Reset

if user_input and gemini_key:
    # 1. Simpan input user
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.rerun() # Refresh untuk menampilkan pesan user dulu

# Logic processing (dijalankan setelah rerun)
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_user_msg = st.session_state.messages[-1]["content"]
    
    with st.spinner("🤖 Moodie sedang menganalisis perasaanmu..."):
        try:
            # Load DB & Chain
            vector_store = load_vector_store()
            chain = get_gemini_chain(gemini_key)
            
            # History Context
            history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-3:]])
            
            # Analyze
            analysis = chain.invoke({"user_input": last_user_msg, "chat_history": history_text})
            
            # Search Movies
            docs = vector_store.similarity_search(analysis['search_keywords'], k=3)
            
            movie_results = []
            for doc in docs:
                meta = doc.metadata
                details = get_movie_details(meta['id'], tmdb_key)
                movie_results.append({
                    "title": meta['title'],
                    "rating": f"{meta.get('rating', 'N/A'):.1f}",
                    "year": meta.get('year', ''),
                    "runtime": details['runtime'],
                    "poster": details['poster'] or "https://via.placeholder.com/300x450?text=No+Poster",
                    "plot": doc.page_content.split("Plot: ")[-1]
                })
            
            # Simpan respons bot
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"{analysis['summary']}\n\n*Mood terdeteksi: {', '.join(analysis['detected_moods'])}*",
                "movies": movie_results
            })
            st.rerun()
            
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")