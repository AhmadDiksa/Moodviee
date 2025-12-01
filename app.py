import streamlit as st
import os
import requests
from typing import List

# --- IMPORTS UTAMA ---
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Moodie AI", page_icon="🎬", layout="wide")

# --- API KEYS (STATIC / SECRETS) ---
# Ganti dengan key asli untuk testing lokal, atau gunakan st.secrets saat deploy
TMDB_API_KEY = st.secrets.get("TMDB_API_KEY", "")
GOOGLE_SEARCH_KEY = st.secrets.get("SEARCH_KEY", "")
GOOGLE_CX = st.secrets.get("SEARCH_CX", "")

# --- CUSTOM CSS (DARK MODE ELEGANT & MIRIP GAMBAR) ---
st.markdown("""
<style>
    /* 1. GLOBAL DARK THEME */
    .stApp { background-color: #0E1117; color: #E6EDF3; }
    
    /* 2. CHAT BUBBLES */
    /* User: Hijau gelap elegan (Kanan) */
    .user-msg { 
        background: linear-gradient(135deg, #238636 0%, #2EA043 100%);
        color: white; 
        padding: 12px 18px; 
        border-radius: 18px 18px 0 18px; 
        margin-bottom: 10px; 
        text-align: right; 
        float: right; 
        max-width: 70%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    
    /* Bot: Abu-abu gelap (Kiri) */
    .bot-msg { 
        background-color: #21262D; 
        border: 1px solid #30363D; 
        color: #C9D1D9; 
        padding: 12px 18px; 
        border-radius: 18px 18px 18px 0; 
        margin-bottom: 10px; 
        display: inline-block; 
        max-width: 80%; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* 3. MOVIE CARD DI CHAT (MIRIP GAMBAR 1) */
    .chat-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 15px;
        margin-top: 10px;
        display: flex;
        gap: 15px;
        align-items: start;
        transition: transform 0.2s;
    }
    .chat-card:hover {
        border-color: #58A6FF;
        transform: translateY(-2px);
    }
    .card-poster {
        width: 80px;
        border-radius: 8px;
        object-fit: cover;
    }
    .card-content { flex: 1; }
    .card-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 0;
    }
    .card-meta {
        font-size: 0.8rem;
        color: #8B949E;
        margin-bottom: 8px;
    }
    .card-plot {
        font-size: 0.85rem;
        color: #C9D1D9;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    
    /* 4. TOMBOL LIHAT DETAIL */
    /* Kita styling tombol native streamlit agar fit di card */
    div.stButton > button {
        background-color: #238636;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.25rem 1rem;
        font-size: 0.9rem;
        width: 100%;
        margin-top: 10px;
        transition: background-color 0.2s;
    }
    div.stButton > button:hover {
        background-color: #2ea043;
        color: white;
    }
    
    /* 5. MODAL / DIALOG STYLING (MIRIP GAMBAR 2) */
    .detail-header {
        font-size: 2rem;
        font-weight: 800;
        color: #FFFFFF;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    .detail-tag {
        background-color: #30363D;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        color: #E6EDF3;
        margin-right: 5px;
    }
    .streaming-badge {
        display: inline-block;
        background-color: #0D1117;
        border: 1px solid #30363D;
        padding: 8px 12px;
        border-radius: 8px;
        margin-right: 10px;
        margin-bottom: 5px;
        color: white;
        text-decoration: none;
        font-size: 0.9rem;
    }
    .streaming-badge:hover { border-color: #58A6FF; color: #58A6FF; }
    
</style>
""", unsafe_allow_html=True)

# --- CLASS DATA ---
class MoodAnalysis(BaseModel):
    detected_moods: List[str] = Field(description="List of detected emotions")
    summary: str = Field(description="Short empathetic summary")
    search_keywords: str = Field(description="English paragraph describing movie PLOT")

# --- BACKEND LOGIC ---
@st.cache_resource
def load_db():
    if not os.path.exists("./chroma_db"): return None
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(embedding_function=embeddings, persist_directory="./chroma_db", collection_name="tmdb_movies")

@st.cache_resource
def get_chain(api_key):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7, google_api_key=api_key)
    parser = JsonOutputParser(pydantic_object=MoodAnalysis)
    prompt = PromptTemplate(
        template="""
        You are an empathetic Movie Therapist.
        HISTORY: {chat_history}
        INPUT: "{user_input}"
        
        Analyze mood and create a movie plot search query.
        {format_instructions}
        """,
        input_variables=["user_input", "chat_history"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    return prompt | llm | parser

def get_details(movie_id):
    """Mengambil Detail Film + Gambar BACKDROP untuk Modal"""
    if not TMDB_API_KEY: 
        return {
            "runtime": "N/A", "review": "No API Key", "trailer": None,
            "poster": "https://via.placeholder.com/150", "backdrop": "https://via.placeholder.com/800x400",
            "genres": [], "vote": 0
        }
    
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&append_to_response=videos,reviews"
        data = requests.get(url).json()
        
        # Images
        poster = f"https://image.tmdb.org/t/p/w500{data.get('poster_path')}" if data.get('poster_path') else "https://via.placeholder.com/150"
        backdrop = f"https://image.tmdb.org/t/p/original{data.get('backdrop_path')}" if data.get('backdrop_path') else poster

        # Meta
        genres = [g['name'] for g in data.get('genres', [])]
        vote = data.get('vote_average', 0)
        
        # Trailer & Review
        videos = data.get("videos", {}).get("results", [])
        trailer = next((f"https://www.youtube.com/watch?v={v['key']}" for v in videos if v["type"]=="Trailer"), None)
        
        reviews = data.get("reviews", {}).get("results", [])
        snippet = reviews[0]['content'][:300] + "..." if reviews else "Belum ada review populer."
        
        return {
            "runtime": f"{data.get('runtime', 0)} min", 
            "review": snippet, 
            "trailer": trailer,
            "poster": poster,
            "backdrop": backdrop,
            "genres": genres,
            "vote": round(vote, 1),
            "overview": data.get("overview", "")
        }
    except:
        return {"runtime": "N/A", "review": "Error", "trailer": None, "poster": "", "backdrop": "", "genres": [], "vote": 0}

def search_links(title):
    if not GOOGLE_SEARCH_KEY: return []
    try:
        query = f'watch "{title}" movie streaming'
        url = "https://www.googleapis.com/customsearch/v1"
        params = {"key": GOOGLE_SEARCH_KEY, "cx": GOOGLE_CX, "q": query, "num": 5}
        res = requests.get(url, params=params).json()
        links = []
        allowed = ["netflix", "disneyplus", "hotstar", "vidio", "primevideo", "hbo", "apple", "viu"]
        if "items" in res:
            for item in res["items"]:
                if any(p in item.get("link", "") for p in allowed):
                    links.append({"title": item["title"], "link": item["link"]})
        return links[:3]
    except: return []

# --- FITUR MODAL / POP-UP (MENGGUNAKAN ST.DIALOG) ---
@st.dialog("🎬 Detail Film", width="large")
def show_movie_details(movie):
    # Header Image (Backdrop)
    st.image(movie['backdrop'], use_container_width=True)
    
    # Judul & Rating (Mirip Gambar 2)
    col_head1, col_head2 = st.columns([3, 1])
    with col_head1:
        st.markdown(f"<div class='detail-header'>{movie['title']} <span style='font-size:1.2rem; font-weight:400; color:#8B949E;'>({movie['year']})</span></div>", unsafe_allow_html=True)
        # Genre Tags
        tags = "".join([f"<span class='detail-tag'>{g}</span>" for g in movie['genres']])
        st.markdown(f"<div style='margin-top:10px; margin-bottom:15px;'>{tags}</div>", unsafe_allow_html=True)
    
    with col_head2:
        # Mood Effect Circle (Simulasi Gambar 2)
        st.metric(label="Mood Match", value=f"{int(movie['rating']*10)}%", delta="Recommended")

    # Layout Konten (Trailer | Plot & Info)
    c1, c2 = st.columns([1.5, 1])
    
    with c1:
        st.markdown("### 🎥 Trailer")
        if movie['trailer']:
            st.video(movie['trailer'])
        else:
            st.info("Trailer tidak tersedia.")
            
        st.markdown("### 💬 Review Komunitas")
        st.markdown(f"""
        <div style="background-color:#161B22; padding:15px; border-radius:10px; border:1px solid #30363D;">
            <div style="font-style:italic; color:#C9D1D9;">"{movie['review']}"</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("### 📝 Sinopsis")
        st.write(movie['overview'])
        
        st.divider()
        st.markdown("### 📺 Streaming Availability")
        
        if movie['links']:
            for link in movie['links']:
                # Ikon Platform Sederhana
                icon = "🌐"
                if "netflix" in link['link']: icon = "🔴 Netflix"
                elif "disney" in link['link']: icon = "🔵 Disney+"
                elif "hbo" in link['link']: icon = "🟣 HBO"
                elif "vidio" in link['link']: icon = "⚪ Vidio"
                
                st.markdown(f"<a href='{link['link']}' target='_blank' class='streaming-badge'>{icon}</a>", unsafe_allow_html=True)
        else:
            st.caption("Link streaming legal tidak ditemukan secara otomatis.")

# --- SIDEBAR UI ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3220/3220743.png", width=60)
    st.title("Moodie AI")
    st.markdown("Your Personal Movie Therapist")
    
    st.divider()
    gemini_key = st.text_input("🔑 Gemini API Key", type="password")
    
    st.markdown("### 🎭 Quick Mood")
    col_moods = st.columns(3)
    if col_moods[0].button("😭"): st.session_state.preset = "Aku sedih banget, butuh nangis."
    if col_moods[1].button("😡"): st.session_state.preset = "Lagi marah nih!"
    if col_moods[2].button("😍"): st.session_state.preset = "Lagi jatuh cinta."

# --- MAIN CHAT LOGIC ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Handle Preset Click
if "preset" in st.session_state:
    user_input_preset = st.session_state.preset
    del st.session_state.preset
else:
    user_input_preset = None

# Render Chat History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">{msg["content"]}</div><div style="clear:both;"></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">{msg["content"]}</div>', unsafe_allow_html=True)
        
        # RENDER MOVIE CARD JIKA ADA
        if "movies" in msg:
            cols = st.columns(len(msg["movies"])) # Grid layout untuk card
            for idx, mov in enumerate(msg["movies"]):
                with cols[idx]:
                    # 1. Render Tampilan Card (HTML/CSS)
                    st.markdown(f"""
                    <div class="chat-card">
                        <img src="{mov['poster']}" class="card-poster">
                        <div class="card-content">
                            <div class="card-title">{mov['title']}</div>
                            <div class="card-meta">{mov['year']} • ⭐ {mov['rating']}</div>
                            <div class="card-plot">{mov['plot'][:60]}...</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 2. Tombol "Lihat Detail" (Native Streamlit agar interaktif)
                    # Tombol diletakkan di luar div HTML agar event handlernya jalan
                    if st.button("👁️ Detail Film", key=f"btn_{mov['title']}_{idx}"):
                        show_movie_details(mov)

# Input Processing
input_text = st.chat_input("Ceritakan perasaanmu...")
final_input = user_input_preset if user_input_preset else input_text

if final_input and gemini_key:
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": final_input})
    st.rerun() # Rerun untuk menampilkan bubble user

# Logic AI (Jalan setelah rerun)
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with st.spinner("🔮 Moodie sedang berpikir..."):
        try:
            db = load_db()
            chain = get_chain(gemini_key)
            
            history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-3:] if m['role'] != 'system'])
            last_msg = st.session_state.messages[-1]["content"]
            
            # Analisis
            analysis = chain.invoke({"user_input": last_msg, "chat_history": history})
            
            # Search
            docs = db.similarity_search(analysis['search_keywords'], k=3)
            
            movies_found = []
            for doc in docs:
                meta = doc.metadata
                details = get_details(meta['id'])
                links = search_links(meta['title'])
                
                # Gabungkan semua data
                movie_data = {
                    "title": meta['title'],
                    "year": meta['year'],
                    "rating": meta['rating'],
                    "plot": doc.page_content.split("Plot: ")[-1],
                    "overview": details['overview'], # Full plot untuk modal
                    "runtime": details['runtime'],
                    "review": details['review'],
                    "trailer": details['trailer'],
                    "poster": details['poster'],
                    "backdrop": details['backdrop'],
                    "genres": details['genres'],
                    "links": links
                }
                movies_found.append(movie_data)
            
            # Simpan Respons Bot
            st.session_state.messages.append({
                "role": "assistant", 
                "content": analysis['summary'],
                "movies": movies_found
            })
            st.rerun()
            
        except Exception as e:
            st.error(f"Error: {e}")