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

# --- STATIC CONFIG (KUNCI YANG DISPLIT DARI UI) ---
# Kamu bisa menaruh kunci asli di sini (Hardcode) ATAU di st.secrets (Recommended)
# Agar aman saat deploy, sebaiknya gunakan st.secrets di Streamlit Cloud.
# Jika testing lokal, kamu bisa ganti string kosong "" di bawah dengan API Key aslimu.

TMDB_API_KEY = st.secrets.get("TMDB_API_KEY", "MASUKKAN_KEY_TMDB_DISINI_JIKA_LOKAL")
GOOGLE_SEARCH_KEY = st.secrets.get("SEARCH_KEY", "MASUKKAN_KEY_SEARCH_DISINI_JIKA_LOKAL")
GOOGLE_CX = st.secrets.get("SEARCH_CX", "MASUKKAN_ID_CX_DISINI_JIKA_LOKAL")

# --- CUSTOM CSS (DARK MODE ELEGANT) ---
st.markdown("""
<style>
    /* Background & Text */
    .stApp { background-color: #0E1117; color: #E6EDF3; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
    
    /* Chat Bubbles */
    .user-msg { 
        background-color: #238636; color: white; padding: 12px 18px; 
        border-radius: 18px 18px 0 18px; margin-bottom: 10px; 
        text-align: right; float: right; max-width: 80%; 
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .bot-msg { 
        background-color: #1F242C; border: 1px solid #30363D; color: #E6EDF3; 
        padding: 12px 18px; border-radius: 18px 18px 18px 0; margin-bottom: 10px; 
        display: inline-block; max-width: 80%; 
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    /* Movie Card Styling */
    /* GANTI BAGIAN .movie-card DI CSS DENGAN INI */
    .movie-card { 
        display: flex; 
        gap: 20px;
        background-color: #161B22; 
        border: 1px solid #30363D; 
        border-radius: 12px; 
        padding: 20px; 
        margin-bottom: 20px; 
        transition: transform 0.2s; 
    }
    .movie-card:hover { border-color: #A371F7; transform: translateY(-3px); }

    /* Tambahan Class untuk Poster */
    .movie-poster img {
        width: 120px;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    .movie-content {
        flex: 1; /* Sisa ruang untuk teks */
    }
    
    .movie-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
    .movie-title { font-size: 1.3rem; font-weight: 700; color: #A371F7; } /* Ungu Elegant */
    .movie-rating { background-color: #30363D; padding: 4px 8px; border-radius: 4px; font-size: 0.85rem; font-weight: bold; }
    
    .movie-plot { font-size: 0.95rem; color: #C9D1D9; line-height: 1.6; margin-bottom: 15px; }
    .review-box { 
        background-color: #0d1117; padding: 12px; border-left: 3px solid #238636; 
        font-style: italic; color: #8B949E; font-size: 0.9rem; margin-bottom: 15px; border-radius: 0 6px 6px 0;
    }
    
    /* Link Button */
    .stream-link { 
        display: inline-block; background-color: #238636; color: white !important; 
        padding: 8px 15px; border-radius: 20px; text-decoration: none; 
        margin-right: 8px; margin-top: 5px; font-size: 0.85rem; font-weight: 600;
        transition: background-color 0.2s;
    }
    .stream-link:hover { background-color: #2ea043; }
    .trailer-link { color: #58A6FF; text-decoration: none; font-weight: bold; margin-left: 10px; font-size: 0.9rem; }
    
</style>
""", unsafe_allow_html=True)

# --- CLASS DATA (DARI FILE ASLI) ---
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
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.7, google_api_key=api_key)
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
    """Menggunakan TMDB_API_KEY Static"""
    if "MASUKKAN" in TMDB_API_KEY: return {"runtime": "N/A", "review": "API Key TMDB belum diset developer.", "trailer": "#"}
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&append_to_response=videos,reviews"
        data = requests.get(url).json()
        
        # Trailer
        videos = data.get("videos", {}).get("results", [])
        trailer = next((f"https://www.youtube.com/watch?v={v['key']}" for v in videos if v["type"]=="Trailer"), None)
        
        # Review Snippet
        reviews = data.get("reviews", {}).get("results", [])
        snippet = reviews[0]['content'][:250] + "..." if reviews else "Belum ada review populer."
        
        return {"runtime": f"{data.get('runtime', 'N/A')} min", "review": snippet, "trailer": trailer}
    except:
        return {"runtime": "N/A", "review": "Gagal load data.", "trailer": None}

def search_links(title):
    """Menggunakan GOOGLE_SEARCH_KEY & CX Static"""
    if "MASUKKAN" in GOOGLE_SEARCH_KEY or "MASUKKAN" in GOOGLE_CX: return []
    try:
        query = f'watch "{title}" movie streaming'
        url = "https://www.googleapis.com/customsearch/v1"
        params = {"key": GOOGLE_SEARCH_KEY, "cx": GOOGLE_CX, "q": query, "num": 8}
        res = requests.get(url, params=params).json()
        
        links = []
        allowed = ["netflix", "disneyplus", "hotstar", "vidio", "primevideo", "hbo", "apple", "viu", "wetv"]
        
        if "items" in res:
            for item in res["items"]:
                link = item.get("link", "").lower()
                title_res = item.get("title", "").lower()
                
                # Filter Sederhana
                if any(p in link for p in allowed) and title.lower().split("(")[0].strip() in title_res:
                    links.append({"title": item["title"], "link": item["link"]})
        return links[:2] # Ambil max 2 link
    except:
        return []

# --- SIDEBAR (USER INPUT) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3220/3220743.png", width=60)
    st.title("Moodie AI")
    st.markdown("Siapkan popcorn, aku siapin filmnya. 🍿")
    
    st.divider()
    
    # INPUT USER HANYA GEMINI
    gemini_key = st.text_input("🔑 Masukkan Gemini API Key:", type="password", help="Dapatkan gratis di aistudio.google.com")
    
    if not gemini_key:
        st.warning("⚠️ Masukkan Key dulu untuk chat.")
    else:
        st.success("Terhubung!")

    st.divider()
    if st.button("🗑️ Hapus Memori Chat"):
        st.session_state.messages = []
        st.rerun()

# --- MAIN CHAT UI ---
st.title("Lagi ngerasa apa hari ini?")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render Chat
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">{msg["content"]}</div><div style="clear:both;"></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">{msg["content"]}</div>', unsafe_allow_html=True)
        
        # Render Movie Cards jika ada
        if "movies" in msg:
            for mov in msg["movies"]:
                # Tombol Streaming
                stream_btns = ""
                if mov['links']:
                    for link in mov['links']:
                        platform_name = "Link Nonton"
                        if "netflix" in link['link']: platform_name = "Netflix"
                        elif "disney" in link['link']: platform_name = "Disney+"
                        elif "vidio" in link['link']: platform_name = "Vidio"
                        elif "apple" in link['link']: platform_name = "Apple TV"
                        
                        stream_btns += f'<a href="{link["link"]}" target="_blank" class="stream-link">▶ {platform_name}</a>'
                else:
                    stream_btns = '<span style="font-size:0.8rem; color:grey;">(Tidak ada link legal ditemukan)</span>'
                
                trailer_html = f'<a href="{mov["trailer"]}" target="_blank" class="trailer-link">🎬 Trailer</a>' if mov["trailer"] else ""

                st.markdown(f"""
                <div class="movie-card">
                    <div class="movie-header">
                        <div class="movie-title">{mov['title']} <span style="color:#8B949E; font-size:1rem;">({mov['year']})</span></div>
                        <div class="movie-rating">⭐ {mov['rating']}</div>
                    </div>
                    <div class="movie-plot">{mov['plot']}</div>
                    <div style="margin-bottom:10px; font-size:0.9rem; color:#8B949E;">⏳ {mov['runtime']} {trailer_html}</div>
                    <div class="review-box">"{mov['review']}"</div>
                    {stream_btns}
                </div>
                """, unsafe_allow_html=True)

# Input Box
# 1. Cek apakah Gemini Key sudah diisi di Sidebar
if not gemini_key:
    # Jika belum isi Key, tampilkan peringatan dan matikan input
    st.info("⬅️ Silakan masukkan Gemini API Key di menu sebelah kiri untuk memulai chat.")
    # Kita hentikan eksekusi di sini agar tidak error
    st.stop()

# 2. Input Box (Hanya muncul jika Key sudah ada)
user_input = st.chat_input("Ceritakan perasaanmu, aku butuh hiburan...")

# 3. Jika user mengetik sesuatu dan menekan Enter
if user_input:
    # Tampilkan pesan user ke layar
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(f'<div class="user-msg">{user_input}</div>', unsafe_allow_html=True)

    # 4. Proses AI
    with st.chat_message("assistant"):
        with st.spinner("🔍 Menganalisis mood & mencari film..."):
            try:
                # Load DB & Chain
                db = load_db()
                if not db:
                    st.error("Database 'chroma_db' tidak ditemukan! Pastikan foldernya ada.")
                    st.stop()

                chain = get_chain(gemini_key)
                
                # Siapkan History Chat
                history = "\n".join([f"{m['role']}: {m.get('content','')}" for m in st.session_state.messages[-3:]])
                
                # --- TAHAP 1: ANALISIS MOOD ---
                analysis = chain.invoke({"user_input": user_input, "chat_history": history})
                
                # --- TAHAP 2: CARI FILM ---
                docs = db.similarity_search(analysis['search_keywords'], k=3)
                
                movies_found = []
                for doc in docs:
                    meta = doc.metadata
                    # Ambil detail tambahan (TMDB & Streaming)
                    details = get_details(meta['id'])
                    links = search_links(meta['title'])
                    
                    movies_found.append({
                        "title": meta['title'],
                        "year": meta['year'],
                        "rating": meta['rating'],
                        "plot": doc.page_content.split("Plot: ")[-1][:180] + "...",
                        "runtime": details['runtime'],
                        "review": details['review'],
                        "trailer": details['trailer'],
                        "links": links
                    })
                
                # --- TAHAP 3: TAMPILKAN HASIL ---
                
                # Teks Balasan Bot
                st.markdown(f'<div class="bot-msg">{analysis["summary"]}</div>', unsafe_allow_html=True)
                
                # Kartu Film
                for mov in movies_found:
                    # Buat tombol streaming
                    stream_btns = ""
                    if mov['links']:
                        for link in mov['links']:
                            platform_name = "Link Nonton"
                            if "netflix" in link['link']: platform_name = "Netflix"
                            elif "disney" in link['link']: platform_name = "Disney+"
                            elif "vidio" in link['link']: platform_name = "Vidio"
                            elif "apple" in link['link']: platform_name = "Apple TV"
                            elif "prime" in link['link']: platform_name = "Prime Video"
                            elif "hbo" in link['link']: platform_name = "HBO"
                            
                            stream_btns += f'<a href="{link["link"]}" target="_blank" class="stream-link">▶ {platform_name}</a>'
                    else:
                        stream_btns = '<span style="font-size:0.8rem; color:grey;">(Tidak ada link legal ditemukan)</span>'
                    
                    trailer_html = f'<a href="{mov["trailer"]}" target="_blank" class="trailer-link">🎬 Trailer</a>' if mov["trailer"] else ""

                    st.markdown(f"""
                    <div class="movie-card">
                        <div class="movie-header">
                            <div class="movie-title">{mov['title']} <span style="color:#8B949E; font-size:1rem;">({mov['year']})</span></div>
                            <div class="movie-rating">⭐ {mov['rating']}</div>
                        </div>
                        <div class="movie-plot">{mov['plot']}</div>
                        <div style="margin-bottom:10px; font-size:0.9rem; color:#8B949E;">⏳ {mov['runtime']} {trailer_html}</div>
                        <div class="review-box">"{mov['review']}"</div>
                        {stream_btns}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Simpan ke session state agar tidak hilang saat reload
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": analysis['summary'],
                    "movies": movies_found
                })
                
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")