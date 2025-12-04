import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Spam SMS Detector",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. CSS DARK PURPLE THEME + ANIMASI KONSISTEN ---
st.markdown("""
<style>
    /* Import Font: Orbitron (Judul) & Rajdhani (Body) */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@400;500;700&display=swap');

    :root {
        /* Palette Ungu Gelap */
        --bg-deep: #090014;       /* Background Utama */
        --bg-card: #150525;       /* Background Kartu */
        --neon-purple: #bc13fe;   /* Neon Ungu */
        --neon-pink: #ff007f;     /* Neon Pink */
        --neon-blue: #4d4dff;     /* Neon Biru */
        --text-bright: #ffffff;   /* Putih Terang */
        --text-dim: #d1b3ff;      /* Ungu Pucat */
        --border-color: #5e1675;  /* Garis Tepi */
    }
    
    /* --- ANIMASI FADE IN UP (STANDAR UNTUK SEMUA) --- */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px); /* Jarak luncur */
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* --- ANIMASI PULSE (Hanya untuk Alert Merah) --- */
    @keyframes pulseGlow {
        0% { box-shadow: 0 0 5px var(--neon-pink); }
        50% { box-shadow: 0 0 20px var(--neon-pink); }
        100% { box-shadow: 0 0 5px var(--neon-pink); }
    }

    /* KELAS ANIMASI UMUM (Terapkan ini ke elemen yang ingin animasi masuk) */
    .animate-enter, .cyber-hero, .cyber-card, .metric-cyber, .hud-box, .custom-alert {
        animation: fadeInUp 0.6s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
    }

    /* --- GLOBAL STYLES --- */
    .stApp {
        background-color: var(--bg-deep);
        background-image: 
            linear-gradient(rgba(138, 43, 226, 0.05) 1px, transparent 1px),
            linear-gradient(90deg, rgba(138, 43, 226, 0.05) 1px, transparent 1px);
        background-size: 40px 40px;
        font-family: 'Rajdhani', sans-serif;
        color: var(--text-bright);
        font-size: 1.1rem;
    }

    /* Header Fix */
    .block-container { padding-top: 1.5rem !important; padding-bottom: 5rem !important; }
    header[data-testid="stHeader"] { background: transparent !important; }
    header[data-testid="stHeader"] button, header[data-testid="stHeader"] svg, div[data-testid="stToolbar"] button, div[data-testid="stToolbar"] svg {
        color: #ffffff !important; fill: #ffffff !important; stroke: #ffffff !important;
    }
    div[data-testid="stDecoration"] { display: none; }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Orbitron', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: var(--text-bright) !important;
    }

    /* --- HERO SECTION --- */
    .cyber-hero {
        text-align: center;
        padding: 4rem 1rem;
        border: 2px solid var(--border-color);
        background: radial-gradient(circle, rgba(60, 10, 80, 0.8) 0%, rgba(9, 0, 20, 0.95) 100%);
        box-shadow: 0 0 25px rgba(188, 19, 254, 0.2);
        margin-bottom: 3rem;
        border-radius: 15px;
    }
    .cyber-title {
        font-size: 4rem; font-weight: 900; color: #fff;
        text-shadow: 0 0 10px #fff, 0 0 20px var(--neon-purple), 0 0 40px var(--neon-purple);
        margin-bottom: 0.5rem;
    }
    .cyber-subtitle { color: var(--text-dim); font-size: 1.3rem; letter-spacing: 2px; font-weight: 500; }

    /* --- BUTTONS --- */
    .stButton > button {
        background: linear-gradient(45deg, #2e004d, #4b0082);
        color: var(--text-bright);
        border: 1px solid var(--neon-purple);
        border-radius: 5px;
        padding: 0.8rem 2.5rem;
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        box-shadow: 0 0 10px rgba(188, 19, 254, 0.3);
        transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    .stButton > button:hover {
        background: var(--neon-purple);
        color: #fff;
        border-color: #fff;
        box-shadow: 0 0 30px var(--neon-purple);
        transform: scale(1.02) translateY(-2px);
    }

    /* --- INPUT TEXT AREA --- */
    .stTextArea textarea {
        background-color: rgba(30, 10, 50, 0.8) !important;
        border: 2px solid var(--border-color) !important;
        color: #ffffff !important;
        font-family: 'Source Code Pro', monospace;
        font-size: 1rem;
        border-radius: 8px !important;
        transition: border-color 0.3s, box-shadow 0.3s;
    }
    .stTextArea textarea:focus {
        border-color: var(--neon-purple) !important;
        box-shadow: 0 0 15px rgba(188, 19, 254, 0.4) !important;
        background-color: rgba(45, 15, 70, 0.9) !important;
    }
    .stTextArea label { color: var(--neon-purple) !important; font-family: 'Orbitron', sans-serif; font-size: 1rem; }

    /* --- RESULT BOXES --- */
    .hud-box {
        border: 2px solid; padding: 2rem; text-align: center;
        background: rgba(15, 5, 25, 0.9); position: relative; margin-top: 1rem;
        box-shadow: 0 0 20px rgba(0,0,0,0.5);
    }
    .hud-box::before { content: ''; position: absolute; top: -2px; left: -2px; width: 20px; height: 20px; border-top: 3px solid var(--text-bright); border-left: 3px solid var(--text-bright); }
    .hud-box::after { content: ''; position: absolute; bottom: -2px; right: -2px; width: 20px; height: 20px; border-bottom: 3px solid var(--text-bright); border-right: 3px solid var(--text-bright); }

    .spam-alert { border-color: var(--neon-pink); box-shadow: inset 0 0 30px rgba(255, 0, 127, 0.2); animation: pulseGlow 2s infinite; }
    .spam-text { color: var(--neon-pink); text-shadow: 0 0 15px var(--neon-pink); font-size: 1.8rem; font-weight: bold; margin-bottom: 10px; }

    .safe-alert { border-color: var(--neon-blue); box-shadow: inset 0 0 30px rgba(77, 77, 255, 0.2); }
    .safe-text { color: #4d4dff; text-shadow: 0 0 15px #4d4dff; font-size: 1.8rem; font-weight: bold; margin-bottom: 10px; }
    .result-desc { color: #e0e0e0; font-size: 1.1rem; }

    /* --- FEATURE CARDS --- */
    .cyber-card {
        border: 1px solid var(--border-color);
        background: linear-gradient(135deg, rgba(21, 5, 37, 0.9), rgba(10, 0, 20, 0.9));
        padding: 25px; text-align: center; border-radius: 10px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .cyber-card:hover {
        border-color: #ffffff;
        box-shadow: 0 0 25px rgba(188, 19, 254, 0.5);
        transform: translateY(-8px) scale(1.02);
    }
    .cyber-card h3 { color: #ffffff !important; margin-top: 10px; text-shadow: 0 0 10px rgba(255, 255, 255, 0.6); }
    .cyber-card p { color: #ffffff !important; opacity: 0.9; font-weight: 400; }
    .cyber-card div { color: #ffffff !important; text-shadow: 0 0 15px rgba(255, 255, 255, 0.8); }

    /* --- SIDEBAR --- */
    section[data-testid="stSidebar"] { background-color: #05000a; border-right: 1px solid var(--border-color); }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: var(--neon-purple) !important; }
    
    /* --- CUSTOM ALERT (STATIC CENTER) --- */
    .custom-alert {
        background-color: rgba(255, 0, 0, 0.2);
        border: 1px solid #ff3333;
        color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 10px;
        box-shadow: 0 0 15px rgba(255, 0, 0, 0.3);
        animation: fadeInUp 0.3s ease-out; /* Animasi muncul sekali */
        font-family: 'Orbitron', sans-serif;
        font-size: 0.9rem;
        letter-spacing: 1px;
    }

</style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if 'page' not in st.session_state:
    st.session_state['page'] = 'landing'

def navigate_to(page):
    st.session_state['page'] = page
    st.rerun()

# --- BACKEND LOGIC ---
current_dir = os.path.dirname(os.path.abspath(__file__))
files_in_dir = os.listdir(current_dir)

def try_read_csv(file_path):
    for encoding in ['utf-8', 'latin-1', 'cp1252', 'ISO-8859-1']:
        try:
            df = pd.read_csv(file_path, encoding=encoding, sep=',', quotechar='"')
            if len(df.columns) > 1: return df
            df = pd.read_csv(file_path, encoding=encoding, sep=';', quotechar='"')
            if len(df.columns) > 1: return df
        except: continue
    return None

def try_read_uploaded_file(uploaded_file):
    uploaded_file.seek(0)
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        for sep in [',', ';']:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=encoding, sep=sep, quotechar='"')
                if len(df.columns) >= 2: return df
            except: continue
    return None

def train_model_from_df(df):
    try:
        df.columns = df.columns.str.strip()
        col_kategori = next((c for c in df.columns if 'kategori' in c.lower() or 'label' in c.lower()), None)
        col_pesan = next((c for c in df.columns if 'pesan' in c.lower() or 'text' in c.lower() or 'sms' in c.lower()), None)

        if not col_kategori or not col_pesan: return None, None, None, None, "Format kolom salah."

        df = df.rename(columns={col_kategori: 'Kategori', col_pesan: 'Pesan'})
        df['Kategori'] = df['Kategori'].astype(str).str.lower().str.strip()
        df['Label_Angka'] = df['Kategori'].map({'spam': 1, 'ham': 0})
        df = df.dropna(subset=['Label_Angka', 'Pesan'])
        
        if df.empty: return None, None, None, None, "Dataset kosong."
        
        X_vectorized = CountVectorizer().fit_transform(df['Pesan'])
        X_train, X_test, y_train, y_test = train_test_split(X_vectorized, df['Label_Angka'], test_size=0.2, random_state=42)
        
        model = MultinomialNB()
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        
        return model, CountVectorizer().fit(df['Pesan']), acc, df, None
    except Exception as e: return None, None, None, None, str(e)

@st.cache_data
def load_local_data():
    target_name = 'sms_spam_indo.csv'
    target_path = os.path.join(current_dir, target_name)
    if os.path.exists(target_path): return try_read_csv(target_path), "DB UTAMA AKTIF"
    
    double_path = os.path.join(current_dir, 'sms_spam_indo.csv.csv')
    if os.path.exists(double_path): return try_read_csv(double_path), "DB (EXT GANDA)"
    
    csvs = [f for f in files_in_dir if f.endswith('.csv')]
    if csvs: return try_read_csv(os.path.join(current_dir, csvs[0])), f"DB ALT: {csvs[0]}"
    return None, "DATABASE HILANG"

# --- 5. LANDING PAGE ---
def show_landing_page():
    st.write("")
    
    # Hero Section
    st.markdown("""
        <div class="cyber-hero">
            <h1 class="cyber-title">SPAM SMS DETECTOR</h1>
            <p class="cyber-subtitle">SISTEM DETEKSI SMS SPAM BERBAHASA INDONESIA</p>
        </div>
    """, unsafe_allow_html=True)
    
    # CTA Button
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        if st.button("AKTIFKAN SISTEM", use_container_width=True):
            navigate_to('app')

    # Features
    st.write("")
    st.write("")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("""
        <div class="cyber-card">
            <div style="font-size:2rem; margin-bottom:10px;">⚡</div>
            <h3>SUPER CEPAT</h3>
            <p>Analisis kilat dengan mesin pemrosesan berkecepatan tinggi.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="cyber-card">
            <div style="font-size:2rem; margin-bottom:10px;">🧠</div>
            <h3>AI CERDAS</h3>
            <p>Dilatih menggunakan ribuan data pesan asli Indonesia.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="cyber-card">
            <div style="font-size:2rem; margin-bottom:10px;">🔒</div>
            <h3>PRIVASI AMAN</h3>
            <p>Data diproses secara lokal, tidak dikirim ke server luar.</p>
        </div>
        """, unsafe_allow_html=True)

# --- 6. MAIN APP ---
def show_main_app():
    # Sidebar
    with st.sidebar:
        st.markdown("### MENU SISTEM")
        if st.button("AKHIRI SESI", use_container_width=True):
            navigate_to('landing')
        
        st.markdown("---")
        
        # Load Data
        df, status_msg = load_local_data()
        
        if df is None:
            st.error("DB ERROR")
            uploaded = st.file_uploader("UPLOAD SOURCE", type=['csv'])
            if uploaded:
                df = try_read_uploaded_file(uploaded)
                if df is None: st.stop()
            else: st.stop()

        # Train
        model, cv, acc, df_clean, err = train_model_from_df(df)
        if err: st.error(err); st.stop()
        
        st.markdown(f"""
        <div style="border:1px solid #bc13fe; padding:15px; text-align:center; background: rgba(20,0,40,0.5); border-radius: 8px;">
            <div style="font-size:0.9rem; color:#bc13fe; font-weight:bold;">AKURASI AI</div>
            <div style="font-size:2rem; color:#fff; font-weight:bold;">{acc*100:.1f}%</div>
        </div>
        <div style="margin-top:10px; font-size:0.8rem; color:#aaa; text-align:center;">{status_msg}</div>
        """, unsafe_allow_html=True)

    # Main Interface (DIBUNGKUS KELAS ANIMASI)
    st.markdown("""
        <div class="animate-enter">
            <h2 style='text-align:center; color:#fff; margin-bottom: 30px;'>KONSOL DETEKSI ANCAMAN</h2>
        </div>
    """, unsafe_allow_html=True)
    
    col_left, col_right = st.columns([3, 2], gap="large")
    
    with col_left:
        # Wrapper Animasi untuk Label Input
        st.markdown("""
            <div class="animate-enter">
                <div style='color:#bc13fe; margin-bottom:10px; font-weight:bold; letter-spacing:1px;'>INPUT PESAN MENCURIGAKAN</div>
            </div>
        """, unsafe_allow_html=True)
        
        input_sms = st.text_area(
            "Label Input", 
            height=200, 
            placeholder=">_ Tempel pesan SMS di sini...",
            label_visibility="collapsed"
        )
        
        # --- POSISI ALERT DI TENGAH KOLOM ---
        alert_placeholder = st.empty()
        
        if st.button("JALANKAN SCAN", type="primary", use_container_width=True):
            if not input_sms:
                # Custom Alert (Static, bukan Toast) - Animasi sama dengan entry
                alert_placeholder.markdown("""
                <div class="custom-alert">
                    ⚠️ HARAP MASUKKAN PESAN TERLEBIH DAHULU!
                </div>
                """, unsafe_allow_html=True)
            else:
                alert_placeholder.empty()
                with st.spinner("MEMINDAI POLA TEXT..."):
                    time.sleep(0.5)
                    vec = cv.transform([input_sms])
                    pred = model.predict(vec)
                    st.session_state['last_result'] = pred[0]

    with col_right:
        # Wrapper Animasi untuk Label Hasil
        st.markdown("""
            <div class="animate-enter">
                <div style='color:#bc13fe; margin-bottom:10px; font-weight:bold; letter-spacing:1px;'>HASIL DIAGNOSA</div>
            </div>
        """, unsafe_allow_html=True)
        
        if 'last_result' in st.session_state:
            res = st.session_state['last_result']
            
            if res == 1:
                st.markdown("""
                <div class="hud-box spam-alert">
                    <div class="spam-text">SPAM TERDETEKSI</div>
                    <p class="result-desc" style="color:#ff99cc;">Tingkat Ancaman: TINGGI</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="hud-box safe-alert">
                    <div class="safe-text">PESAN AMAN</div>
                    <p class="result-desc" style="color:#ccccff;">Tidak ditemukan ancaman.</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="animate-enter" style="border:1px dashed #5e1675; padding:3rem; text-align:center; color:#d1b3ff; font-size:1rem; background: rgba(20,0,40,0.3);">
                MENUNGGU DATA INPUT...
            </div>
            """, unsafe_allow_html=True)

# --- 7. EXECUTION ---
if st.session_state['page'] == 'landing':
    show_landing_page()
elif st.session_state['page'] == 'app':
    show_main_app()