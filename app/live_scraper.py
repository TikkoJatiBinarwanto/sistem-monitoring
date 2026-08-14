"""live_scraper.py — Scraper + Inference pipeline (Tanpa Stopword/Stemming)."""
import json
import os
import time
import streamlit as st
import pandas as pd
import re

# ─────────────────────────────────────────────────────────────────────────────
# GOOGLE PLAY SCRAPER
# ─────────────────────────────────────────────────────────────────────────────
from google_play_scraper import Sort, reviews, search

def search_app_in_playstore(query: str):
    """Mencari aplikasi di Google Play Store berdasarkan query, dengan fallback known apps."""
    try:
        results = list(search(query, lang="id", country="id", n_hits=5))
    except Exception as e:
        st.error(f"Pencarian gagal: {e}")
        return []

    # ── Known apps fallback: pastikan app populer selalu muncul ───────────────
    known = {
        "gemini":         {"appId": "com.google.android.apps.bard",         "title": "Google Gemini"},
        "chatgpt":        {"appId": "com.openai.chatgpt",                    "title": "ChatGPT"},
        "grok":           {"appId": "ai.x.grok",                             "title": "Grok"},
        "deepseek":       {"appId": "com.deepseek.chat",                     "title": "DeepSeek"},
        "claude":         {"appId": "com.anthropic.claude",                  "title": "Claude"},
        "copilot":        {"appId": "com.microsoft.copilot",                 "title": "Microsoft Copilot"},
        "perplexity":     {"appId": "com.perplexity.ai",                     "title": "Perplexity AI"},
        "meta ai":        {"appId": "com.meta.ai",                           "title": "Meta AI"},
        "poe":            {"appId": "com.poe.android",                       "title": "Poe"},
        "bard":           {"appId": "com.google.android.apps.bard",          "title": "Google Gemini"},
    }

    seen = {r["appId"] for r in results}
    q_lower = query.lower().strip()
    for key, info in known.items():
        if key in q_lower and info["appId"] not in seen:
            results.insert(0, info)

    return results

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING — Gensim Native (bukan joblib)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_ml_models():
    """
    Memuat model LDA terbaru (format Gensim native .gensim) dan topic_labels.json.
    Dictionary langsung diambil dari model untuk menjamin konsistensi word ID.
    """
    import gensim.utils
    from gensim.models import LdaModel

    root      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(root, "models")

    # Temukan file .gensim terbaru secara otomatis (tanggal tertinggi)
    gensim_files = sorted(
        [f for f in os.listdir(model_dir) if f.startswith("lda_model") and f.endswith(".gensim")]
    )
    if not gensim_files:
        raise FileNotFoundError(
            "Model LDA (.gensim) tidak ditemukan di folder models/. "
            "Jalankan Notebook 04 terlebih dahulu."
        )

    lda_path  = os.path.join(model_dir, gensim_files[-1])

    # ── Numpy 2.x → 1.x compat ──────────────────────────────────────────────
    # Model saved with numpy 2.x has pickle GLOBAL opcodes for
    # numpy.random._mt19937.MT19937 etc. which don't exist in numpy 1.x.
    # Fix: monkey-patch the module-level __bit_generator_ctor so that when
    # pickle resolves the GLOBAL and calls it with a CLASS object (not str),
    # the class is correctly identified.
    import sys
    import types
    import pickle as _pickle_mod
    import numpy.random._pickle as _nrp
    import numpy.random._common as _np_common

    _FAKE_MODULES = [
        "numpy.random._mt19937", "numpy.random._pcg64",
        "numpy.random._sfc64",   "numpy.random._philox",
        "numpy.random._generator",
    ]
    _saved_modules = {}
    for _mod_name in _FAKE_MODULES:
        if _mod_name not in sys.modules:
            _fake = types.ModuleType(_mod_name)
            _fake.MT19937      = _np_common.MT19937
            _fake.PCG64        = _np_common.PCG64
            _fake.PCG64DXSM    = getattr(_np_common, "PCG64DXSM", _np_common.PCG64)
            _fake.SFC64        = _np_common.SFC64
            _fake.Philox       = getattr(_np_common, "Philox", _np_common.MT19937)
            # Patch __bit_generator_ctor on the fake module so pickle resolves
            # the GLOBAL opcode to THIS patched version (module-level lookup)
            _orig_local = _nrp.__bit_generator_ctor
            def _make_ctor(orig=_orig_local):
                def ctor(bit_generator="MT19937"):
                    if isinstance(bit_generator, type):
                        bit_generator = bit_generator.__name__
                    elif isinstance(bit_generator, str) and "." in bit_generator:
                        bit_generator = bit_generator.rsplit(".", 1)[-1]
                    return orig(bit_generator)
                return ctor
            _fake.__bit_generator_ctor = _make_ctor()
            sys.modules[_mod_name] = _fake
            _saved_modules[_mod_name] = _fake

    # Also patch the canonical module-level ctor (some pickle paths use it directly)
    _orig_ctor = _nrp.__bit_generator_ctor
    def _patched_ctor(bit_generator="MT19937"):
        if isinstance(bit_generator, type):
            bit_generator = bit_generator.__name__
        elif isinstance(bit_generator, str) and "." in bit_generator:
            bit_generator = bit_generator.rsplit(".", 1)[-1]
        return _orig_ctor(bit_generator)
    _nrp.__bit_generator_ctor = _patched_ctor

    # Force pure Python pickle so find_class override is honored
    import gensim.utils as _gu
    _orig_unpickle = _gu.unpickle
    def _pure_unpickle(fname, *args, **kwargs):
        with open(fname, "rb") as f:
            return _pickle_mod.Unpickler(f, encoding="latin1").load()
    _gu.unpickle = _pure_unpickle

    try:
        lda_model = LdaModel.load(lda_path)
    finally:
        _gu.unpickle = _orig_unpickle
        _nrp.__bit_generator_ctor = _orig_ctor
        for _mod_name in _saved_modules:
            sys.modules.pop(_mod_name, None)

    # Gunakan Dictionary yang sudah tertanam di dalam model (WAJIB untuk konsistensi word ID)
    vectorizer = lda_model.id2word

    labels_path = os.path.join(model_dir, "topic_labels.json")
    with open(labels_path, "r", encoding="utf-8") as f:
        topic_labels = json.load(f)

    return vectorizer, lda_model, topic_labels


# ─────────────────────────────────────────────────────────────────────────────
# TEXT CLEANING — Pipeline tanpa Stopword & Stemming
# Pipeline: noise removal → case folding → slang normalization
# ─────────────────────────────────────────────────────────────────────────────
from indoNLP.preprocessing import replace_slang

# Kamus slang manual (pelengkap indoNLP)
slang_dict = {
    # ===== Typos / singkatan umum =====
    'yg': 'yang', 'yh': 'ya', 'krn': 'karena', 'karna': 'karena',
    'krna': 'karena', 'dgn': 'dengan', 'dg': 'dengan', 'dn': 'dan',
    'utk': 'untuk', 'tuk': 'untuk', 'bwt': 'untuk', 'untk': 'untuk',
    'dr': 'dari', 'pd': 'pada', 'tp': 'tapi', 'tpi': 'tapi',
    'jg': 'juga', 'jd': 'jadi', 'blm': 'belum', 'belom': 'belum',
    'msh': 'masih', 'sdh': 'sudah', 'udh': 'sudah', 'udah': 'sudah',
    'dh': 'sudah', 'skrg': 'sekarang', 'skrang': 'sekarang',
    'skrng': 'sekarang', 'sampe': 'sampai', 'ampe': 'sampai',
    'sy': 'saya', 'ak': 'aku', 'gw': 'saya', 'gue': 'saya',
    'lo': 'kamu', 'lu': 'kamu', 'kau': 'kamu',
    'bs': 'bisa', 'bsa': 'bisa', 'hrs': 'harus', 'dlm': 'dalam',
    'sm': 'sama', 'lg': 'lagi', 'td': 'tadi',
    'bbrp': 'beberapa', 'aja': 'saja', 'doang': 'saja',
    'cuma': 'hanya', 'cmn': 'hanya', 'cm': 'hanya',
    'tau': 'tahu', 'gatau': 'tidak_tahu', 'gimana': 'bagaimana',
    'gmn': 'bagaimana', 'knp': 'kenapa', 'knpa': 'kenapa',
    'knapa': 'kenapa', 'emg': 'memang',
    'klo': 'kalau', 'kalo': 'kalau', 'gitu': 'begitu',
    'gt': 'begitu', 'trs': 'terus', 'trus': 'terus',
    'blg': 'bilang', 'blang': 'bilang', 'gini': 'begini',
    'gtw': 'tidak_tahu',
    # ===== Variasi 'tidak' =====
    'gak': 'tidak', 'ga': 'tidak', 'nggak': 'tidak', 'ngga': 'tidak',
    'gk': 'tidak', 'nggk': 'tidak', 'ngk': 'tidak',
    'gada': 'tidak_ada', 'kagak': 'tidak', 'kaga': 'tidak',
    'ndak': 'tidak', 'gx': 'tidak', 'tdk': 'tidak',
    # ===== Kata umum ulasan =====
    'banget': 'sangat', 'bgt': 'sangat', 'bgtu': 'begitu',
    'pdhl': 'padahal', 'pdhal': 'padahal',
    'abis': 'habis', 'abisan': 'habisan',
    'cepet': 'cepat', 'dapet': 'dapat', 'bener': 'benar',
    'beneran': 'benaran',
    'kek': 'seperti', 'kayak': 'seperti', 'kaya': 'seperti',
    'gabisa': 'tidak_bisa', 'gatau': 'tidak_tahu',
    # ===== Colloquial evaluative =====
    'makasih': 'terimakasih', 'thanks': 'terimakasih',
    'parah': 'buruk', 'parahh': 'buruk', 'parahhh': 'buruk',
    'wkwk': 'tertawa', 'wkwkwk': 'tertawa',
    'hehe': 'tertawa', 'haha': 'tertawa',
    'loh': 'lho', 'deh': 'saja', 'sih': 'saja',
    'dong': 'saja', 'kok': 'kenapa', 'kan': 'bukan',
    'nih': 'ini', 'tuh': 'itu', 'nah': 'ini',
    # ===== Slang / typo compound =====
    'baguss': 'bagus', 'bangett': 'sangat',
    'mantapp': 'bagus', 'okee': 'bagus',
    'terimakasihh': 'terimakasih',
}

# Overlay: kata yang tidak di-cover indoNLP (dari scan dataset)
slang_overlay = {
    'gua': 'saya', 'gw': 'saya', 'gue': 'saya',
    'baguss': 'bagus', 'bangett': 'sangat', 'mantapp': 'bagus',
    'okee': 'bagus', 'sangatt': 'sangat', 'bisaa': 'bisa',
    'lagii': 'lagi', 'terimakasihh': 'terimakasih',
    'membantuu': 'membantu', 'membantuuu': 'membantu',
    'sumpahhh': 'sumpah', 'terimakasii': 'terimakasih',
    'pokonyaa': 'pokoknya', 'geminii': 'gemini',
    'vidio': 'video', 'errorr': 'error', 'erorr': 'error',
    'effisien': 'efisien', 'bksa': 'bisa', 'kennapa': 'kenapa',
    'analisi': 'analisis', 'menimbulkana': 'menimbulkan',
    'disuru': 'disuruh',
    'tida': 'tidak', 'byar': 'bayar', 'mambantu': 'membantu',
    'mebantu': 'membantu', 'perbayar': 'berbayar',
    'bangus': 'bagus', 'buagus': 'bagus', 'gbisa': 'bisa',
    'bisah': 'bisa', 'jagan': 'jangan', 'gogle': 'google',
    'goggle': 'google', 'gimini': 'gemini', 'donlod': 'download',
    'reting': 'rating', 'unistal': 'uninstall',
    'langanan': 'langganan', 'amplikasi': 'aplikasi',
    'chatgbt': 'chatgpt', 'chtgpt': 'chatgpt',
    'terimaksi': 'terimakasih', 'detil': 'detail',
    'fahami': 'pahami', 'loding': 'loading',
}


def cleansing_pipeline(text: str) -> str:
    """Pipeline: noise removal → case folding → slang normalization. Tanpa stopword & stemming."""
    if not isinstance(text, str):
        return ""

    # 1. Noise Removal
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@[\w]+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # 1b. Collapse repeated characters (>2x) → max 2x (seruuuuu → seruu)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # 2. Case Folding
    text = text.lower()

    # 3. Tokenisasi & Slang Mapping
    tokens = text.split()
    text_slang = replace_slang(' '.join(tokens))
    tokens = text_slang.split()
    tokens = [slang_dict.get(t, t) for t in tokens]
    tokens = [slang_overlay.get(t, t) for t in tokens]

    return ' '.join(tokens)


# Backwards-compatible alias
def clean_text_id(text: str, stopword=None, stemmer=None) -> str:
    """Alias untuk cleansing_pipeline. Parameter stopword/stemmer diabaikan."""
    return cleansing_pipeline(text)


# ─────────────────────────────────────────────────────────────────────────────
# FETCH & INFERENCE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def fetch_and_process_reviews(app_id: str, app_name: str, limit: int = 50) -> pd.DataFrame:
    """
    Ambil ulasan terbaru dari Google Play, bersihkan, lalu prediksi topiknya
    menggunakan model LDA Gensim yang tersimpan.
    """
    if not app_id or not isinstance(app_id, str):
        st.error("ID Aplikasi tidak valid (kosong). Silakan pilih aplikasi lain.")
        return pd.DataFrame()

    st.info(f"Mengambil {limit} ulasan terbaru dari Google Play untuk **{app_name}**...")

    result, _ = reviews(app_id, lang="id", country="id", sort=Sort.NEWEST, count=limit)

    if not result:
        st.warning("Tidak ada ulasan ditemukan.")
        return pd.DataFrame()

    df_new = pd.DataFrame(result)
    df_mapped = pd.DataFrame({
        "text":       df_new["content"],
        "rating":     df_new["score"],
        "date":       df_new["at"],
        "app_source": str(app_name or "").lower(),
    })

    # ── Preprocessing ────────────────────────────────────────────────────────
    st.info("Membersihkan teks (Normalisasi Slang)...")

    cleaned, progress_bar = [], st.progress(0)
    total = len(df_mapped)
    for i, text in enumerate(df_mapped["text"]):
        cleaned.append(cleansing_pipeline(text))
        progress_bar.progress((i + 1) / total)
    progress_bar.empty()

    df_mapped["text_clean"] = cleaned
    df_mapped = df_mapped[df_mapped["text_clean"].str.strip() != ""].copy()

    if df_mapped.empty:
        st.warning("Semua teks kosong setelah pembersihan.")
        return pd.DataFrame()

    # ── Topic Inference ───────────────────────────────────────────────────────
    st.info("Memprediksi topik menggunakan model LDA...")
    vectorizer, lda_model, topic_labels = load_ml_models()

    # Konversi ke BoW menggunakan Dictionary bawaan model
    corpus = [vectorizer.doc2bow(text.split()) for text in df_mapped["text_clean"]]

    dominant_topics, max_probs, all_probs = [], [], []
    n_topics = lda_model.num_topics

    for bow in corpus:
        dist = dict(lda_model.get_document_topics(bow, minimum_probability=0))
        probs = [dist.get(i, 0.0) for i in range(n_topics)]
        best  = int(max(range(n_topics), key=lambda i: probs[i]))
        dominant_topics.append(best)
        max_probs.append(float(probs[best]))
        all_probs.append(probs)

    df_mapped["dominant_topic"]    = dominant_topics
    df_mapped["topic_probability"] = max_probs
    df_mapped["topic_label"]       = df_mapped["dominant_topic"].astype(str).map(topic_labels)

    # Kolom probabilitas per-topik (berguna untuk analisis lanjutan)
    for i in range(n_topics):
        label = topic_labels.get(str(i), f"topik_{i}").split("&")[0].strip().lower().replace(" ", "_")
        df_mapped[f"prob_{label}"] = [p[i] for p in all_probs]

    st.success(f"✅ Berhasil memproses **{len(df_mapped)}** ulasan.")
    return df_mapped


# ─────────────────────────────────────────────────────────────────────────────
# SAVE TO CSV
# ─────────────────────────────────────────────────────────────────────────────
def save_to_csv(df_new: pd.DataFrame, app_id: str) -> int:
    """
    Append DataFrame baru ke file CSV per-aplikasi.
    Mengembalikan jumlah baris unik yang berhasil ditambahkan.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "data", "final", f"topic_data_{app_id}.csv")

    # Kolom wajib yang harus ada
    base_cols = ["text", "rating", "date", "app_source", "text_clean",
                 "dominant_topic", "topic_probability", "topic_label"]
    extra_cols = [c for c in df_new.columns if c.startswith("prob_")]
    all_cols   = base_cols + extra_cols

    # Pastikan hanya kolom yang ada
    df_new = df_new[[c for c in all_cols if c in df_new.columns]]

    try:
        if os.path.exists(path):
            df_existing = pd.read_csv(path)
            prev_len    = len(df_existing)
        else:
            df_existing = pd.DataFrame()
            prev_len    = 0

        df_combined = pd.concat([df_existing, df_new]).drop_duplicates(
            subset=["text", "date"], keep="last"
        )
        df_combined.to_csv(path, index=False)
        return len(df_combined) - prev_len

    except Exception as e:
        st.error(f"Gagal menyimpan ke CSV: {e}")
        return 0
