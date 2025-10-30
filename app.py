
# -----------------------------------------
# Imports
# -----------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import random
import time
import json
import traceback
from datetime import datetime
from typing import Tuple, Dict, Any, List

# ML & similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb

# Visualization
import plotly.express as px
import plotly.graph_objects as go

# Optional SHAP (explainability)
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# -----------------------------------------
# App configuration
# -----------------------------------------
APP_TITLE = "🎬 Context-Aware Multilingual Movie Recommender — Extended"
APP_ICON = "🎥"
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

# data dir
BASE_DIR = os.path.dirname(__file__) if "__file__" in globals() else "."
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

MOVIES_PATH = os.path.join(DATA_DIR, "movies.dat")
RATINGS_PATH = os.path.join(DATA_DIR, "ratings.dat")
USERS_PATH = os.path.join(DATA_DIR, "users.dat")
MODEL_PATH = os.path.join(DATA_DIR, "xgb_model.json")
ENCODERS_PATH = os.path.join(DATA_DIR, "encoders.json")
LOG_PATH = os.path.join(DATA_DIR, "app.log")

INDIAN_LANGUAGES = ['Hindi', 'Tamil', 'Telugu', 'Malayalam', 'Kannada', 'Bengali', 'Marathi', 'Punjabi']

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# -----------------------------------------
# Logging helper
# -----------------------------------------
def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass
    # also print for debugging
    print(line, end='')

# -----------------------------------------
# Utilities: time buckets & weather (India)
# -----------------------------------------
def time_bucket(hour: int) -> str:
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour < 22:
        return 'Evening'
    else:
        return 'Night'

def indian_weather_by_month(month: int) -> str:
    if month in [12, 1]:
        return 'Cool'
    elif month in [2, 3]:
        return 'Moderate'
    elif month in [4, 5, 6, 7, 8, 9]:
        return 'Hot'
    else:
        return 'Pleasant'

# -----------------------------------------
# Data generation helpers (movies, users, ratings)
# These produce files under data/ if you don't have them
# -----------------------------------------
REGIONAL_TITLES = {
    'Hindi': {
        'prefixes': ['Dil ', 'Pyaar ', 'Ek ', 'Main ', 'Mere ', 'Tum ', 'Zindagi ', 'Ishq ', 'Kuch '],
        'suffixes': ['Safar', 'Kahani', 'Raaz', 'Dastaan', 'Zindagi', 'Mohabbat', 'Ishq', 'Roshni', 'Yaadein']
    },
    'Tamil': {
        'prefixes': ['Thalaiva ', 'Kaadhal ', 'Kannu ', 'Urimai ', 'Thamizh ', 'Nenjam '],
        'suffixes': ['Kavithai', 'Kadhalan', 'Manam', 'Vaazhkai', 'Vaali', 'Thalaivan']
    },
    'Telugu': {
        'prefixes': ['Prema ', 'Naa ', 'Mana ', 'Oka ', 'Iddaru ', 'Nuvvu '],
        'suffixes': ['Katha', 'Lokam', 'Premalo', 'Manasu', 'Gunde', 'Rahasyam']
    },
    'Malayalam': {
        'prefixes': ['Oru ', 'Njan ', 'Ente ', 'Njanum ', 'Aaru ', 'Ee '],
        'suffixes': ['Kadha', 'Premam', 'Jeevitham', 'Sneham', 'Lokam', 'Samayam']
    },
    'Bengali': {
        'prefixes': ['Ek ', 'Amar ', 'Tomar ', 'Prothom ', 'Shesh ', 'Bondhu '],
        'suffixes': ['Bhalobasha', 'Golpo', 'Jibon', 'Raat', 'Prohor', 'Shokal']
    },
    'Marathi': {
        'prefixes': ['Maza ', 'Tumcha ', 'Aamcha ', 'Ekach ', 'Prema ', 'Nav '],
        'suffixes': ['Gaon', 'Prem', 'Ahe', 'Maitri', 'Sathi', 'Khel']
    },
    'Punjabi': {
        'prefixes': ['Mera ', 'Tera ', 'Pyaar ', 'Dil ', 'Zindagi ', 'Yaar '],
        'suffixes': ['Pind', 'Sardar', 'Mutiyaar', 'Pyaar', 'Dil', 'Gabru']
    },
    'Kannada': {
        'prefixes': ['Nanna ', 'Ninna ', 'Prema ', 'Jeeva ', 'Hrudaya ', 'Surya '],
        'suffixes': ['Kathe', 'Preethi', 'Sangama', 'Jeevana', 'Geethe', 'Belaku']
    }
}

GENRES = {
    'Drama': 15, 'Comedy': 12, 'Romance': 12, 'Action': 10, 'Family': 8,
    'Social': 8, 'Musical': 6, 'Thriller': 5, 'Historical': 5, 'Devotional': 4,
    'War': 3, 'Sports': 3, 'Political': 3, 'Mythology': 3, 'Crime': 3,
    'Horror': 2, 'Biographical': 2, 'Folk': 2, 'Art House': 1, 'Documentary': 1
}

FESTIVALS = ['Diwali', 'Holi', 'Pongal', 'Durga Puja', 'Ganesh Chaturthi', 'Onam', 'Baisakhi', 'Navratri', 'Eid', 'Christmas']

def generate_year():
    weights = [1] * 30 + [2] * 20 + [3] * 15 + [4] * 10 + [5] * 5
    years = list(range(1950, 2026))
    w = weights[:len(years)]
    return random.choices(years, weights=w)[0]

def generate_movie_title(language: str) -> str:
    titles = REGIONAL_TITLES.get(language, REGIONAL_TITLES['Hindi'])
    prefix = random.choice(titles['prefixes'])
    suffix = random.choice(titles['suffixes'])
    if random.random() < 0.05:
        festival = random.choice(FESTIVALS)
        return f"{prefix}{festival}"
    if random.random() < 0.03:
        number = random.choice(['100', '786', '420', '99', '3', '7', '21'])
        return f"{prefix}{suffix} {number}"
    return f"{prefix}{suffix}"

def generate_genres() -> str:
    num = random.choices([1,2,3], weights=[20,60,20])[0]
    chosen = random.choices(list(GENRES.keys()), weights=list(GENRES.values()), k=num)
    return '|'.join(sorted(set(chosen)))

def generate_movies_file(total: int = 5000, path: str = MOVIES_PATH):
    """
    Create a movies.dat with many movies across languages.
    """
    log(f"Generating movies.dat with {total} movies...")
    base = {
        'Hindi': 0.34, 'Tamil': 0.15, 'Telugu': 0.15, 'Malayalam': 0.08,
        'Kannada': 0.08, 'Bengali': 0.06, 'Marathi': 0.07, 'Punjabi': 0.07
    }
    counts = {lang: max(1, int(total * pct)) for lang, pct in base.items()}
    movies = []
    mid = 1
    for lang, cnt in counts.items():
        for _ in range(cnt):
            title = generate_movie_title(lang)
            year = generate_year()
            genres = generate_genres()
            full_title = f"{title} ({year}) [{lang}]"
            movies.append(f"{mid}::{full_title}::{genres}")
            mid += 1
    random.shuffle(movies)
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(movies))
    log(f"Saved {len(movies)} movies to {path}")

def generate_users_file(num_users: int = 2000, path: str = USERS_PATH):
    log(f"Generating users.dat with {num_users} users...")
    genders = ['M', 'F']
    occupations = [str(i) for i in range(21)]
    users = []
    for uid in range(1, num_users + 1):
        gender = random.choice(genders)
        age = random.choices([18,22,25,30,35,40,45,50,55], weights=[5,10,30,20,10,8,6,5,6])[0]
        occ = random.choice(occupations)
        zipcode = str(random.randint(10000,99999))
        users.append(f"{uid}::{gender}::{age}::{occ}::{zipcode}")
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(users))
    log(f"Saved users to {path}")

def generate_ratings_file(num_users: int = 2000, num_ratings: int = 100000, movies_path: str = MOVIES_PATH, path: str = RATINGS_PATH):
    log(f"Generating ratings.dat with {num_ratings} ratings...")
    with open(movies_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    movie_ids = [int(l.split("::")[0]) for l in lines]
    now_ts = int(time.time())
    ratings = []
    for _ in range(num_ratings):
        user = random.randint(1, num_users)
        movie = random.choice(movie_ids)
        rating = random.choices([1,2,3,4,5], weights=[5,10,35,35,15])[0]
        delta_days = random.randint(0, 5*365)
        ts = now_ts - delta_days * 24*3600 - random.randint(0, 24*3600)
        ratings.append(f"{user}::{movie}::{rating}::{ts}")
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(ratings))
    log(f"Saved ratings to {path}")

def ensure_data_files(min_movies: int = 100, min_ratings: int = 1000, min_users: int = 100):
    try:
        if not os.path.exists(MOVIES_PATH) or sum(1 for _ in open(MOVIES_PATH, 'r', encoding='utf-8')) < min_movies:
            generate_movies_file(total=5000, path=MOVIES_PATH)
    except Exception as e:
        log(f"Error generating movies.dat: {e}")
    try:
        if not os.path.exists(USERS_PATH) or sum(1 for _ in open(USERS_PATH, 'r', encoding='utf-8')) < min_users:
            generate_users_file(num_users=2000, path=USERS_PATH)
    except Exception as e:
        log(f"Error generating users.dat: {e}")
    try:
        if not os.path.exists(RATINGS_PATH) or sum(1 for _ in open(RATINGS_PATH, 'r', encoding='utf-8')) < min_ratings:
            generate_ratings_file(num_users=2000, num_ratings=100000, movies_path=MOVIES_PATH, path=RATINGS_PATH)
    except Exception as e:
        log(f"Error generating ratings.dat: {e}")

# -----------------------------------------
# Reading helpers (cached)
# -----------------------------------------
@st.cache_data(show_spinner=False)
def read_movies(path: str = MOVIES_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, sep='::', engine='python', names=['movieId','title','genres'], encoding='utf-8')
    df['language'] = df['title'].str.extract(r'\[(.*?)\]', expand=False).fillna('Unknown')
    df['clean_title'] = df['title'].str.replace(r'\s*\(\d{4}\)\s*\[.*?\]', '', regex=True)
    return df

@st.cache_data(show_spinner=False)
def read_ratings(path: str = RATINGS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, sep='::', engine='python', names=['userId','movieId','rating','timestamp'], encoding='utf-8')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['time_of_day'] = df['timestamp'].dt.hour.apply(time_bucket)
    df['weather'] = df['timestamp'].dt.month.apply(indian_weather_by_month)
    return df

@st.cache_data(show_spinner=False)
def read_users(path: str = USERS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, sep='::', engine='python', names=['userId','gender','age','occupation','zipcode'], encoding='utf-8')
    return df

# -----------------------------------------
# Mood mapping and feature preparation
# -----------------------------------------
def create_mood_map(movies_df: pd.DataFrame) -> Dict[str, str]:
    positive = {'love','pyaar','prema','happy','joy','smile','dil','ishq','sukhi','pyaar','pream'}
    negative = {'hate','raaz','dard','duniya','maya','viraha','bewafa','huzoor'}
    mood = {}
    for t in movies_df['clean_title'].unique():
        text = str(t).lower()
        score = 0
        for p in positive:
            if p in text: score += 1
        for n in negative:
            if n in text: score -= 1
        mood[t] = 'Positive' if score>0 else ('Negative' if score<0 else 'Neutral')
    return mood

def prepare_features(ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Merge ratings and movies, create context features, one-hot genres, and encoders
    Returns: X, y, encoders (dict)
    """
    log("Preparing features...")
    merged = ratings_df.merge(movies_df[['movieId','genres','language','clean_title']], on='movieId', how='left')
    # If no location in ratings, create random locations for demo
    if 'location' not in merged.columns:
        cities = ['Mumbai','Delhi','Bangalore','Chennai','Hyderabad','Kolkata']
        merged['location'] = np.random.choice(cities, size=len(merged))
    # mood
    mood_map = create_mood_map(movies_df)
    merged['mood'] = merged['clean_title'].map(mood_map).fillna('Neutral')
    # one-hot time_of_day and weather
    merged = pd.get_dummies(merged, columns=['time_of_day','weather'], prefix=['tod','weather'])
    # genre one-hot
    all_genres = sorted({g for sub in movies_df['genres'].dropna().str.split('|') for g in sub})
    for genre in all_genres:
        merged[f'genre_{genre}'] = merged['genres'].apply(lambda x: int(genre in x.split('|')) if pd.notna(x) else 0)
    # encoders
    le_loc = LabelEncoder()
    merged['location_encoded'] = le_loc.fit_transform(merged['location'].astype(str))
    le_mood = LabelEncoder()
    merged['mood_encoded'] = le_mood.fit_transform(merged['mood'].astype(str))
    le_lang = LabelEncoder()
    merged['language_encoded'] = le_lang.fit_transform(merged['language'].astype(str))
    # label
    merged['rating_label'] = merged['rating'].apply(lambda x: 1 if x>=4 else 0)
    # feature columns
    feature_cols = ['location_encoded','mood_encoded','language_encoded']
    for c in merged.columns:
        if c.startswith('tod_') or c.startswith('weather_'):
            feature_cols.append(c)
    genre_cols = [c for c in merged.columns if c.startswith('genre_')]
    feature_cols += genre_cols
    X = merged[feature_cols].fillna(0)
    y = merged['rating_label']
    encoders = {
        'le_location_classes': le_loc.classes_.tolist(),
        'le_mood_classes': le_mood.classes_.tolist(),
        'le_language_classes': le_lang.classes_.tolist(),
        'feature_cols': feature_cols
    }
    log(f"Feature matrix prepared with {len(feature_cols)} features.")
    return X, y, encoders

# -----------------------------------------
# Training helper (fixed for xgboost versions)
# -----------------------------------------
def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame = None, y_valid: pd.Series = None,
                  params: Dict[str, Any] = None, num_boost_round: int = 200) -> xgb.XGBClassifier:
    """
    Train XGBoost classifier. Handles both old and new XGBoost APIs for early stopping.
    """
    log("Starting XGBoost training...")
    default_params = {
        'n_estimators': num_boost_round,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': SEED,
        'n_jobs': -1
    }
    if params:
        default_params.update(params)
    # Ensure n_estimators aligns with requested rounds
    default_params['n_estimators'] = num_boost_round
    model = xgb.XGBClassifier(**{k:v for k,v in default_params.items() if k != 'eval_metric'})
    # Try old early_stopping parameter first; fallback to callback interface if TypeError occurs
    if X_valid is not None and y_valid is not None:
        try:
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=False,
                early_stopping_rounds=20
            )
        except TypeError:
            # fallback for xgboost >= 2.0
            try:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    verbose=False,
                    callbacks=[xgb.callback.EarlyStopping(rounds=20)]
                )
            except Exception as e:
                log(f"Failed with callback early stopping: {e}")
                # attempt training without early stopping
                model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)
    log("XGBoost training finished.")
    return model

def evaluate_model(model: xgb.XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, 'predict_proba') else None
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    log(f"Model evaluated: acc={acc:.4f}, auc={auc}")
    return {'accuracy': acc, 'auc': auc, 'report': report, 'confusion_matrix': cm}

# -----------------------------------------
# Content matrix & CF helpers
# -----------------------------------------
@st.cache_data(show_spinner=False)
def build_genre_matrix(movies_df: pd.DataFrame) -> pd.DataFrame:
    log("Building content (genre+language) matrix...")
    genre_set = sorted({g for sub in movies_df['genres'].dropna().str.split('|') for g in sub})
    mat = pd.DataFrame(0, index=movies_df.index, columns=genre_set + ['language'])
    for idx, row in movies_df.iterrows():
        for g in (row['genres'] or '').split('|'):
            if g in genre_set:
                mat.at[idx, g] = 1
        mat.at[idx, 'language'] = row['language']
    lang_dummies = pd.get_dummies(mat['language'], prefix='lang')
    mat = mat.drop(columns=['language']).join(lang_dummies)
    log("Content matrix ready.")
    return mat

def get_content_similar_movies(movie_id: int, movies_df: pd.DataFrame, mat: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if movie_id not in movies_df['movieId'].values:
        return pd.DataFrame()
    idx = movies_df.index[movies_df['movieId'] == movie_id][0]
    vecs = mat.values
    sim = cosine_similarity([vecs[idx]], vecs)[0]
    top_idx = np.argsort(-sim)
    top_idx = top_idx[top_idx != idx][:top_n]
    return movies_df.iloc[top_idx][['movieId','title','genres','language']].assign(similarity=sim[top_idx])

@st.cache_data(show_spinner=False)
def build_item_item_matrix(ratings_df: pd.DataFrame, min_ratings: int = 5):
    log("Building item-item collaborative filtering model...")
    pivot = ratings_df.pivot_table(index='movieId', columns='userId', values='rating', fill_value=0)
    counts = (pivot > 0).sum(axis=1)
    pivot = pivot.loc[counts[counts >= min_ratings].index]
    if pivot.shape[0] < 2:
        log("Not enough data for item-item model.")
        return None, None
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(pivot.values)
    log("Item-item model built.")
    return model, pivot

def get_similar_movies_by_cf(movie_id: int, model: NearestNeighbors, pivot: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if model is None or pivot is None or movie_id not in pivot.index:
        return pd.DataFrame()
    idx = list(pivot.index).index(movie_id)
    distances, indices = model.kneighbors(pivot.values[idx:idx+1], n_neighbors=top_n+1)
    inds = indices.flatten()[1:]
    sims = 1 - distances.flatten()[1:]
    similar_ids = pivot.index[inds]
    return pd.DataFrame({'movieId': similar_ids, 'similarity': sims})

# -----------------------------------------
# Start Streamlit UI
# -----------------------------------------
st.title(APP_TITLE)
st.markdown("Extended edition — multi-language, context-aware, explainable, and production-friendly.")

# Ensure data files exist (generate samples if missing)
ensure_data_files(min_movies=100, min_ratings=1000, min_users=100)

# Sidebar: data upload option
st.sidebar.header("📥 Data / Upload")
use_uploaded = st.sidebar.checkbox("Upload local files instead of using stored ones?", value=False)
uploaded_movies = None
uploaded_ratings = None
uploaded_users = None
if use_uploaded:
    uploaded_movies = st.sidebar.file_uploader("Upload movies.dat", type=['dat','txt','csv'])
    uploaded_ratings = st.sidebar.file_uploader("Upload ratings.dat", type=['dat','txt','csv'])
    uploaded_users = st.sidebar.file_uploader("Upload users.dat", type=['dat','txt','csv'])

# Load dataframes
try:
    if use_uploaded and uploaded_movies is not None:
        movies_df = pd.read_csv(io.StringIO(uploaded_movies.getvalue().decode('utf-8')), sep='::', engine='python', names=['movieId','title','genres'])
        movies_df['language'] = movies_df['title'].str.extract(r'\[(.*?)\]').fillna('Unknown')
        movies_df['clean_title'] = movies_df['title'].str.replace(r'\s*\(\d{4}\)\s*\[.*?\]', '', regex=True)
    else:
        movies_df = read_movies(MOVIES_PATH)
    if use_uploaded and uploaded_ratings is not None:
        ratings_df = pd.read_csv(io.StringIO(uploaded_ratings.getvalue().decode('utf-8')), sep='::', engine='python', names=['userId','movieId','rating','timestamp'])
        ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
        ratings_df['time_of_day'] = ratings_df['timestamp'].dt.hour.apply(time_bucket)
        ratings_df['weather'] = ratings_df['timestamp'].dt.month.apply(indian_weather_by_month)
    else:
        ratings_df = read_ratings(RATINGS_PATH)
    if use_uploaded and uploaded_users is not None:
        users_df = pd.read_csv(io.StringIO(uploaded_users.getvalue().decode('utf-8')), sep='::', engine='python', names=['userId','gender','age','occupation','zipcode'])
    else:
        users_df = read_users(USERS_PATH)
    log("Data loaded.")
except Exception as e:
    st.error("Failed to load data. See logs.")
    log(f"Data load error: {e}\n{traceback.format_exc()}")
    st.stop()

# Sidebar: model & UI parameters
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Model & UI Settings")
n_estimators = st.sidebar.slider("XGBoost: n_estimators", 50, 1000, 300, step=50)
max_depth = st.sidebar.slider("XGBoost: max_depth", 3, 12, 6)
learning_rate = st.sidebar.slider("XGBoost: learning_rate", 0.01, 0.5, 0.1, step=0.01)
min_ratings_threshold = st.sidebar.slider("Min ratings per movie (when showing top-rated)", 1, 200, 10)
train_test_split_ratio = st.sidebar.slider("Train/Test split: test size (%)", 10, 50, 20)

available_languages = sorted(movies_df['language'].dropna().unique().tolist())
if not available_languages:
    available_languages = INDIAN_LANGUAGES

# Main tabs
tabs = st.tabs(["📊 Overview","🤖 Train Model","🎯 Recommend","📈 Analytics"])

# ------------------ Overview ------------------
with tabs[0]:
    st.header("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Movies", f"{movies_df['movieId'].nunique():,}")
    c2.metric("Total Ratings", f"{len(ratings_df):,}")
    c3.metric("Unique Users", f"{ratings_df['userId'].nunique():,}")
    c4.metric("Avg Rating", f"{ratings_df['rating'].mean():.2f}")
    st.subheader("Languages")
    lang_counts = movies_df['language'].value_counts().reset_index()
    lang_counts.columns = ['Language','Movies']
    fig = px.bar(lang_counts, x='Language', y='Movies', title="Movies per Language")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Sample Movies")
    st.dataframe(movies_df[['movieId','clean_title','language','genres']].sample(20).reset_index(drop=True), use_container_width=True)

# ------------------ Train Model ------------------
with tabs[1]:
    st.header("Train XGBoost Model (Contextual)")
    st.write("Prepare contextual features (time, weather, mood, genres) and train model to predict if user will 'like' (rating >=4).")
    if st.button("🔁 Prepare Features & Train Model"):
        try:
            X, y, encoders = prepare_features(ratings_df, movies_df)
            feature_cols = encoders['feature_cols']
            test_size = train_test_split_ratio / 100.0
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=SEED, stratify=y)
            params = {'max_depth': max_depth, 'learning_rate': learning_rate}
            model = train_xgboost(X_train, y_train, X_valid=X_test, y_valid=y_test, params=params, num_boost_round=n_estimators)
            eval_res = evaluate_model(model, X_test, y_test)
            st.success(f"Model trained — Accuracy: {eval_res['accuracy']:.3f} — AUC: {eval_res['auc']:.3f}" if eval_res['auc'] else f"Model trained — Accuracy: {eval_res['accuracy']:.3f}")
            cm = eval_res['confusion_matrix']
            fig = px.imshow(cm, labels=dict(x="Predicted", y="Actual"), x=['Not Liked','Liked'], y=['Not Liked','Liked'], title="Confusion Matrix")
            st.plotly_chart(fig, use_container_width=True)
            # Save model & encoders to disk and session
            try:
                model.save_model(MODEL_PATH)
                with open(ENCODERS_PATH, 'w', encoding='utf-8') as f:
                    json.dump(encoders, f)
                st.info(f"Saved model to {MODEL_PATH} and encoders to {ENCODERS_PATH}")
            except Exception as e:
                st.warning(f"Could not save model/encoders: {e}")
            st.session_state['model'] = model
            st.session_state['encoders'] = encoders
            st.session_state['feature_cols'] = feature_cols
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
        except Exception as e:
            st.error("Training failed. See logs.")
            log(f"Training error: {e}\n{traceback.format_exc()}")

    if os.path.exists(MODEL_PATH) and 'model' not in st.session_state:
        if st.button("🔁 Load saved model from disk"):
            try:
                m = xgb.XGBClassifier()
                m.load_model(MODEL_PATH)
                enc = {}
                if os.path.exists(ENCODERS_PATH):
                    enc = json.load(open(ENCODERS_PATH, 'r', encoding='utf-8'))
                st.session_state['model'] = m
                st.session_state['encoders'] = enc
                st.success("Loaded model and encoders into session.")
            except Exception as e:
                st.error("Failed to load model.")
                log(f"Model load error: {e}\n{traceback.format_exc()}")

# ------------------ Recommend ------------------
with tabs[2]:
    st.header("Personalized Recommendations")
    st.write("Select languages (multi-select) and a recommendation strategy.")
    selected_languages = st.multiselect("Select Languages", options=available_languages, default=["Hindi"])
    all_genres = sorted({g for sub in movies_df['genres'].dropna().str.split('|') for g in sub})
    selected_genres = st.multiselect("Select Genres (optional)", options=all_genres)
    times = sorted(ratings_df['time_of_day'].unique().tolist())
    selected_time = st.selectbox("Time of Day", options=times, index=0 if 'Morning' in times else 0)
    weather_opts = sorted(ratings_df['weather'].unique().tolist())
    selected_weather = st.selectbox("Weather", options=weather_opts)
    mood_opts = ['Positive','Neutral','Negative']
    selected_mood = st.selectbox("Mood", options=mood_opts, index=1)
    min_ratings_for_display = st.slider("Minimum #ratings for a movie to appear", 1, 500, min_ratings_threshold)
    strategy = st.radio("Strategy", ['Top Rated', 'Content Similarity',])

    df_lang_filtered = movies_df[movies_df['language'].isin(selected_languages)] if selected_languages else movies_df
    if selected_genres:
        df_lang_filtered = df_lang_filtered[df_lang_filtered['genres'].apply(lambda x: any(g in x for g in selected_genres))]
    if df_lang_filtered.empty:
        st.warning("No movies match your language/genre selection.")
    else:
        if strategy == 'Top Rated':
            st.subheader("Top-Rated (filtered)")
            stats = ratings_df.groupby('movieId').agg(num_ratings=('rating','count'), avg_rating=('rating','mean')).reset_index()
            merged = df_lang_filtered.merge(stats, on='movieId', how='left').fillna({'num_ratings':0,'avg_rating':0})
            filtered = merged[merged['num_ratings'] >= min_ratings_for_display]
            filtered = filtered.sort_values(['avg_rating','num_ratings'], ascending=[False, False]).head(50)
            for _, row in filtered.iterrows():
                st.markdown(f"**{row['clean_title']}** [{row['language']}] — {row['genres']} — ⭐ {row['avg_rating']:.2f} ({int(row['num_ratings'])} reviews)")

        elif strategy == 'Content Similarity':
            st.subheader("Content-based Similarity")
            genre_mat = build_genre_matrix(movies_df)
            seed_choice = st.selectbox("Choose seed movie", options=df_lang_filtered['title'].sample(min(200, len(df_lang_filtered))).tolist())
            if st.button("Find Similar (Content)"):
                try:
                    seed_row = movies_df[movies_df['title'] == seed_choice].iloc[0]
                    recs = get_content_similar_movies(int(seed_row['movieId']), movies_df, genre_mat, top_n=20)
                    if recs.empty:
                        st.warning("No similar movies found.")
                    else:
                        for _, r in recs.iterrows():
                            st.markdown(f"**{r['title']}** — {r['language']} — {r['genres']}")
                except Exception as e:
                    st.error("Error finding content-similar movies.")
                    log(f"Content-sim error: {e}\n{traceback.format_exc()}")

        elif strategy == 'Item-Item CF':
            st.subheader("Item-Item Collaborative Filtering")
            cf_model, pivot = build_item_item_matrix(ratings_df, min_ratings=5)
            movie_for_cf = st.selectbox("Select movie (CF seed)", options=movies_df['title'].sample(min(200, len(movies_df))).tolist())
            if st.button("Find Similar (CF)"):
                try:
                    seed_movie = movies_df[movies_df['title'] == movie_for_cf].iloc[0]
                    sims = get_similar_movies_by_cf(int(seed_movie['movieId']), cf_model, pivot, top_n=20)
                    if sims.empty:
                        st.warning("Not enough data for CF recommendations.")
                    else:
                        sims = sims.merge(movies_df[['movieId','title','language','genres']], on='movieId', how='left')
                        for _, r in sims.iterrows():
                            st.markdown(f"**{r['title']}** — {r['language']} — {r['genres']} — sim {r['similarity']:.3f}")
                except Exception as e:
                    st.error("CF error.")
                    log(f"CF error: {e}\n{traceback.format_exc()}")

        elif strategy == 'ML Prediction (per movie)':
            st.subheader("ML Prediction")
            if 'model' not in st.session_state:
                st.info("No trained model in session. Train in the Train tab first.")
            else:
                movie_choice = st.selectbox("Select movie to analyze", options=df_lang_filtered['title'].sample(min(200, len(df_lang_filtered))).tolist())
    
                selected_mood = st.selectbox("Mood", options=users_df['mood'].unique().tolist())
                selected_time = st.selectbox("Time of Day", options=users_df['time_of_day'].unique().tolist())
                selected_weather = st.selectbox("Weather", options=users_df['weather'].unique().tolist())

                if st.button("Predict Like / Not-Like"):
                    try:
                        enc = st.session_state['encoders']
                        feature_cols = st.session_state['feature_cols']
                        feature_dict = {c:0 for c in feature_cols}
                        # encode location
                        try:
                            le_loc = LabelEncoder()
                            le_loc.classes_ = np.array(enc['le_location_classes'])
                            feature_dict['location_encoded'] = int(le_loc.transform([user_location])[0])
                        except Exception:
                            feature_dict['location_encoded'] = 0
                        try:
                            le_mood = LabelEncoder()
                            le_mood.classes_ = np.array(enc['le_mood_classes'])
                            feature_dict['mood_encoded'] = int(le_mood.transform([selected_mood])[0])
                        except Exception:
                            feature_dict['mood_encoded'] = 0
                        try:
                            le_lang = LabelEncoder()
                            le_lang.classes_ = np.array(enc['le_language_classes'])
                            lang = movies_df[movies_df['title'] == movie_choice]['language'].iloc[0]
                            feature_dict['language_encoded'] = int(le_lang.transform([lang])[0])
                        except Exception:
                            feature_dict['language_encoded'] = 0
                        # time_of_day and weather
                        tod_col = f'tod_{selected_time}'
                        if tod_col in feature_dict:
                            feature_dict[tod_col] = 1
                        weather_col = f'weather_{selected_weather}'
                        if weather_col in feature_dict:
                            feature_dict[weather_col] = 1
                        # set genres
                        movie_genres = movies_df[movies_df['title'] == movie_choice]['genres'].iloc[0]
                        for g in (movie_genres or '').split('|'):
                            col = f'genre_{g}'
                            if col in feature_dict:
                                feature_dict[col] = 1
                        Xp = pd.DataFrame([feature_dict])[feature_cols].fillna(0)
                        model = st.session_state['model']
                        pred = model.predict(Xp)[0]
                        prob = model.predict_proba(Xp)[0] if hasattr(model,'predict_proba') else None
                        if pred == 1:
                            st.success(f"Model predicts you will LIKE '{movie_choice}'" + (f" (Confidence: {prob[1]*100:.1f}%)" if prob is not None else ""))
                        else:
                            st.warning(f"Model predicts you may NOT like '{movie_choice}'" + (f" (Confidence: {prob[0]*100:.1f}%)" if prob is not None else ""))
                    except Exception as e:
                        st.error("Prediction error.")
                        log(f"Prediction error: {e}\n{traceback.format_exc()}")



# ------------------ Analytics ------------------
with tabs[3]:
    st.header("Analytics Dashboard")
    st.subheader("Rating distribution")
    st.plotly_chart(px.histogram(ratings_df, x='rating', nbins=5, title='Rating Distribution'), use_container_width=True)
    st.subheader("Avg rating by language")
    avg_by_lang = ratings_df.merge(movies_df[['movieId','language']], on='movieId').groupby('language')['rating'].mean().reset_index()
    st.plotly_chart(px.bar(avg_by_lang, x='language', y='rating', title='Avg Rating by Language'), use_container_width=True)
    st.subheader("Top movies by avg rating (min reviews)")
    min_rev = st.slider("Min reviews", 1, 500, 20)
    stats = ratings_df.groupby('movieId').agg(num_ratings=('rating','count'), avg_rating=('rating','mean')).reset_index()
    top = stats[stats['num_ratings'] >= min_rev].merge(movies_df[['movieId','clean_title','language']], on='movieId').sort_values(['avg_rating','num_ratings'], ascending=[False,False]).head(50)
    st.dataframe(top[['clean_title','language','avg_rating','num_ratings']])


