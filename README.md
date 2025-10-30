# 🎬 Context-Aware Multilingual Movie Recommender — Extended

### 🧠 Extended Edition — Multi-language, Context-aware, Explainable, and Production-friendly



---

## 📖 Overview

This project is an **AI-powered movie recommendation system** designed to provide **personalized, multilingual, and context-aware** movie suggestions.  
It enhances the traditional recommendation pipeline by incorporating **user mood, time of day, weather, and preferred language**, enabling a more human-like recommendation experience.

Built using **Streamlit**, **Python**, and **XGBoost**, the system combines **Collaborative Filtering**, **Content-based Similarity**, and **Machine Learning** approaches to deliver explainable and diverse recommendations.

---

## 🚀 Features

- 🌐 **Multilingual Support:** Handles movies in Hindi, Telugu, Malayalam, Kannada, and more.
- 😃 **Context-Aware Recommendations:** Takes user mood, time, and weather into account.
- 🧩 **Multiple Strategies:**
  - Top Rated Movies
  - Content-Based Similarity
- 🧠 **Explainable ML:** (optional) Uses XGBoost for per-movie “like/dislike” predictions.
- 📊 **Interactive Dashboard:** Displays dataset statistics and analytics.
- 🛠️ **Streamlit UI:** Fast, lightweight, and user-friendly web interface.

---

## 🏗️ Project Structure

Recommendator_Movies/
│
├── app.py # Main Streamlit Application
├── generate_movies.py # Generates or loads multilingual movie dataset
├── generate_ratings.py # Simulates or processes user ratings
├── movies.dat # Movie metadata (movieId, title, genres, language)
├── ratings.dat # User ratings (userId, movieId, rating, timestamp)
├── users.dat # User profiles (userId, gender, age, occupation, location)
├── requirements.txt # Required Python dependencies
└── README.md # Project documentation

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/Recommendator_Movies.git
cd Recommendator_Movies
2️⃣ Create a virtual environment
python3 -m venv venv
source venv/bin/activate     # For Mac/Linux
venv\Scripts\activate        # For Windows
3️⃣ Install dependencies
pip install -r requirements.txt
4️⃣ Run the app
streamlit run app.py
🧩 How It Works
Dataset Loading:
Reads movie, user, and rating data (movies.dat, ratings.dat, users.dat).
Feature Engineering:
Extracts genres, encodes categorical features like language, mood, time, and weather.
Training:
Uses XGBoost for binary classification (like/dislike) based on contextual features.
Recommendation:
Top Rated: Ranks by average rating.
Content-Based: Uses genre and language similarity.
Visualization:
Displays data insights — total movies, ratings, users, and average rating via charts.
📊 Example Dashboard
Dataset Overview Section:
Total Movies: 60,000
Total Ratings: 595,938
Unique Users: 5,000
Avg Rating: 3.46
Charts:
Movies per Language
Ratings Distribution
Top Genres by Popularity
📚 Technologies Used
Category	Tools
Programming	Python 3.x
ML Model	XGBoost
Web Framework	Streamlit
Libraries	pandas, numpy, sklearn, matplotlib, seaborn
Visualization	Plotly / Altair
Data Format	.dat (MovieLens-style structured data)
🧑‍💻 Example Commands
To re-train ML model:
python app.py --train
To launch Streamlit dashboard:
streamlit run app.py
To update data:
python generate_movies.py
python generate_ratings.py
🧠 Future Improvements
Integration with real-world APIs (IMDb, TMDB)
Deep learning embeddings (transformers)
Sentiment-aware review analysis
Enhanced explainability with SHAP
