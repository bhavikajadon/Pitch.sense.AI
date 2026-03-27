import streamlit as st
import spacy
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="PitchSense AI", page_icon="💡", layout="wide")

# --- 2. CACHING FOR PERFORMANCE ---
# Streamlit re-runs the script on every click. Caching prevents reloading the heavy AI models.
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

@st.cache_data
def load_and_prep_data(file_name="Shark Tank US dataset.csv"):
    try:
        df = pd.read_csv(file_name)
        text_column = None
        for col in df.columns:
            if col.lower() in ['description', 'pitch', 'idea', 'product description', 'details']:
                text_column = col
                break
        if not text_column:
            text_column = df.columns[1]

        df = df.dropna(subset=[text_column])
        corpus = df[text_column].astype(str).tolist()

        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        vectorizer.fit(corpus)
        feature_names = vectorizer.get_feature_names_out()
        return df, text_column, vectorizer, feature_names
    except Exception as e:
        return None, None, None, None

# Load models and data
nlp = load_nlp()
df, text_column, vectorizer, feature_names = load_and_prep_data()

# --- 3. CORE ANALYSIS FUNCTION ---
def analyze_pitch(pitch_text):
    doc = nlp(str(pitch_text))
    funding, equity, startup_name = "Not Found", "Not Found", "Not Found"
    
    for ent in doc.ents:
        if ent.label_ == "MONEY": funding = ent.text
        elif ent.label_ == "PERCENT": equity = ent.text
        elif ent.label_ == "ORG" and startup_name == "Not Found": startup_name = ent.text

    try:
        single_vector = vectorizer.transform([str(pitch_text)]).todense()
        df_tfidf = pd.DataFrame(single_vector, index=["tfidf"], columns=feature_names).T
        top_keywords = df_tfidf.sort_values(by="tfidf", ascending=False).head(3).index.tolist()
    except:
        top_keywords = ["N/A"]

    blob = TextBlob(str(pitch_text))
    sentiment_score = blob.sentiment.polarity
    
    sentiment_label = "Neutral"
    if sentiment_score > 0.1: sentiment_label = "Positive"
    elif sentiment_score < -0.1: sentiment_label = "Negative"

    base_score = 50
    risk = "Medium"
    base_score += (sentiment_score * 20)

    pitch_lower = str(pitch_text).lower()
    positive_words = ["growth", "profitability", "scalable", "revenue", "patent", "unique"]
    negative_words = ["debt", "loss", "struggling", "decline", "competitors", "crowded"]

    if any(word in pitch_lower for word in positive_words):
        base_score += 20; risk = "Low"
    if any(word in pitch_lower for word in negative_words):
        base_score -= 25; risk = "High"

    final_score = min(max(int(base_score), 0), 100)

    return {
        "startup_name": startup_name, "funding": funding, "equity": equity,
        "keywords": ", ".join(top_keywords).title(), "sentiment": sentiment_label,
        "score": final_score, "risk": risk
    }

# --- 4. FRONTEND UI ---
st.title("💡 PitchSense AI")
st.subheader("NLP-Based Startup Pitch Analyzer for Investment Decision Making")

# Sidebar for Project Details
with st.sidebar:
    st.header("Project Details")
    st.write("**Name:** Bhavika Jadon")
    st.write("**Reg No:** RA2411056010333")
    st.divider()
    if df is not None:
        st.success(f"✅ Dataset Loaded: {len(df)} pitches ready.")
    else:
        st.error("❌ Dataset not found. Please ensure 'Shark Tank US dataset.csv' is in the folder.")

# Main Interface Tabs
tab1, tab2 = st.tabs(["📝 Analyze Custom Pitch", "🦈 Explore Shark Tank Database"])

with tab1:
    st.markdown("### Test the NLP Pipeline")
    user_pitch = st.text_area("Paste a startup pitch here:", height=150, placeholder="We are a food delivery startup seeking $500,000 for 10% equity. We have achieved 200% growth and profitability in 6 months.")
    
    if st.button("Analyze Pitch", type="primary"):
        if user_pitch and vectorizer:
            with st.spinner("Analyzing with PitchSense AI..."):
                results = analyze_pitch(user_pitch)
                
                # Visual Score Cards
                col1, col2, col3 = st.columns(3)
                col1.metric("Investment Score", f"{results['score']}/100")
                col2.metric("Risk Level", results['risk'])
                col3.metric("Sentiment", results['sentiment'])
                
                # Data Extraction Table
                st.markdown("#### 🔍 Extracted Entities & Insights")
                st.table({
                    "Feature": ["Startup Name", "Funding Mentioned", "Equity Mentioned", "Core Themes"],
                    "Value": [results['startup_name'], results['funding'], results['equity'], results['keywords']]
                })
        else:
            st.warning("Please enter a pitch to analyze.")

with tab2:
    st.markdown("### Batch Analysis on Historical Data")
    if df is not None:
        # Create a dropdown to select a startup from the dataset
        sample_size = min(50, len(df))
        startup_options = df[text_column].head(sample_size).apply(lambda x: str(x)[:60] + "...").tolist()
        
        selected_pitch_preview = st.selectbox("Select a historical pitch from the dataset to analyze:", startup_options)
        
        # Find the full text of the selected pitch
        selected_index = startup_options.index(selected_pitch_preview)
        full_pitch_text = df.iloc[selected_index][text_column]
        
        st.info(f"**Full Pitch Text:**\n\n{full_pitch_text}")
        
        if st.button("Run Analysis on Selected Pitch"):
            with st.spinner("Processing..."):
                hist_results = analyze_pitch(full_pitch_text)
                h_col1, h_col2, h_col3 = st.columns(3)
                h_col1.metric("Investment Score", f"{hist_results['score']}/100")
                h_col2.metric("Risk Level", hist_results['risk'])
                h_col3.metric("Sentiment", hist_results['sentiment'])
                
                st.write("**Extracted Keywords:**", hist_results['keywords'])
                st.write("**Financials:**", f"Funding: {hist_results['funding']} | Equity: {hist_results['equity']}")