# 💡 PitchSense AI: NLP-Driven Startup Pitch Analyzer

![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Streamlit App](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)
![NLP Frameworks](https://img.shields.io/badge/NLP-spaCy%20%7C%20TextBlob-4EAA25.svg)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

**PitchSense AI** is an intelligent, automated pipeline designed to bridge the gap between unstructured startup pitches and quantifiable investment decisions. By leveraging advanced Natural Language Processing (NLP) techniques and Term Frequency-Inverse Document Frequency (TF-IDF) vectorization, this tool simulates the preliminary screening process utilized by venture capitalists and angel investors.

---

##  The Problem & Business Value
Venture capitalists receive thousands of pitches annually. Manually reading, extracting financial data, and assessing the viability of each pitch is time-consuming and prone to human bias. 

**PitchSense AI provides:**
* **Scalability:** The ability to batch-process historical pitch data in seconds.
* **Objectivity:** A data-driven scoring engine that evaluates risk and sentiment without subjective bias.
* **Actionable Insights:** Immediate extraction of core financial asks (funding and equity) alongside statistically significant business themes.

---

##  Key Features

* **Intelligent Entity Extraction:** Utilizes state-of-the-art Named Entity Recognition (NER) via `spaCy` to automatically identify and extract crucial data points such as the Startup Name, Funding Amount (`MONEY`), and Equity Offered (`PERCENT`).
* **Contextual Keyword Analysis:** Employs a `scikit-learn` TF-IDF vectorizer trained on a historical Shark Tank US dataset to filter out generic business jargon and identify the unique value propositions of a given pitch.
* **Sentiment & Risk Engine:** Analyzes the underlying tone of the pitch using `TextBlob`, cross-referencing vocabulary against predefined growth and risk matrices to calculate an objective **Investment Score (0-100)**.
* **Interactive Web Dashboard:** A clean, responsive UI built with `Streamlit` that allows users to test custom pitches in real-time or explore historical data from the Shark Tank database.

---

##  System Architecture & Methodology

The application follows a linear NLP pipeline:
1. **Input Acquisition:** Raw text ingested via the Streamlit frontend.
2. **Text Normalization & Preprocessing:** Tokenization and lemmatization handled by the `en_core_web_sm` model.
3. **Feature Engineering:** Text is transformed into numerical vectors to extract top `n-gram` features.
4. **Scoring Logic:** Baseline scores are dynamically adjusted based on polarity (sentiment) and keyword presence (e.g., "scalable", "profitability" vs. "debt", "loss").
5. **Output Generation:** Structured JSON data is mapped to the frontend metric cards.

---

##  Technology Stack

* **Core Language:** Python 3.x
* **Frontend Framework:** Streamlit
* **Natural Language Processing:** spaCy, TextBlob, NLTK
* **Machine Learning & Math:** Scikit-learn, Pandas, NumPy
* **Data Source:** Shark Tank US Historical Dataset (CSV)
