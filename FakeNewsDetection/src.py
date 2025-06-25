import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Download ALL required NLTK resources with error handling
try:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('omw-1.4')  # Required for WordNet lemmatizer
    nltk.download('punkt_tab') # Specifically needed for tokenization tables
except Exception as e:
    st.error(f"Error downloading NLTK resources: {str(e)}")
    st.stop()

# Set page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon=":newspaper:",
    layout="wide"
)

# Custom CSS with high contrast colors
st.markdown("""
<style>
    :root {
        --primary: #ffffff;
        --secondary: #f0f2f6;
        --accent: #0068c9;
        --text: #000000;
        --warning: #ff4b4b;
        --success: #00d154;
        --header: #002366;
    }
    
    .main {
        background-color: var(--primary);
        color: var(--text);
    }
    
    .sidebar .sidebar-content {
        background-color: var(--secondary) !important;
        color: var(--text);
        border-right: 1px solid #ddd;
    }
    
    .stTextInput>div>div>input {
        background-color: white;
        color: black;
        border: 2px solid var(--accent);
    }
    
    .stTextArea>div>div>textarea {
        background-color: white;
        color: black;
        border: 2px solid var(--accent);
    }
    
    .stButton>button {
        background-color: var(--accent);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        font-size: 1rem;
    }
    
    .stButton>button:hover {
        background-color: #0056b3;
        color: white;
    }
    
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        font-size: 18px;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .real-news {
        background-color: var(--success);
        color: white;
        border-left: 5px solid #008a3e;
    }
    
    .fake-news {
        background-color: var(--warning);
        color: white;
        border-left: 5px solid #c00000;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: var(--header) !important;
    }
    
    .stDataFrame {
        background-color: white !important;
        color: black !important;
        border: 1px solid #ddd;
    }
    
    .css-1aumxhk {
        background-color: white;
        color: black;
    }
    
    /* Better contrast for all text */
    body {
        color: #000000 !important;
    }
    
    /* Better table styling */
    table {
        color: #000000 !important;
    }
    
    /* Better select box contrast */
    .st-bd, .st-cb, .st-ca {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# Load dataset from local files
def load_dataset():
    try:
        # Load all TSV files
        train_df = pd.read_csv('train.tsv', sep='\t', header=None)
        test_df = pd.read_csv('test.tsv', sep='\t', header=None)
        valid_df = pd.read_csv('valid.tsv', sep='\t', header=None)
        
        # Combine all datasets
        df = pd.concat([train_df, test_df, valid_df])
        
        # Set column names (assuming LIAR dataset format)
        df.columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party', 
                     'barely_true', 'false', 'half_true', 'mostly_true', 'pants_fire', 'context']
        
        # Simplify labels
        df['label'] = df['label'].map({
            'true': 1,
            'mostly-true': 1,
            'half-true': 1,
            'barely-true': 0,
            'false': 0,
            'pants-fire': 0
        })
        
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# Text preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    words = nltk.word_tokenize(text)
    # Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    # Stem words
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

# Train and save model function
def train_model():
    with st.spinner("Loading and preparing dataset..."):
        df = load_dataset()
        if df is None:
            return None, None, None
            
        df['processed_text'] = df['statement'].apply(preprocess_text)
        
        # Feature extraction
        tfidf = TfidfVectorizer(max_features=5000)
        X = tfidf.fit_transform(df['processed_text']).toarray()
        y = df['label']
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Logistic Regression
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        # Save model and vectorizer
        joblib.dump(model, 'fake_news_detector.pkl')
        joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return model, tfidf, accuracy

# Prediction function
def predict_news(text, model, vectorizer):
    # Preprocess text
    processed_text = preprocess_text(text)
    # Vectorize
    text_vec = vectorizer.transform([processed_text]).toarray()
    # Predict
    prediction = model.predict(text_vec)
    probability = model.predict_proba(text_vec)
    
    return prediction[0], probability[0]

# Main App
def main():
    st.title("📰 Fake News Detection System")
    st.markdown("""
    <div style="color: #000000;">
    This application uses Natural Language Processing (NLP) and Machine Learning to classify news articles as real or fake.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'vectorizer' not in st.session_state:
        st.session_state.vectorizer = None
    if 'accuracy' not in st.session_state:
        st.session_state.accuracy = None
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose a page", ["Home", "Try the Model", "Model Details"])
    
    if app_mode == "Home":
        st.header("Welcome to the Fake News Detector")
        st.markdown("""
        <div style="color: #000000;">
        ### About Fake News
        Fake news refers to misinformation or disinformation published as news to mislead readers for political or financial gain.
        
        ### How It Works
        1. Enter a news article or statement in the text box
        2. Our model analyzes the text using NLP techniques
        3. Get an instant prediction with confidence score
        
        ### Dataset Information
        We use the LIAR dataset which contains labeled statements from PolitiFact.
        </div>
        """, unsafe_allow_html=True)
        
        if st.checkbox("Show sample data"):
            df = load_dataset()
            if df is not None:
                st.dataframe(df.head().style.set_properties(**{
                    'background-color': 'white',
                    'color': 'black',
                    'border': '1px solid #ddd'
                }))
    
    elif app_mode == "Try the Model":
        st.header("Test the Fake News Detector")
        
        # Load or train model
        if st.session_state.model is None:
            st.session_state.model, st.session_state.vectorizer, st.session_state.accuracy = train_model()
            if st.session_state.model is not None:
                st.success(f"Model trained successfully with accuracy: {st.session_state.accuracy:.2%}")
            else:
                st.error("Failed to train model. Please check your dataset files.")
                return
        
        # Input text area
        news_text = st.text_area("Enter the news article text:", height=200,
                               placeholder="Paste news content here...")
        
        if st.button("Analyze News", type="primary"):
            if news_text.strip() == "":
                st.warning("Please enter some text to analyze")
            else:
                prediction, probability = predict_news(news_text, st.session_state.model, st.session_state.vectorizer)
                
                # Display prediction
                if prediction == 1:
                    st.markdown(f"""
                    <div class="prediction-box real-news">
                        ✅ This news appears to be REAL with {probability[1]:.2%} confidence
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box fake-news">
                        ❌ This news appears to be FAKE with {probability[0]:.2%} confidence
                    </div>
                    """, unsafe_allow_html=True)
    
    elif app_mode == "Model Details":
        st.header("Model Information")
        st.markdown("""
        <div style="color: #000000;">
        ### Technical Details
        - **Algorithm**: Logistic Regression
        - **Feature Extraction**: TF-IDF with 5000 features
        - **Text Preprocessing**:
            - Lowercasing
            - Special character removal
            - Tokenization
            - Stopword removal
            - Lemmatization
            - Stemming
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.accuracy:
            st.metric("Model Accuracy", f"{st.session_state.accuracy:.2%}")
        
        if st.session_state.model is not None:
            # Load data for evaluation metrics
            df = load_dataset()
            if df is not None:
                df['processed_text'] = df['statement'].apply(preprocess_text)
                tfidf = st.session_state.vectorizer
                X = tfidf.transform(df['processed_text']).toarray()
                y = df['label']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Get predictions
                y_pred = st.session_state.model.predict(X_test)
                
                # Show classification report
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.set_properties(**{
                    'background-color': 'white',
                    'color': 'black',
                    'border': '1px solid #ddd'
                }).highlight_max(axis=0))
                
                # Show confusion matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(8,6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=['Fake', 'Real'], 
                            yticklabels=['Fake', 'Real'],
                            ax=ax)
                ax.set_xlabel('Predicted', fontsize=12)
                ax.set_ylabel('Actual', fontsize=12)
                ax.set_title('Confusion Matrix', fontsize=14)
                st.pyplot(fig)

if __name__ == "__main__":
    main()