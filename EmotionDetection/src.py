import streamlit as st
import pandas as pd
import re
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Emotion Detection App",
    page_icon="😊",
    layout="wide"
)

# Emotion icons mapping
EMOTION_ICONS = {
    'sadness': '😢',
    'joy': '😊',
    'happy': '😄',
    'love': '❤️',
    'anger': '😠',
    'fear': '😨',
    'surprise': '😲',
    'neutral': '😐'
}

@st.cache_resource
def load_or_train_model():
    """Load existing model or train a new one"""
    try:
        # Try to load existing model
        model = joblib.load('models/emotion_model.joblib')
        le = joblib.load('models/label_encoder.joblib')
        vectorizer = joblib.load('models/vectorizer.joblib')
        st.success("Loaded pre-trained model")
        return model, le, vectorizer
        
    except FileNotFoundError:
        st.warning("No pre-trained model found. Training new model...")
        
        # Load your dataset
        try:
            df = pd.read_pickle("merged_training.pkl")
        except Exception as e:
            st.error(f"Could not load dataset: {str(e)}")
            st.stop()

        # Verify columns - using YOUR column names
        text_col = 'text'
        emotion_col = 'emotions'
        
        if text_col not in df.columns or emotion_col not in df.columns:
            st.error(f"Required columns not found. Available columns: {df.columns.tolist()}")
            st.stop()

        # Preprocess text
        def clean_text(text):
            text = str(text).lower()
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\d+', '', text)
            return text.strip()
        
        df['cleaned_text'] = df[text_col].apply(clean_text)

        # Encode labels
        le = LabelEncoder()
        df['emotion_encoded'] = le.fit_transform(df[emotion_col])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['cleaned_text'],
            df['emotion_encoded'],
            test_size=0.2,
            random_state=42
        )

        # Train model
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        X_train_vec = vectorizer.fit_transform(X_train)

        model = LogisticRegression(
            max_iter=1000,
            multi_class='multinomial',
            solver='lbfgs',
            class_weight='balanced'
        )
        model.fit(X_train_vec, y_train)

        # Save model
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/emotion_model.joblib')
        joblib.dump(le, 'models/label_encoder.joblib')
        joblib.dump(vectorizer, 'models/vectorizer.joblib')
        
        st.success("Model trained and saved successfully!")
        return model, le, vectorizer

def predict_emotion(text, model, vectorizer, le):
    """Predict emotion for new text"""
    cleaned_text = re.sub(r'[^\w\s]', '', str(text).lower())
    text_vec = vectorizer.transform([cleaned_text])
    pred = model.predict(text_vec)
    return le.inverse_transform(pred)[0]

def main():
    st.title("Emotion Detection from Text")
    
    # Load or train model
    model, le, vectorizer = load_or_train_model()
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        - **Model**: Logistic Regression with TF-IDF
        - **Text Column**: `text`
        - **Emotion Column**: `emotions`
        - **Dataset**: Your custom dataset
        """)
        
        st.header("Try Examples")
        examples = {
            "I'm feeling wonderful today!": "joy",
            "This makes me furious!": "anger",
            "I'm scared of what might happen": "fear",
            "I'm head over heels for you": "love",
            "I feel completely alone": "sadness"
        }
        
        for text, emotion in examples.items():
            if st.button(f"{EMOTION_ICONS.get(emotion, '')} {text[:25]}..."):
                st.session_state.input_text = text
    
    # Main content
    tab1, tab2 = st.tabs(["Detect Emotion", "Model Info"])
    
    with tab1:
        st.subheader("Analyze Text")
        text_input = st.text_area(
            "Enter your text here:", 
            value=st.session_state.get("input_text", ""),
            height=150
        )
        
        if st.button("Detect Emotion"):
            if not text_input.strip():
                st.warning("Please enter some text")
            else:
                with st.spinner("Analyzing..."):
                    emotion = predict_emotion(text_input, model, vectorizer, le)
                    
                    # Display result
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.markdown(f"## {EMOTION_ICONS.get(emotion.lower(), '')}")
                    with col2:
                        st.markdown(f"## Predicted Emotion: **{emotion.capitalize()}**")
                    
                    # Show probabilities
                    st.subheader("Confidence Scores")
                    text_vec = vectorizer.transform([re.sub(r'[^\w\s]', '', str(text_input).lower())])
                    probas = model.predict_proba(text_vec)[0]
                    
                    prob_df = pd.DataFrame({
                        'Emotion': le.classes_,
                        'Probability': probas
                    }).sort_values('Probability', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.barplot(
                        x='Probability', 
                        y='Emotion', 
                        data=prob_df,
                        palette='viridis'
                    )
                    ax.set_title('Prediction Confidence')
                    ax.set_xlim(0, 1)
                    st.pyplot(fig)
    
    with tab2:
        st.subheader("Model Information")
        st.markdown("""
        ### Model Architecture
        - **Text Processing**: TF-IDF Vectorization (10,000 features)
        - **Classifier**: Logistic Regression (Multinomial)
        - **Features**: Uni+Bi-grams, English stopwords removed
        
        ### Training Data
        - Using your custom dataset from `merged_training.pkl`
        - Text column: `text`
        - Emotion labels column: `emotions`
        """)

if __name__ == "__main__":
    main()