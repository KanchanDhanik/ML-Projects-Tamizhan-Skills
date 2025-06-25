import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Plant Disease Detector",
    layout="wide"
)

# Title
st.title("🌱 Plant Disease Detection System")

# ======================================================================
# CRITICAL PATH SETTINGS - MUST MATCH YOUR EXACT FOLDER STRUCTURE
# ======================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(current_dir, "plantvillage dataset", "color")

# Verify the path exists
if not os.path.exists(dataset_dir):
    st.error(f"ERROR: Dataset folder not found at: {dataset_dir}")
    st.error("Please make sure you have this exact folder structure:")
    st.error("PlantDiseaseDetection/")
    st.error("├── plantvillage dataset/")
    st.error("│   └── color/")
    st.error("│       ├── Tomato___Early_blight/")
    st.error("│       ├── Tomato___Late_blight/")
    st.error("│       └── ... etc ...")
    st.error(f"Current directory contents: {os.listdir(current_dir)}")
    st.stop()

# ======================================================================
# SELECT CLASSES (must exist in your dataset)
# ======================================================================
SELECTED_CLASSES = [
    'Tomato___Early_blight',
    'Tomato___Late_blight', 
    'Tomato___healthy',
    'Potato___Early_blight',
    'Potato___healthy'
]

# Verify these classes exist
missing_classes = [c for c in SELECTED_CLASSES if not os.path.exists(os.path.join(dataset_dir, c))]
if missing_classes:
    st.error(f"Missing classes: {missing_classes}")
    st.info(f"Available classes: {os.listdir(dataset_dir)[:10]}...")
    st.stop()

# ======================================================================
# MODEL SETTINGS
# ======================================================================
IMG_SIZE = (128, 128)  # Smaller size for faster training
BATCH_SIZE = 16
EPOCHS = 5
model_path = os.path.join(current_dir, "plant_disease_model.h5")

# ======================================================================
# TRAINING FUNCTION
# ======================================================================
def train_model():
    try:
        # 1. Prepare data generators
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            validation_split=0.2  # 20% for validation
        )

        st.write(f"✔ Loading data from: {dataset_dir}")

        train_generator = train_datagen.flow_from_directory(
            dataset_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training',
            classes=SELECTED_CLASSES
        )

        val_generator = train_datagen.flow_from_directory(
            dataset_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            classes=SELECTED_CLASSES,
            shuffle=False
        )

        # 2. Build model
        model = models.Sequential([
            layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(64, (3,3), activation='relu'),
            layers.MaxPooling2D((2,2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(len(SELECTED_CLASSES), activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # 3. Train model
        with st.spinner(f"Training for {EPOCHS} epochs..."):
            history = model.fit(
                train_generator,
                steps_per_epoch=train_generator.samples // BATCH_SIZE,
                epochs=EPOCHS,
                validation_data=val_generator,
                validation_steps=val_generator.samples // BATCH_SIZE
            )

        # 4. Save and show results
        model.save(model_path)
        st.success("✔ Model trained successfully!")
        
        # Plot accuracy/loss
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history.history['accuracy'], label='Train Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.legend()
        
        ax2.plot(history.history['loss'], label='Train Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.legend()
        
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Training failed: {str(e)}")
        st.error("Please check:")
        st.error(f"- Dataset path: {dataset_dir}")
        st.error(f"- Folder exists: {os.path.exists(dataset_dir)}")
        st.error(f"- Contents: {os.listdir(dataset_dir)[:5]}...")

# ======================================================================
# PREDICTION FUNCTION
# ======================================================================
def predict_disease(image):
    try:
        if not os.path.exists(model_path):
            st.error("❌ Model not found! Train first.")
            return None, None
            
        model = tf.keras.models.load_model(model_path)
        img = image.resize(IMG_SIZE)
        img_array = img_to_array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        pred = model.predict(img_array)
        return SELECTED_CLASSES[np.argmax(pred)], np.max(pred)
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        return None, None

# ======================================================================
# STREAMLIT APP
# ======================================================================
def main():
    st.sidebar.title("Menu")
    app_mode = st.sidebar.radio(
        "Choose Mode",
        ["🏠 Home", "🛠 Train Model", "🔍 Detect Disease", "ℹ Dataset Info"]
    )

    if app_mode == "🏠 Home":
        st.header("Welcome!")
        st.image("https://images.unsplash.com/photo-1586771107445-d3ca888129ce", width=600)
        st.write("This app detects diseases in tomato and potato plants.")

    elif app_mode == "🛠 Train Model":
        st.header("Train Model")
        st.write(f"Will train on: {SELECTED_CLASSES}")
        if st.button("Start Training"):
            train_model()

    elif app_mode == "🔍 Detect Disease":
        st.header("Detect Disease")
        uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, width=300)
            
            if st.button("Analyze"):
                pred_class, confidence = predict_disease(image)
                if pred_class:
                    st.success(f"Prediction: {pred_class} ({confidence:.2%})")

    elif app_mode == "ℹ Dataset Info":
        st.header("Dataset Information")
        st.write(f"Location: {dataset_dir}")
        st.write("First 10 classes:")
        st.write(os.listdir(dataset_dir)[:10])

if __name__ == "__main__":
    main()