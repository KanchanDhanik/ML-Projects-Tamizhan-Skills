import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

class LoanEligibilityPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        
    def load_data(self, filepath):
        """Load and preprocess the data"""
        data = pd.read_csv(filepath)
        
        # Preprocessing
        data['Dependents'] = data['Dependents'].replace('3+', 3)
        data['Loan_Amount_Term'] = data['Loan_Amount_Term'].astype(float)
        
        # Handle missing values
        data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
        data['Married'].fillna(data['Married'].mode()[0], inplace=True)
        data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)
        data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace=True)
        data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)
        data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0], inplace=True)
        data['LoanAmount'].fillna(data['LoanAmount'].median(), inplace=True)
        
        return data
    
    def preprocess_data(self, data):
        """Encode categorical variables and scale numerical features"""
        categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
        numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Dependents']
        
        # Label encoding for categorical variables
        for col in categorical_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            self.label_encoders[col] = le
        
        # Standard scaling for numerical variables
        self.scaler = StandardScaler()
        data[numerical_cols] = self.scaler.fit_transform(data[numerical_cols])
        
        X = data.drop(['Loan_ID', 'Loan_Status'], axis=1)
        y = data['Loan_Status']
        
        return X, y
    
    def train_model(self, X, y):
        """Train Random Forest Classifier"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.2f}")
        
    def save_model(self):
        """Save the trained model and preprocessing objects"""
        joblib.dump(self.model, 'loan_model.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')
        joblib.dump(self.label_encoders, 'label_encoder.pkl')
    
    def load_saved_model(self):
        """Load the saved model and preprocessing objects"""
        self.model = joblib.load('loan_model.pkl')
        self.scaler = joblib.load('scaler.pkl')
        self.label_encoders = joblib.load('label_encoder.pkl')
    
    def predict_loan_eligibility(self, input_data):
        """Predict loan eligibility for new data"""
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Preprocess the input data
        categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
        numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Dependents']
        
        # Label encode categorical variables
        for col in categorical_cols:
            le = self.label_encoders[col]
            input_df[col] = le.transform(input_df[col])
        
        # Scale numerical variables
        input_df[numerical_cols] = self.scaler.transform(input_df[numerical_cols])
        
        # Make prediction
        prediction = self.model.predict(input_df)
        probability = self.model.predict_proba(input_df)
        
        return prediction[0], probability[0][1]

# Train and save the model when this file is run directly
if __name__ == "__main__":
    predictor = LoanEligibilityPredictor()
    data = predictor.load_data('train.csv')
    X, y = predictor.preprocess_data(data)
    predictor.train_model(X, y)
    predictor.save_model()
    print("Model trained and saved successfully!")