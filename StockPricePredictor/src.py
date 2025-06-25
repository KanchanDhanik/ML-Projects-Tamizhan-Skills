import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

def load_data():
    """Load and prepare the stock data with handling for unusual CSV format"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "NASDAQ_Composite_Full_History.csv")
        
        # Read CSV with header=None to handle irregular format
        df = pd.read_csv(csv_path, header=None)
        
        # Check if we have the unusual format where column names are in first row
        if df.iloc[0,0] == 'Ticker':
            # Reconstruct the dataframe properly
            new_header = df.iloc[0]  # Get the first row for header
            df = df[1:]              # Take the data less the header row
            df.columns = new_header  # Set the header row as the df header
            
            # Convert date strings to datetime
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Set Date as index and sort
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            # Convert all numeric columns to float
            numeric_cols = ['Price', 'Close', 'High', 'Low', 'Open', 'Volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Use 'Close' price for prediction
            data = df['Close'].values.reshape(-1, 1)
            
            print("\nSuccessfully processed unusual CSV format")
            print("First 5 rows of cleaned data:")
            print(df.head())
            
            return data, df
        
        # If normal CSV format, process normally
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        data = df['Close'].values.reshape(-1, 1)
        return data, df
        
    except Exception as e:
        print(f"\nError loading data: {str(e)}")
        print("\nPlease check your CSV file structure.")
        print("Expected format is either:")
        print("1. Normal format with Date, Open, High, Low, Close columns")
        print("2. Or the unusual format shown in your file")
        raise

# [Keep all other functions the same: preprocess_data, build_model, train_model, plot_results]

if __name__ == "__main__":
    print("\n=== NASDAQ Stock Price Prediction ===")
    
    try:
        # 1. Load data
        print("\n[1/5] Loading and preparing data...")
        data, df = load_data()
        
        # 2. Preprocess
        print("\n[2/5] Preprocessing data...")
        look_back = 60
        X_train, y_train, X_test, y_test, scaler = preprocess_data(data, look_back)
        
        # 3. Build model
        print("\n[3/5] Building LSTM model...")
        model = build_model(look_back)
        model.summary()
        
        # 4. Train model
        print("\n[4/5] Training model...")
        model, history = train_model(model, X_train, y_train, X_test, y_test)
        
        # 5. Evaluate and plot
        print("\n[5/5] Generating results...")
        plot_results(model, X_test, y_test, scaler, data)
        
        # Save model
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nasdaq_lstm_model.h5")
        model.save(model_path)
        print(f"\nModel successfully saved to {model_path}")
        
    except Exception as e:
        print(f"\nProgram failed: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Verify your CSV file is in the correct format")
        print("2. Check the file is not corrupted")
        print("3. Ensure all packages are properly installed")