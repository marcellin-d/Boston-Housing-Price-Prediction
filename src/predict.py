import logging
import joblib
import pandas as pd
from src.data_preprocessing import DataSplitting
from sklearn.preprocessing import StandardScaler

# Configuration du logging
logging.basicConfig(level=logging.INFO)

def load_model(model_path: str):
    """
    Load the trained model from a file.
    """
    try:
        model = joblib.load(model_path)
        logging.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading the model: {e}")
        return None

def load_new_data(file_path: str) -> pd.DataFrame:
    """
    Load new data for prediction from a CSV file.
    """
    try:
        new_data = pd.read_csv(file_path)
        logging.info(f"New data loaded from {file_path}")
        return new_data
    except Exception as e:
        logging.error(f"Error loading new data: {e}")
        return None

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the new data (handling missing values, scaling features, etc.).
    """
    try:
        # Assuming you have a DataSplitting class to handle preprocessing
        data_splitter = DataSplitting(data)
        
        # Normalize features
        normalized_features = data_splitter.normalize_data(data)
        
        logging.info("Data preprocessing completed.")
        return normalized_features
    except Exception as e:
        logging.error(f"Error preprocessing the data: {e}")
        return None

def make_predictions(model, data: pd.DataFrame):
    """
    Make predictions using the trained model.
    """
    try:
        predictions = model.predict(data)
        return predictions
    except Exception as e:
        logging.error(f"Error making predictions: {e}")
        return None

def main(model_path: str, new_data_path: str):
    """
    Main function to load the model, preprocess new data, and make predictions.
    """
    # Load the trained model
    model = load_model(model_path)
    if model is None:
        return
    
    # Load new data
    new_data = load_new_data(new_data_path)
    if new_data is None:
        return
    
    # Preprocess the new data
    preprocessed_data = preprocess_data(new_data)
    if preprocessed_data is None:
        return
    
    # Make predictions
    predictions = make_predictions(model, preprocessed_data)
    if predictions is not None:
        # Display or save predictions
        logging.info("Predictions: ")
        print(predictions)
        # Optionally save predictions to a CSV
        prediction_df = pd.DataFrame(predictions, columns=["Predicted Prices"])
        file_path_ = r"C:\Users\ThinkPad\marcel\boston price pred\results\predictions.csv"
        prediction_df.to_csv(file_path_, index=False)
        print(prediction_df.head())
        logging.info("Predictions saved to 'predictions.csv'.")
    
