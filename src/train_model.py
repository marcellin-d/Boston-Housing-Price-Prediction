import logging
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from src.data_ingestion import IngestData
from src.data_preprocessing import DataPreprocessing, DataSplitting

# Configuration du logging
logging.basicConfig(level=logging.INFO)

def train_model_and_evaluate(file_path: str):
    """
    Main function to train the model using the dataset.
    """
    try:
        # Step 1: Data ingestion
        logging.info("Ingesting data...")
        data_loader = IngestData(file_path)
        df = data_loader.ingest_data_from_csv()
        
        if df is None:
            raise ValueError("Data ingestion failed.")
        
        logging.info("Data ingested successfully.")
        
        # Step 2: Preprocessing the data
        logging.info("Preprocessing the data...")
        preprocessing = DataPreprocessing(df)
        cleaned_df = preprocessing.clean_data()
        
        logging.info("Data preprocessing completed.")
        
        # Step 3: Splitting the data into features and labels
        logging.info("Splitting data into features and labels...")
        data_splitter = DataSplitting(cleaned_df)
        features, labels = data_splitter.split_data()

        if features is None or labels is None:
            raise ValueError("Data splitting failed.")
        
        # Step 4: Normalizing the features
        logging.info("Normalizing the features...")
        normalized_features = data_splitter.normalize_data(features)

        # Step 5: Train-test split
        logging.info("Splitting data into training and testing sets...")
        X_train, y_train, X_test, y_test = data_splitter.split_data_training_and_testing(normalized_features, labels)

        if X_train is None or y_train is None:
            raise ValueError("Train-test split failed.")
        
        # Step 6: Model training (Evaluate multiple models)
        logging.info("Training and evaluating multiple models...")

        # Initializing models
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor()
        }

        best_model = None
        # Start with the worst score
        best_score = float('-inf')  

        # Step 7: Train, predict, and evaluate models
        for model_name, model in models.items():
            logging.info(f"{'-'*10} {model_name} {'-'*10}")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Evaluate the model
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Log evaluation metrics
            logging.info(f"Model: {model_name}")
            logging.info(f"Mean Squared Error: {mse}")
            logging.info(f"R2 Score: {r2}")
            
            # Compare and store the best model
            if r2 > best_score:
                best_score = r2
                best_model = model_name

        # Log the best model based on R2 score
        logging.info(f"The best model is: {best_model} with an R2 score of {best_score}")

        # Optional: Save the best model (e.g., using joblib)
        model_path = r"C:\Users\ThinkPad\marcel\boston price pred\models\best_model.pkl"
        joblib.dump(models[best_model], model_path)

        logging.info("Model training and evaluation completed successfully.")
    
    except Exception as e:
        logging.error(f"Error in training the model: {e}")
