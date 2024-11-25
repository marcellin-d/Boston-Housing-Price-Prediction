import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configuration du logging
logging.basicConfig(level=logging.INFO)

class DataPreprocessing:
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize with the Pandas DataFrame.
        """
        self.df = dataframe

    def handle_missing_values(self) -> pd.DataFrame:
        """
        Handles missing values by either dropping rows or filling them with a default value.
        """
        try:
            # Check for duplicate rows and drop them
            if self.df.duplicated().sum() > 0:
                self.df = self.df.drop_duplicates()
                logging.info("Duplicate rows handled (rows dropped).")
            
            return self.df
        except Exception as e:
            logging.error(f"Error handling missing values: {e}")
            return self.df 
            
    def clean_data(self) -> pd.DataFrame:
        """
        Cleans the data by handling missing values and dropping duplicates.
        """
        try:
            self.handle_missing_values()
            logging.info("Data cleaning completed successfully.")
            return self.df
        except Exception as e:
            logging.error(f"Error cleaning data: {e}")
            return self.df 
    

class DataSplitting:
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize with the Pandas DataFrame.
        """
        self.df = dataframe
        
    def split_data(self) -> tuple:
        """ 
        Splits the data into Features and Labels.
        """
        try:
            # Ensure "MEDV" column exists in the dataframe
            if "MEDV" not in self.df.columns:
                raise ValueError("MEDV column not found in the dataframe.")
            
            features = self.df.drop("MEDV", axis=1)
            labels = self.df["MEDV"]
            logging.info("Data split into features and labels.")
            return features, labels
        except Exception as e:
            logging.error(f"Error splitting data: {e}")
            return None, None 
            
    def normalize_data(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes the data using StandardScaler.
        """
        try:
            # Normalize data using StandardScaler
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(features)
            
            # Return as a Pandas DataFrame with normalized features
            df_scaled = pd.DataFrame(df_scaled, columns=features.columns)
            
            return df_scaled
        except Exception as e:
            logging.error(f"Error normalizing data: {e}")
            return features
            
    def split_data_training_and_testing(self, features: pd.DataFrame, labels: pd.Series) -> tuple:
        try:
            # Perform the data split using train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42, shuffle=True
            )
            logging.info("Data splitting successfully completed.")
            return X_train, y_train, X_test, y_test
        except Exception as e:
            logging.error(f"Error while splitting data: {e}")
            return None, None, None, None
