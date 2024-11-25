
import logging
import polars as pl
import pandas as pd
# Define a class for ingesting data from sources

# Configuration du logging
logging.basicConfig(level=logging.INFO)

class IngestData:
    """ Ingests data from a CSV, Parquet, or JSON file and returns a Polars DataFrame"""
    def __init__(self,file_path:str):
        self.file_path = file_path
        
    def ingest_data_from_csv(self)-> pd.DataFrame:
        """ Ingests data from a CSV file and returns a Polars DataFrame"""
        try:
            dataframe = pd.read_csv(self.file_path)
            return dataframe
        except Exception as e:
            logging.error(f"Error ingesting data from {self.file_path}: {e}")
            
    def ingest_data_from_parquet(self)-> pl.DataFrame:
        """ Ingests data from a Parquet file and returns a Polars DataFrame"""
        try:
            dataframe = pl.read_parquet(self.file_path)
            return dataframe
        except Exception as e:
            logging.error(f"Error ingesting data from {self.file_path}: {e}")
            
    def ingest_data_from_json(self)-> pl.DataFrame:
        """ Ingests data from a JSON file and returns a Polars DataFrame"""
        try:
            dataframe = pl.read_json(self.file_path)
            return dataframe
        except Exception as e:
            logging.error(f"Error ingesting data from {self.file_path}: {e}")
            