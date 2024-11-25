# Pipeline

from src.train_model import train_model_and_evaluate
from src.predict import main

if __name__ == "__main__":
    file_path:str = "./data/boston.csv"
    train_model_and_evaluate(file_path=file_path)
    
    model_path = "./models/best_model.pkl"
    new_data_path = "./data/new_data.csv"  
    main(model_path, new_data_path)
