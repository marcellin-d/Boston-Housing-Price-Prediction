
---

# Boston Housing Price Prediction

## Description

This project aims to predict the housing prices in Boston using various machine learning techniques, including linear regression, decision trees, and random forests. The data pipeline is designed to handle data ingestion, preprocessing, normalization, splitting into training and testing sets, and model evaluation.

## Table of Contents

- [Installation](#installation)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Model Training and Evaluation](#model-training-and-evaluation)
  - [Predictions](#predictions)
- [Model Evaluation](#model-evaluation)
- [Results](#results)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/username/boston-price-prediction.git
   ```

2. **Navigate to the project directory:**

   ```bash
   cd boston-price-prediction
   ```

3. **Create and activate a virtual environment:**

   For Linux/Mac:
   ```bash
   python -m venv dataenv
   source dataenv/bin/activate
   ```

   For Windows:
   ```bash
   python -m venv dataenv
   .\dataenv\Scripts\activate
   ```

4. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Dependencies

- `pandas`
- `polars`
- `scikit-learn`
- `numpy`
- `logging`

## Project Structure

Here’s the structure of the project:

```bash
boston-price-prediction/
│
├── data/
│   ├── boston_data.csv         # Raw data from the Boston Housing Dataset
│   └── new_data.csv            # New data for predictions
│
├── models/
│   └── best_model.pkl          # Best performing model (Random Forest)
│
├── src/
│   ├── pipeline.py             # Main data pipeline for preprocessing
│   ├── data_preprocessing.py   # Class for data preprocessing
│   ├── data_splitting.py       # Class for splitting data into features and labels
│   └── predict.py              # Predictions with the trained model
│
├── predictions.csv             # File containing prediction results
├── requirements.txt            # List of Python dependencies
└── README.md                   # Project documentation
```

## Usage

### Data Preparation

Load the raw data using `pipeline.py`:

```python
from src.pipeline import DataPreprocessing

# Load data
df = pd.read_csv('data/boston_data.csv')

# Preprocess the data
data_preprocessor = DataPreprocessing(df)
df_cleaned = data_preprocessor.clean_data()
```

Split the data into features and labels using `DataSplitting`:

```python
from src.data_splitting import DataSplitting

data_splitter = DataSplitting(df_cleaned)
features, labels = data_splitter.split_data()
```

### Model Training and Evaluation

Train models (linear regression, decision tree, and random forest) and evaluate them:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Split data into train and test sets
X_train, X_test, y_train, y_test = data_splitter.split_data_training_and_testing(features, labels)

# Train the Random Forest model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Make predictions and evaluate
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')
```

### Predictions

Load new data and make predictions:

```python
from src.predict import Predict

# Load new data
new_data = pd.read_csv('data/new_data.csv')

# Make predictions using the trained model
predictions = Predict('models/best_model.pkl').predict(new_data)
```

The prediction results will be saved in the `predictions.csv` file.

## Model Evaluation

The evaluated models are as follows:

- **Linear Regression:** R² = 0.6687, MSE = 24.29
- **Decision Tree:** R² = 0.6892, MSE = 22.79
- **Random Forest:** R² = 0.8824, MSE = 8.62 (Best model)

## Results

The predictions made on new data are:

```python
[20.52, 26.193, 13.487, 38.116, 19.928, 26.178, 13.336, 22.043, 17.745, 28.522]
```

The results are also saved in the `predictions.csv` file.


---

### Key Sections:

1. **Clear Introduction:** Brief explanation of the project’s goal.
2. **Installation & Dependencies:** Steps for cloning and setting up the project.
3. **Project Structure:** Shows how the project is organized.
4. **Detailed Usage:** Instructions for data preparation, model training and evaluation, and making predictions.
5. **Model Evaluation:** Summarizes the performance of each model.
6. **Results:** Displays the predictions generated.


## 📫 Contact

For questions or suggestions, feel free to reach out:  
- **Name**: Marcellin DJAMBO
- **Email**: djambomarcellin@gmail.com
- **LinkedIn**: [My LinkedIn Profile](https://www.linkedin.com/in/marcellindjambo)
