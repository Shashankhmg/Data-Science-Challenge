import pandas as pd
from sklearn.model_selection import StratifiedKFold
from src.data_utils import split_data
from src.model_config import get_models_and_params
from src.train_evaluate import train_and_evaluate_model
from logging_config import setup_logger

# Set up logger
logger = setup_logger()

# Load and prepare the data
data = pd.read_csv("/Users/shashankhmg/Documents/AXA-Casestudy/Data-Science-Challenge/data/model_data/cleaned_model_data.csv")
logger.info('Data successfully read')
data = data.iloc[:, 1:] # removing the unnamed column

# Get models and hyperparameters
models, param_grid = get_models_and_params()

# Define the target columns
targets = ["risk_level"]

# Iterate over the target columns
for target_column in targets:
    logger.info(f"Processing target: {target_column}")
    
    # Iterate over the models
    for model_name, model in models.items():
        logger.info(f"Training and evaluating {model_name} for {target_column}...")
                
        # Train and evaluate the model
        train_and_evaluate_model(model,target_column, data, param_grid= None)
