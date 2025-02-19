from src.data_utils import split_data
from sklearn.model_selection import train_test_split
from src.models import ModelRunner
from logging_config import setup_logger

# Set up logger
logger = setup_logger()

def train_and_evaluate_model(model, target_column, data, param_grid=None):
    """
    Function to train and evaluate a given model using an initial split and k-fold cross-validation.

    Args:
        model: The model to train (e.g., DecisionTree, RandomForest).
        param_grid: The hyperparameter grid for the model.
        target_column: The target column for classification.
        data: The dataset to use.
        skf: StratifiedKFold instance for cross-validation.
    """
    # Remove rows with classes that have fewer than 10 samples
    data = data[data.groupby(target_column)[target_column].transform('count') > 10]

    # Step 1: Initial split (80% train+val, 20% test)
    train_val_data, test_data = train_test_split(
        data, test_size=0.2, stratify=data[target_column], random_state=42
    )
    
    logger.info(f"Initial split: Train+Val ({len(train_val_data)}) | Test ({len(test_data)})")

    # Step 2: Perform K-Fold on the 80% train+val split

    #we have the train data and we pass it to the split data, there it should run the model for each fold technically. 
    # Split train_val_data into train and validation sets for the current fold
    train_data_X = train_val_data.drop(columns=[target_column])
    train_data_Y = train_val_data[target_column]
    X_test, y_test = test_data.drop(columns=[target_column]), test_data[target_column]



    split_data(train_data_X,train_data_Y, X_test, y_test, model, target_column, param_grid)

    
