import pandas as pd
from sklearn.model_selection import StratifiedKFold
from src.models import ModelRunner
from logging_config import setup_logger

# Set up logger
logger = setup_logger()


def split_data(X_train_val, y_train_val, X_test, y_test, model, target_column, param_grid=None):
    """
    Splits the provided train+validation set into train and validation sets for one target.

    Args:
        X_train_val (pd.DataFrame): Features of the train+validation set.
        y_train_val (pd.Series): Target values of the train+validation set.
        target_column (str): Name of the target column.

    Returns:
        dict: Dictionary containing train and validation splits for features (X) and target (y).
    """
    # Stratified K-Fold for inner split
    skf_inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for fold, (train_index, val_index) in enumerate(skf_inner.split(X_train_val, y_train_val), start=1):
        X_train = X_train_val.iloc[train_index]
        X_val = X_train_val.iloc[val_index]
        y_train = y_train_val.iloc[train_index]
        y_val = y_train_val.iloc[val_index]
        logger.info(f"Fold {fold}:")
        logger.info(f"Training size: {X_train.shape}, Validation size: {X_val.shape}")

        logger.info(f"Running Fold {fold} for model {model.__class__.__name__} and target {target_column}...")

        # Initialize and train the model using ModelRunner
        model_runner = ModelRunner(
            model=model,
            param_grid=param_grid,
            cv=10,
            scoring="accuracy",
            search_method="None"
        )
        logger.info('training started')
        model_runner.train(X_train, y_train, f"{target_column}_train_fold{fold}")
        logger.info('validation started')
        # Evaluate on train and validation sets
        model_runner.evaluate(X_val, y_val, f"{target_column}_val_fold{fold}", target_column)
        logger.info(f"Completed Fold {fold} for model {model} and target {target_column}\n")

    # Step 3: Evaluate the best model on the held-out test set
    logger.info(f"Evaluating the best model on the test set...")
    
    model_runner.evaluate(X_test, y_test, f"{target_column}_test", target_column)

    # Save the best model
    model_runner.save_model(f"models/{model.__class__.__name__}_best_model_{target_column}.pkl")

    logger.info(f"Completed evaluation for model {model.__class__.__name__} and target {target_column}")
