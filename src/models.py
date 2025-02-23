from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report,mean_absolute_error, mean_squared_error, r2_score
from joblib import dump  # For saving the model
import os
import json
import sys
sys.path.append(os.path.dirname(r'C:\Users\Girish\OneDrive - HORIBA\Project%20ML\model'))
from logging_config import setup_logger
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
# Set up logger
logger = setup_logger()

class ModelRunner:
    def __init__(self, model, param_grid=None, cv=5, scoring='accuracy', model_type = 'classifier', search_method='None'):
        """
        Initialize the model runner with optional grid search.
        
        Args:
            model (object): Scikit-learn model class (e.g., DecisionTreeClassifier).
            param_grid (dict): Grid of hyperparameters for grid search.
            cv (int): Number of folds for cross-validation.
            scoring (str): Scoring metric for evaluation.
            search_method (str): Search method - 'grid_Search', 'random_search', or default.
        """
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.grid_search = None
        self.random_state = 42
        self.search_method = search_method
        self.model_type = model_type

    def train(self, X_train, y_train, dataset_name):
        """
        Train the model, using GridSearchCV or RandomizedSearchCV if param_grid is provided.
        
        Args:
            X_train (pd.DataFrame): Training feature set.
            y_train (pd.Series): Training target set.
        """
        training_metrics = {}
        if self.search_method == 'grid_Search':
            print(f"Running GridSearchCV for {self.model.__class__.__name__}...")
            self.grid_search = GridSearchCV(
                estimator=self.model(),
                param_grid=self.param_grid,
                cv=self.cv,
                scoring=self.scoring,
            )
            self.grid_search.fit(X_train, y_train)
            print(f"Best Parameters: {self.grid_search.best_params_}")
            self.model = self.grid_search.best_estimator_
            # Log model parameters
            training_metrics["model_parameters"] = self.model.get_params()

            # Calculate training metrics
            predictions = self.model.predict(X_train)
            if self.model_type == 'classifier':
                training_metrics.update({
                    "accuracy": accuracy_score(y_train, predictions),
                    "precision": precision_score(y_train, predictions, average='weighted'),
                    "recall": recall_score(y_train, predictions, average='weighted'),
                    "f1_score": f1_score(y_train, predictions, average='weighted'),
                    "classification_report": classification_report(y_train, predictions, output_dict=True)
                })
                base_path = r"/Users/shashankhmg/Documents/AXA-Casestudy/Data-Science-Challenge/src/RF/Regressor/Train"


                metrics_file = os.path.join(base_path, f"RF_{dataset_name}_metrics.json")
                with open(metrics_file, "w") as f:
                    json.dump(training_metrics, f, indent=4)
                logger.info(f"Training metrics saved to {metrics_file}")
            else: 
                # Compute Metrics for Training Data
                training_metrics = {
                    "MAE": mean_absolute_error(y_train, predictions),
                    "MSE": mean_squared_error(y_train, predictions),
                    "RMSE": np.sqrt(mean_squared_error(y_train, predictions)),
                    "R2_Score": r2_score(y_train, predictions),
                }
                base_path = r"/Users/shashankhmg/Documents/AXA-Casestudy/Data-Science-Challenge/src/RF/Regressor/Train"


                metrics_file = os.path.join(base_path, f"RF_{dataset_name}_metrics.json")
                with open(metrics_file, "w") as f:
                    json.dump(training_metrics, f, indent=4)
                logger.info(f"Training metrics saved to {metrics_file}")

        elif self.search_method == 'random_search':
            logger.info(f"Running RandomizedSearchCV for {self.model.__class__.__name__}...")
            self.grid_search = RandomizedSearchCV(
                estimator=self.model(),
                param_distributions=self.param_grid,
                cv=self.cv,
                scoring=self.scoring,
                n_iter=10,  # Number of random combinations to try
                random_state=self.random_state,
                verbose=2  # Increase verbosity to see progress
            )
            self.grid_search.fit(X_train, y_train)
            logger.info(f"Best Parameters: {self.grid_search.best_params_}")
            self.model = self.grid_search.best_estimator_
            # Log model parameters
            training_metrics["model_parameters"] = self.model.get_params()

            # Log all parameter combinations and their results
            cv_results = self.grid_search.cv_results_

            # Convert to DataFrame for easier analysis
            import pandas as pd
            cv_results_df = pd.DataFrame(cv_results)

            # Save the results to a CSV for review
            results_file = "/Users/shashankhmg/Documents/AXA-Casestudy/Data-Science-Challenge/src/DT/random_search_results.csv"
            cv_results_df.to_csv(results_file, index=False)
            logger.info(f"All model parameters and results saved to {results_file}")



            # Calculate training metrics
            predictions = self.model.predict(X_train)
            if self.model_type == 'classifier':
                training_metrics.update({
                    "accuracy": accuracy_score(y_train, predictions),
                    "precision": precision_score(y_train, predictions, average='weighted'),
                    "recall": recall_score(y_train, predictions, average='weighted'),
                    "f1_score": f1_score(y_train, predictions, average='weighted'),
                    "classification_report": classification_report(y_train, predictions, output_dict=True)
                })
                base_path = r"/Users/shashankhmg/Documents/AXA-Casestudy/Data-Science-Challenge/src/RF/Regressor/Train"


                metrics_file = os.path.join(base_path, f"RF_{dataset_name}_metrics.json")
                with open(metrics_file, "w") as f:
                    json.dump(training_metrics, f, indent=4)
                logger.info(f"Training metrics saved to {metrics_file}")
            else: 
                # Compute Metrics for Training Data
                training_metrics = {
                    "MAE": mean_absolute_error(y_train, predictions),
                    "MSE": mean_squared_error(y_train, predictions),
                    "RMSE": np.sqrt(mean_squared_error(y_train, predictions)),
                    "R2_Score": r2_score(y_train, predictions),
                }
                base_path = r"/Users/shashankhmg/Documents/AXA-Casestudy/Data-Science-Challenge/src/RF/Regressor/Train"


                metrics_file = os.path.join(base_path, f"RF_{dataset_name}_metrics.json")
                with open(metrics_file, "w") as f:
                    json.dump(training_metrics, f, indent=4)
                logger.info(f"Training metrics saved to {metrics_file}")
        

        else:
            # Train with default parameters
            logger.info(f"Training {self.model.__name__} with default parameters...")
            self.model = self.model()  # Instantiate the model with default parameters
            self.model.fit(X_train, y_train)
            # Log model parameters
            training_metrics["model_parameters"] = self.model.get_params()

            # Calculate training metrics
            predictions = self.model.predict(X_train)
            if self.model_type == 'classifier':
                training_metrics.update({
                    "accuracy": accuracy_score(y_train, predictions),
                    "precision": precision_score(y_train, predictions, average='weighted'),
                    "recall": recall_score(y_train, predictions, average='weighted'),
                    "f1_score": f1_score(y_train, predictions, average='weighted'),
                    "classification_report": classification_report(y_train, predictions, output_dict=True)
                })
                base_path = r"/Users/shashankhmg/Documents/AXA-Casestudy/Data-Science-Challenge/src/LR/Train"


                metrics_file = os.path.join(base_path, f"LR_{dataset_name}_metrics.json")
                with open(metrics_file, "w") as f:
                    json.dump(training_metrics, f, indent=4)
                logger.info(f"Training metrics saved to {metrics_file}")
            else: 
                # Compute Metrics for Training Data
                training_metrics = {
                    "MAE": mean_absolute_error(y_train, predictions),
                    "MSE": mean_squared_error(y_train, predictions),
                    "RMSE": np.sqrt(mean_squared_error(y_train, predictions)),
                    "R2_Score": r2_score(y_train, predictions),
                }
                base_path = r"/Users/shashankhmg/Documents/AXA-Casestudy/Data-Science-Challenge/src/LR/Train"


                metrics_file = os.path.join(base_path, f"LR_{dataset_name}_metrics.json")
                with open(metrics_file, "w") as f:
                    json.dump(training_metrics, f, indent=4)
                logger.info(f"Training metrics saved to {metrics_file}")




    def evaluate(self, X, y, dataset_name, target_name):
        """
        Evaluate the trained model on a given dataset.
        
        Args:
            X (pd.DataFrame): Feature set.
            y (pd.Series): Target set.
            dataset_name (str): Name of the dataset (train, val, test).
            target_name (str): Name of the target variable.
        
        Returns:
            dict: Evaluation metrics.
        """
        predictions = self.model.predict(X)
        if self.model_type == 'classifier':
                training_metrics.update({
                    "accuracy": accuracy_score(y, predictions),
                    "precision": precision_score(y, predictions, average='weighted'),
                    "recall": recall_score(y, predictions, average='weighted'),
                    "f1_score": f1_score(y, predictions, average='weighted'),
                    "classification_report": classification_report(y, predictions, output_dict=True)
                })
                base_path = r"/Users/shashankhmg/Documents/AXA-Casestudy/Data-Science-Challenge/src/RF/Val"


                metrics_file = os.path.join(base_path, f"LR_{dataset_name}_metrics.json")
                with open(metrics_file, "w") as f:
                    json.dump(training_metrics, f, indent=4)
                metrics_file = os.path.join(base_path, f'LR_{target_name}_{dataset_name}_validation.json')

        else: 
            # Compute Metrics for Training Data
            training_metrics = {
                "MAE": mean_absolute_error(y, predictions),
                "MSE": mean_squared_error(y, predictions),
                "RMSE": np.sqrt(mean_squared_error(y, predictions)),
                "R2_Score": r2_score(y, predictions),
            }
            base_path = r"/Users/shashankhmg/Documents/AXA-Casestudy/Data-Science-Challenge/src/LR/Val"



            metrics_file = os.path.join(base_path, f"LR_{dataset_name}_metrics.json")
            with open(metrics_file, "w") as f:
                json.dump(training_metrics, f, indent=4)
            metrics_file = os.path.join(base_path, f'LR_{target_name}_{dataset_name}_validation.json')


    def save_model(self, target_name):
        """
        Save the trained model to a file.
        
        Args:
            target_name (str): Name of the target variable.
        """
        base_path = r"/Users/shashankhmg/Documents/AXA-Casestudy/Data-Science-Challenge/src/LR"
        model_path = os.path.join(base_path, f'LR.joblib')
        dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")
