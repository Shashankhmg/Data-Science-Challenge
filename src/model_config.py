from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
# Define a function that returns a dictionary of models and their hyperparameters
def get_models_and_params():
    models = {
        "Decision Tree": DecisionTreeClassifier
    }

    param_grid = {
        "DecisionTree": { "criterion": ["gini", "entropy", "log_loss"],  # Criteria for splitting
                         "ccp_alpha": [0.0001, 0.001, 0.01, 0.1, 1],  
                            "splitter": ["best", "random"],               # Strategy for choosing the split
                            "max_depth": list(range(100, 2000, 200)),  # Control depth of the tree
                            "min_samples_split": [50, 100, 150, 200],          # Minimum samples required to split an internal node
                            "min_samples_leaf": [1, 50, 100, 150, 200],            # Minimum samples required to be a leaf node
                            "max_features": [None, "sqrt", "log2"],       # Number of features to consider when looking for the best split
                            "max_leaf_nodes": [None] + list(range(50, 1000, 50)),  # Maximum number of leaf nodes
                            "min_impurity_decrease": [0.0, 0.01, 0.1],    # Minimum impurity decrease for splitting
                            "class_weight": [None, "balanced"],           # Adjust weights to handle class imbalance
                            },
        "RandomForest": {"max_depth": [50]},
        "NaiveBayes": {"var_smoothing": [1e-9, 1e-8]},
        "XGBoost": {"learning_rate": [0.01, 0.1], "max_depth": [3, 5]}
    }

    return models, param_grid
