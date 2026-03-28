from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

@dataclass
class ModelSpec:
    name: str
    pipeline: Pipeline
    param_grid: Dict[str, Any]

def model_zoo(random_state: int = 42) -> Tuple[ModelSpec, ...]:
    models = []

    # Logistic Regression: remove deprecated multi_class arg; use lbfgs (multinomial-capable)
    pipe_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs"))
    ])
    grid_lr = {"clf__C": np.logspace(-2, 2, 7)}
    models.append(ModelSpec("LogReg(L2)", pipe_lr, grid_lr))

    # kNN
    pipe_knn = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier())
    ])
    grid_knn = {"clf__n_neighbors": [1, 3, 5, 7, 11, 15]}
    models.append(ModelSpec("kNN", pipe_knn, grid_knn))

    # Linear SVM
    pipe_lsvm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC())
    ])
    grid_lsvm = {"clf__C": np.logspace(-2, 2, 7)}
    models.append(ModelSpec("LinearSVM", pipe_lsvm, grid_lsvm))

    # RBF SVM
    pipe_svm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf"))
    ])
    grid_svm = {"clf__C": np.logspace(-1, 3, 7),
                "clf__gamma": np.logspace(-4, -1, 8)}
    models.append(ModelSpec("RBF-SVM", pipe_svm, grid_svm))

    # Random Forest
    pipe_rf = Pipeline([("clf", RandomForestClassifier(random_state=random_state))])
    grid_rf = {"clf__n_estimators": [200, 400],
               "clf__max_depth": [None, 10, 20]}
    models.append(ModelSpec("RandomForest", pipe_rf, grid_rf))

    # Decision Tree
    pipe_dt = Pipeline([("clf", DecisionTreeClassifier(random_state=random_state))])
    grid_dt = {"clf__max_depth": [None, 5, 10, 20]}
    models.append(ModelSpec("DecisionTree", pipe_dt, grid_dt))

    # AdaBoost
    pipe_ada = Pipeline([("clf", AdaBoostClassifier(random_state=random_state))])
    grid_ada = {"clf__n_estimators": [50, 100, 200],
                "clf__learning_rate": [0.1, 0.5, 1.0]}
    models.append(ModelSpec("AdaBoost", pipe_ada, grid_ada))

    return tuple(models)
