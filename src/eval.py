import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve, validation_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def kfold_scores(estimator, X, y, cv=5, scoring="accuracy", random_state=42):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    return cross_val_score(estimator, X, y, cv=skf, scoring=scoring, n_jobs=-1)

def plot_learning_curve(estimator, X, y, cv=5, scoring="accuracy", train_sizes=np.linspace(0.2, 1.0, 5)):
    plt.figure()
    tr_sizes, tr_scores, te_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, train_sizes=train_sizes, n_jobs=-1, shuffle=True, random_state=42
    )
    plt.plot(tr_sizes, tr_scores.mean(axis=1), marker="o", label="Train")
    plt.plot(tr_sizes, te_scores.mean(axis=1), marker="s", label="CV")
    plt.xlabel("Training examples"); plt.ylabel(scoring); plt.title("Learning Curve"); plt.legend(); plt.tight_layout()

def plot_validation_curve(estimator, X, y, cv=5, param_name="clf__C", param_range=None, scoring="accuracy"):
    if param_range is None:
        param_range = np.logspace(-2, 2, 7)
    plt.figure()
    tr_scores, te_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range, cv=cv, scoring=scoring, n_jobs=-1
    )
    plt.semilogx(param_range, tr_scores.mean(axis=1), marker="o", label="Train")
    plt.semilogx(param_range, te_scores.mean(axis=1), marker="s", label="CV")
    plt.xlabel(param_name); plt.ylabel(scoring); plt.title(f"Validation Curve: {param_name}"); plt.legend(); plt.tight_layout()

def plot_confusion(y_true, y_pred, labels=None, normalize="true"):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    plt.figure(); disp.plot(include_values=True, xticks_rotation="vertical"); plt.tight_layout()

def print_report(y_true, y_pred, target_names=None):
    print(classification_report(y_true, y_pred, target_names=target_names))
