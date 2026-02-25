from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix
)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_classification(model, X_test, y_test, verbose=True):
    """
    Evaluate classification model performance.

    Returns
    -------
    accuracy : float
    report : str
    y_pred_classes : np.ndarray
    """

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    report = classification_report(
        y_true_classes,
        y_pred_classes,
        digits=4
    )

    if verbose:
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)

    return accuracy, report, y_pred_classes

def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names=None,
    title="Confusion Matrix",
    save_path=None
):
    """
    Plot confusion matrix heatmap.
    """

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names if class_names else range(cm.shape[0]),
        yticklabels=class_names if class_names else range(cm.shape[0]),
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return cm