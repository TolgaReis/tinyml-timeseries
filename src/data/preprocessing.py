# src/data/preprocessing.py

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical


def clean_nan_inf(X):
    """
    Replace NaN, +Inf, -Inf values with 0.

    Parameters
    ----------
    X : np.ndarray

    Returns
    -------
    np.ndarray
    """

    X_clean = np.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return X_clean


def clean_train_test(X_train, X_test):
    """
    Clean both train and test sets.
    """

    X_train = clean_nan_inf(X_train)
    X_test = clean_nan_inf(X_test)

    print("NaN & Inf values cleaned.")

    return X_train, X_test

def minmax_fit_transform_per_feature(X_train: np.ndarray, X_test: np.ndarray):
    """
    MinMax normalize each feature/channel independently.
    Assumes X shape: (N, T, C)

    Returns:
      X_train_normalized, X_test_normalized, scalers
    """
    if X_train.ndim != 3:
        raise ValueError(f"Expected X_train shape (N, T, C), got {X_train.shape}")
    if X_test.ndim != 3:
        raise ValueError(f"Expected X_test shape (N, T, C), got {X_test.shape}")

    num_samples, num_timesteps, num_features = X_train.shape

    scalers = [MinMaxScaler() for _ in range(num_features)]

    X_train_normalized = np.array([
        scalers[i].fit_transform(X_train[:, :, i]) for i in range(num_features)
    ]).transpose(1, 2, 0)

    X_test_normalized = np.array([
        scalers[i].transform(X_test[:, :, i]) for i in range(num_features)
    ]).transpose(1, 2, 0)

    print("Data normalization complete (MinMax per feature).")
    return X_train_normalized, X_test_normalized, scalers

def one_hot_encode_labels(y_train, y_val, y_test):
    """
    Convert labels to one-hot encoding.

    Returns:
        y_train_ohe, y_val_ohe, y_test_ohe, num_classes
    """

    num_classes = len(np.unique(y_train))

    y_train_ohe = to_categorical(y_train, num_classes=num_classes)
    y_val_ohe = to_categorical(y_val, num_classes=num_classes)
    y_test_ohe = to_categorical(y_test, num_classes=num_classes)

    print(f"One-hot encoding complete. Classes: {num_classes}")

    return y_train_ohe, y_val_ohe, y_test_ohe, num_classes