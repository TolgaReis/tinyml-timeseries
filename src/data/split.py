# src/data/split.py

from sklearn.model_selection import train_test_split


def stratified_split(X, y, test_size=0.2, random_state=42):
    """
    Splits dataset with stratification when possible.
    Falls back to random split if stratification fails.
    """

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        print("Stratified split successful")

    except ValueError:
        print("Stratified split failed → using random split")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )

    return X_train, X_test, y_train, y_test