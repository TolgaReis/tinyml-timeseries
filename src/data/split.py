# src/data/split.py
import numpy as np
from collections import defaultdict


def _find_segments(y):
    segments = []
    start = 0
    for i in range(1, len(y)):
        if y[i] != y[i - 1]:
            segments.append((start, i, int(y[start])))
            start = i
    segments.append((start, len(y), int(y[start])))
    return segments


def stratified_split(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Splits dataset with stratification when possible.
    Falls back to random split if stratification fails.

    For each contiguous label segment, splits temporally:
        train | val | test
    preserving cycle boundaries and preventing leakage.

    Args:
        test_size: fraction of total data for test
        val_size:  fraction of total data for validation (0 = no val split)

    Returns:
        If val_size > 0: X_train, X_val, X_test, y_train, y_val, y_test
        If val_size = 0: X_train, X_test, y_train, y_test
    """
    y = np.asarray(y)
    all_labels = set(np.unique(y))
    train_size = 1.0 - test_size - val_size
    assert train_size > 0, "train_size must be > 0"

    segments = _find_segments(y)

    train_idx, val_idx, test_idx = [], [], []

    for start, end, label in segments:
        seg_len = end - start
        n_test  = max(1, round(seg_len * test_size))
        n_val   = max(1, round(seg_len * val_size)) if val_size > 0 else 0
        n_train = seg_len - n_test - n_val

        if n_train <= 0:
            # Segment too small — give all to train
            train_idx.extend(range(start, end))
            continue

        # Temporal order: [train | val | test]
        t_end = start + n_train
        v_end = t_end + n_val

        train_idx.extend(range(start, t_end))
        if val_size > 0:
            val_idx.extend(range(t_end, v_end))
        test_idx.extend(range(v_end, end))

    train_idx = np.array(train_idx)
    val_idx   = np.array(val_idx)
    test_idx  = np.array(test_idx)

    # --- Verify no leakage ---
    assert len(set(train_idx) & set(test_idx)) == 0, "Train/Test leakage!"
    if val_size > 0:
        assert len(set(train_idx) & set(val_idx)) == 0, "Train/Val leakage!"
        assert len(set(val_idx)   & set(test_idx)) == 0, "Val/Test leakage!"

    X_train = X[train_idx]
    X_test  = X[test_idx]
    y_train = y[train_idx]
    y_test  = y[test_idx]

    # --- Verify all labels present ---
    for split_name, split_y in [("train", y_train), ("test", y_test)]:
        missing = all_labels - set(np.unique(split_y))
        if missing:
            raise ValueError(f"Labels {missing} missing from {split_name}!")

    # --- Report ---
    total = len(y)
    def dist(yy): return {int(l): int(np.sum(yy == l)) for l in sorted(all_labels)}

    if val_size > 0:
        X_val  = X[val_idx]
        y_val  = y[val_idx]
        missing_val = all_labels - set(np.unique(y_val))
        if missing_val:
            raise ValueError(f"Labels {missing_val} missing from val!")
        print(f"Cycle-boundary temporal split successful")
        print(f"  Train: {len(train_idx)} ({len(train_idx)/total:.1%}) | {dist(y_train)}")
        print(f"  Val:   {len(val_idx)}   ({len(val_idx)/total:.1%})  | {dist(y_val)}")
        print(f"  Test:  {len(test_idx)}  ({len(test_idx)/total:.1%}) | {dist(y_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        print(f"Cycle-boundary temporal split successful")
        print(f"  Train: {len(train_idx)} ({len(train_idx)/total:.1%}) | {dist(y_train)}")
        print(f"  Test:  {len(test_idx)}  ({len(test_idx)/total:.1%}) | {dist(y_test)}")
        return X_train, X_test, y_train, y_test