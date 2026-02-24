# src/data/io.py
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Tuple


def find_repo_root(start: Path | None = None) -> Path:
    """
    Walks up from `start` (or current working directory) until it finds a folder
    that contains 'data' and 'src'. Returns that folder as repo root.
    """
    p = (start or Path.cwd()).resolve()

    for _ in range(10):
        if (p / "data").exists() and (p / "src").exists():
            return p
        if p.parent == p:
            break
        p = p.parent

    raise FileNotFoundError(
        "Repo root not found. Make sure you're running inside the repository."
    )


def load_data_sliding(pkl_rel_path: str = "data/data_sliding.pkl") -> Tuple[Any, Any]:
    """
    Loads (ed_data, ed_labels) from a pickle file under the repo root.
    Default: data/data_sliding.pkl
    """
    repo_root = find_repo_root()
    pkl_path = repo_root / pkl_rel_path

    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Pickle not found: {pkl_path}\n"
            "Run scripts/download_dataset.py to generate it."
        )

    with open(pkl_path, "rb") as f:
        ed_data, ed_labels = pickle.load(f)

    return ed_data, ed_labels