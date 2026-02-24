#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import pickle
import shutil

REPO_ROOT = Path(__file__).resolve().parents[1]

TEMP_CACHE = REPO_ROOT / "data" / "_tmp_kagglehub"
os.environ["KAGGLEHUB_CACHE"] = str(TEMP_CACHE)

import kagglehub

DATASET = "tolgareis/real-appliance-power-signatures-dataset"

def main():
    TEMP_CACHE.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(kagglehub.dataset_download(DATASET))

    pkl_files = list(dataset_path.rglob("data_sliding.pkl"))
    if not pkl_files:
        raise FileNotFoundError("data_sliding.pkl not found in dataset")

    pkl_path = pkl_files[0]
    print("Found:", pkl_path)

    target = REPO_ROOT / "data" / "data_sliding.pkl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(pkl_path.read_bytes())
    print("Saved to:", target)

    shutil.rmtree(TEMP_CACHE, ignore_errors=True)
    print("Temporary cache removed")

    with open(target, "rb") as f:
        ed_data, ed_labels = pickle.load(f)

    print("Loaded:", ed_data.shape, ed_labels.shape)

if __name__ == "__main__":
    main()