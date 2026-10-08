"""
download_data.py
----------------
Helper script to download the Luke Barousse data_jobs dataset
from Hugging Face and save it to data/jobs.csv.

This uses the Hugging Face `datasets` library (free, no API key).

Run:
    python download_data.py

If the datasets library is not installed, it will be installed automatically.
"""

import os
import subprocess
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATA_PATH = os.path.join(DATA_DIR, "jobs.csv")

os.makedirs(DATA_DIR, exist_ok=True)


def install_datasets_lib():
    """Install huggingface datasets library if not present."""
    try:
        import datasets  # noqa
        print("[OK] huggingface 'datasets' library already installed.")
    except ImportError:
        print("[INFO] Installing 'datasets' library ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
        print("[OK] Installed 'datasets' library.")


def download_csv():
    """Download the data_jobs dataset and save as CSV."""
    if os.path.exists(DATA_PATH):
        print(f"[OK] Dataset already exists at: {DATA_PATH}")
        return

    print("[INFO] Downloading dataset from Hugging Face ...")
    print("       This may take a few minutes on first download (~230 MB).")

    from datasets import load_dataset  # noqa
    dataset = load_dataset("lukebarousse/data_jobs", split="train")

    print(f"[INFO] Dataset loaded. Total rows: {len(dataset)}")
    print("[INFO] Converting to CSV ...")

    df = dataset.to_pandas()
    df.to_csv(DATA_PATH, index=False)

    print(f"[OK] Saved to: {DATA_PATH}")
    print(f"     Rows: {len(df)}, Columns: {list(df.columns)}")


if __name__ == "__main__":
    install_datasets_lib()
    download_csv()
    print("\n[OK] Dataset ready! You can now run:  streamlit run app.py")
