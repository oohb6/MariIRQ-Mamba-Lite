import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description="Create train/validation/test splits")
    parser.add_argument("--data_file", type=Path, default=Path("data/processed/maritime/data.npz"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or args.data_file.parent

    data = np.load(args.data_file)
    X, y = data["X"], data["y"]
    unique = np.unique(y)
    if unique.min() != 0 or len(unique) != unique.max() + 1:
        raise ValueError("Labels must be continuous and start from 0")

    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X,
        y,
        train_size=0.9,
        random_state=args.seed,
        stratify=y,
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid,
        y_train_valid,
        train_size=0.9,
        random_state=args.seed,
        stratify=y_train_valid,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "train.npz", X=X_train, y=y_train)
    np.savez_compressed(output_dir / "valid.npz", X=X_valid, y=y_valid)
    np.savez_compressed(output_dir / "test.npz", X=X_test, y=y_test)

    print(f"Train: {X_train.shape}, {y_train.shape}")
    print(f"Valid: {X_valid.shape}, {y_valid.shape}")
    print(f"Test:  {X_test.shape}, {y_test.shape}")


if __name__ == "__main__":
    main()
