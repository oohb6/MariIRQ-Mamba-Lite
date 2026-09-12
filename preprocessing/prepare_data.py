import argparse
import glob
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Convert CPU interrupt traces to MariIRQ NPZ format")
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("data/processed/maritime"))
    parser.add_argument("--seq_len", type=int, default=5000)
    parser.add_argument("--sample_interval", type=float, default=0.001)
    parser.add_argument("--scale_min", type=float, default=0.0)
    parser.add_argument("--scale_max", type=float, default=1500.0)
    return parser.parse_args()


def pad_sequence(sequence, length):
    if len(sequence) >= length:
        return sequence[:length]
    return np.pad(sequence, (0, length - len(sequence)), "constant", constant_values=0.0)


def load_from_npz(path):
    data = np.load(path, allow_pickle=True)
    traces = data["x"] if "x" in data else data["X"]
    if "y" not in data:
        raise KeyError(f"{path} does not contain a 'y' array")
    return traces, data["y"]


def load_from_pkl(filepaths):
    traces = []
    labels = []

    for filepath in tqdm(filepaths, desc="Loading pkl files"):
        with open(filepath, "rb") as handle:
            while True:
                try:
                    item = pickle.load(handle)
                except EOFError:
                    break

                trace_data, label = item[0], item[1]
                if isinstance(trace_data[0], list):
                    traces.extend(trace_data)
                else:
                    traces.append(trace_data)
                labels.append(label)

    traces = np.asarray(traces)
    if len(labels) == 0:
        raise ValueError("No samples were loaded")

    if isinstance(labels[0], str):
        domains = sorted(set(labels))
        mapping = {domain: idx for idx, domain in enumerate(domains)}
        labels = np.asarray([mapping[label] for label in labels])
    else:
        labels = np.asarray(labels)

    return traces, labels


def load_data(input_path):
    if input_path.is_file():
        if input_path.suffix == ".npz":
            return load_from_npz(input_path)
        if input_path.suffix == ".pkl":
            return load_from_pkl([input_path])
        raise ValueError(f"Unsupported input file: {input_path}")

    if input_path.is_dir():
        data_npz = input_path / "data.npz"
        if data_npz.exists():
            return load_from_npz(data_npz)

        pkl_files = sorted(glob.glob(str(input_path / "*.pkl")))
        if not pkl_files:
            raise FileNotFoundError(f"No .npz or .pkl data found under {input_path}")
        return load_from_pkl(pkl_files)

    raise FileNotFoundError(input_path)


def convert_traces(traces, seq_len, sample_interval, scale_min, scale_max):
    values = np.asarray(traces).reshape(-1)
    values = values[values != 0]
    global_min = float(values.min()) if values.size else 0.0
    global_max = float(values.max()) if values.size else 1.0
    if global_max == global_min:
        global_max = global_min + 1.0

    converted = []
    for trace in tqdm(traces, desc="Converting traces"):
        trace = np.asarray(trace, dtype=np.float32)
        padded = pad_sequence(trace, seq_len).astype(np.float32)

        mask = padded != 0
        padded[mask] = (
            (padded[mask] - global_min)
            / (global_max - global_min)
            * (scale_max - scale_min)
            + scale_min
        )

        timestamps = np.arange(seq_len, dtype=np.float32) * sample_interval
        valid_length = min(len(trace), seq_len)
        timestamps[valid_length:] = 0.0
        timestamps[:valid_length][timestamps[:valid_length] == 0.0] = 1e-6

        converted.append(np.stack([timestamps, padded], axis=-1))

    return np.asarray(converted, dtype=np.float32)


def normalize_labels(labels):
    labels = np.asarray(labels, dtype=np.int64)
    unique = np.unique(labels)
    mapping = {old: new for new, old in enumerate(sorted(unique.tolist()))}
    return np.asarray([mapping[int(label)] for label in labels], dtype=np.int64)


def main():
    args = parse_args()
    traces, labels = load_data(args.input_path)
    X = convert_traces(
        traces,
        args.seq_len,
        args.sample_interval,
        args.scale_min,
        args.scale_max,
    )
    y = normalize_labels(labels)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / "data.npz"
    np.savez_compressed(output_file, X=X, y=y)

    print(f"Saved: {output_file}")
    print(f"X: {X.shape} | y: {y.shape} | classes: {len(np.unique(y))}")


if __name__ == "__main__":
    main()
