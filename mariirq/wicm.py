import numpy as np
from torch.utils.data import Dataset


class WICMDataset(Dataset):
    def __init__(self, X, labels, args):
        self.X = X
        self.labels = labels
        self.args = args

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        matrix, last_idx = self.process_data(self.X[index])
        return (matrix, last_idx), self.labels[index]

    def process_data(self, data):
        values = data[: self.args.seq_len, 1].astype(np.float32)
        matrix = build_wicm_matrix(
            values,
            window_size=self.args.wicm_window_size,
            stride=self.args.wicm_window_stride,
            high_io_threshold=self.args.high_io_threshold,
        )

        matrix = matrix.reshape(1, *matrix.shape)
        if self.args.log_transform:
            matrix = np.log1p(matrix)

        return matrix.astype(np.float32), matrix.shape[-1] - 1


def build_wicm_matrix(values, window_size=20, stride=20, high_io_threshold=800.0):
    if len(values) < window_size:
        raise ValueError("Trace length must be at least one WICM window")

    num_windows = (len(values) - window_size) // stride + 1
    matrix = np.zeros((4, num_windows), dtype=np.float32)

    for window_idx in range(num_windows):
        start = window_idx * stride
        window = values[start : start + window_size]

        matrix[0, window_idx] = np.sum(window)
        matrix[1, window_idx] = np.sum(window > high_io_threshold)
        matrix[2, window_idx] = np.std(window)

        deltas = np.abs(np.diff(window))
        matrix[3, window_idx] = np.max(deltas) if deltas.size else 0.0

    return matrix
