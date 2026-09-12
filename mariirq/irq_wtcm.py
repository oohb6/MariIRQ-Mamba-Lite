import math

import numpy as np
from torch.utils.data import Dataset


class IRQCountDataset(Dataset):
    def __init__(self, X, labels, args):
        self.X = X
        self.labels = labels
        self.args = args

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        matrix, current_index = self.process_data(self.X[index])
        return (matrix, current_index), self.labels[index]

    def process_data(self, data):
        time = pad_sequence(data[:, 0].astype(np.float32), self.args.seq_len)
        values = pad_sequence(data[:, 1].astype(np.float32), self.args.seq_len)

        matrix, current_index = build_irq_wtcm(
            values,
            time,
            maximum_cell_number=self.args.maximum_cell_number,
            max_matrix_len=self.args.max_matrix_len,
            maximum_load_time=self.args.maximum_load_time,
            bin_size=self.args.irq_bin_size,
            cluster_window=self.args.irq_cluster_window,
            amp_threshold=self.args.irq_amp_threshold,
        )

        rows = self.args.maximum_cell_number + 3
        matrix = matrix.reshape(1, rows, self.args.max_matrix_len)
        if self.args.log_transform:
            matrix = np.log1p(matrix)

        return matrix.astype(np.float32), current_index


def pad_sequence(sequence, length):
    if len(sequence) >= length:
        return sequence[:length]
    return np.pad(sequence, (0, length - len(sequence)), "constant", constant_values=0.0)


def build_irq_wtcm(
    values,
    time,
    maximum_cell_number=32,
    max_matrix_len=4998,
    maximum_load_time=5.0,
    bin_size=75.0,
    cluster_window=20,
    amp_threshold=None,
):
    C = maximum_cell_number
    cols = max_matrix_len
    feature = np.zeros((C + 3, cols), dtype=np.float32)
    window_width = maximum_load_time / cols
    bin_size = max(float(bin_size), 1e-6)
    threshold = bin_size if amp_threshold is None else float(amp_threshold)

    current_index = 0
    column_values = []
    rolling_values = []

    for value, timestamp in zip(values, time):
        if timestamp == 0 and value == 0:
            break

        row = int(min(math.floor(value / bin_size), C))
        col = min(max(math.floor(timestamp / window_width), 0), cols - 1)
        feature[row, col] += 1

        rolling_values.append(value)
        if len(rolling_values) > cluster_window:
            rolling_values.pop(0)

        if col != current_index:
            previous_mean = np.mean(column_values) if column_values else 0.0
            feature[C + 1, col] = abs(value - previous_mean)

            recent = np.asarray(rolling_values, dtype=np.float32)
            if recent.size > 1:
                cluster_count = int(np.sum(np.abs(np.diff(recent)) > threshold) + 1)
            else:
                cluster_count = int(recent.size > 0)
            feature[C + 2, current_index] = cluster_count

            current_index = col
            column_values = [value]
        else:
            column_values.append(value)

    recent = np.asarray(rolling_values, dtype=np.float32)
    if recent.size > 1:
        cluster_count = int(np.sum(np.abs(np.diff(recent)) > threshold) + 1)
    else:
        cluster_count = int(recent.size > 0)
    feature[C + 2, current_index] = cluster_count

    return feature, current_index
