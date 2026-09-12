# Data

Raw datasets are not included in the repository.

After preprocessing, place the paper split under:

```text
data/processed/maritime/
├── data.npz
├── train.npz
├── valid.npz
└── test.npz
```

Each NPZ file must contain arrays named `X` and `y`.
