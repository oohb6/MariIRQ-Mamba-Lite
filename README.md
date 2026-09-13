# MariIRQ-Mamba-Lite

Official implementation of:

**MariIRQ-Mamba-Lite: Lightweight CPU Interrupt Side-Channel Fingerprinting for Maritime Websites**

MariIRQ-Mamba-Lite contains two stages:

- **MariIRQ-Adapt**: CPU interrupt traces → IRQ-WTCM → Mamba2-based classifier.
- **MariIRQ-Mamba-Lite**: CPU interrupt traces → WICM → bidirectional Mamba2 → classifier.

## Repository Structure

```text
MariIRQ-Mamba-Lite/
├── mariirq/
│   ├── irq_wtcm.py
│   ├── wicm.py
│   ├── countmamba_backbone.py
│   ├── bidirectional_mamba2.py
│   ├── mariirq_mamba_lite.py
│   ├── utils.py
│   └── __init__.py
├── preprocessing/
│   ├── prepare_data.py
│   ├── split_dataset.py
│   └── __init__.py
├── train_adapt.py
├── test_adapt.py
├── train_lite.py
├── test_lite.py
├── data/
├── checkpoints/
├── requirements.txt
└── README.md
```

## Environment

The experiments were conducted with:

- Python 3.10
- PyTorch 2.1.2 + CUDA 11.8
- Mamba-SSM 2.2.2
- causal-conv1d 1.4.0
- NVIDIA GeForce RTX 3080 Ti (12 GB)

Install the required packages:

```bash
pip install -r requirements.txt
```

Mamba-SSM and causal-conv1d should be installed with versions compatible with the local PyTorch and CUDA environment.

## Dataset

The experimental dataset contains:

- 100 maritime websites
- 100 traces per website
- 10,000 traces in total
- 5 s per trace
- 5000 interrupt samples per trace
- approximately 1 ms sampling interval
- 81 / 9 / 10 traces per class for training / validation / testing

Each sample is stored as:

```text
X[i, :, 0] = timestamp
X[i, :, 1] = CPU interrupt value
y[i]       = website label
```

## Data Preparation

Convert the raw traces to NPZ format:

```bash
python preprocessing/prepare_data.py \
  --input_path /path/to/raw_data \
  --output_dir data/processed/maritime
```

Create the training, validation, and test splits:

```bash
python preprocessing/split_dataset.py \
  --data_file data/processed/maritime/data.npz
```

The processed dataset is expected to contain:

```text
data/processed/maritime/
├── train.npz
├── valid.npz
└── test.npz
```

## MariIRQ-Adapt

MariIRQ-Adapt converts CPU interrupt traces into the IRQ-WTCM representation and applies a Mamba2-based classifier.

### IRQ-WTCM Configuration

| Parameter | Value |
|---|---:|
| Trace length | 5000 |
| Matrix size | 35 × 4998 |
| Amplitude-bin width | 75 |
| Maximum bin index | 32 |
| AMP_DELTA threshold | 75 |
| CLUSTER window | 20 |
| Log transform | `log1p` |

The 35 rows consist of 33 amplitude-bin rows together with the AMP_DELTA and CLUSTER descriptors.

### Training

```bash
python train_adapt.py
```

The default configuration uses:

```text
Embedding dimension: 256
Mamba2 depth:        3
Batch size:          128
Learning rate:       2e-3
Epochs:              50
Seed:                2024
```

### Evaluation

```bash
python test_adapt.py
```

The default checkpoint path is:

```text
checkpoints/mariirq_adapt/max_f1.pth
```

## MariIRQ-Mamba-Lite

MariIRQ-Mamba-Lite compresses each interrupt trace into the Windowed Interrupt Characteristic Matrix (WICM).

Each 20 ms window contains four features:

- `TOTAL`: sum of interrupt values
- `HIGH_IO`: number of samples above the high-I/O threshold
- `STD`: within-window standard deviation
- `MAX_DELTA`: maximum adjacent absolute change

The final WICM representation is:

```text
4 × 250
```

### Model Configuration

| Parameter | Value |
|---|---:|
| PatchEmbed kernel | `(4, 1)` |
| Embedding dimension | 192 |
| Bidirectional Mamba2 blocks | 3 |
| Head dimension | 32 |
| DropPath | 0.1 |
| WICM window size | 20 |
| WICM stride | 20 |
| HIGH_IO threshold | 800 |

### Training

```bash
python train_lite.py
```

The default configuration uses:

```text
Embedding dimension: 192
Mamba2 depth:        3
Head dimension:      32
Batch size:          128
Learning rate:       3e-3
Epochs:              50
Seed:                2024
```

### Evaluation

```bash
python test_lite.py
```

The default checkpoint path is:

```text
checkpoints/mariirq_mamba_lite/max_f1.pth
```

## Results

### Classification Performance

| Method | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) |
|---|---:|---:|---:|---:|
| CountMamba | 24.00 | 26.54 | 24.00 | 23.42 |
| BiggerFish CNN-LSTM | 87.00 | 88.06 | 87.00 | 86.83 |
| MariIRQ-Adapt | **93.40** | **93.84** | **93.40** | **93.38** |
| MariIRQ-Mamba-Lite | 91.70 | 92.23 | 91.70 | 91.67 |

### Efficiency

| Metric | MariIRQ-Adapt | MariIRQ-Mamba-Lite |
|---|---:|---:|
| Input size | 35 × 4998 | 4 × 250 |
| Input elements | 174,930 | 1,000 |
| Parameters | 2.959 M | 2.279 M |
| Checkpoint size | 11.310 MiB | 8.722 MiB |
| Training time | 6.27 h | 0.48 h |
| Peak training GPU memory | 10,430.23 MiB | 2,801.78 MiB |

## Attribution

MariIRQ-Adapt is developed based on the CountMamba framework.

If you use the Stage-I implementation, please also cite the original CountMamba work:

```bibtex
@inproceedings{deng2025countmamba,
  title={CountMamba: A Generalized Website Fingerprinting Attack via Coarse-Grained Representation and Fine-Grained Prediction},
  author={Deng, X. and Zhao, R. and Wang, Y. and Zhan, M. and Xue, Z. and Wang, Y.},
  booktitle={2025 IEEE Symposium on Security and Privacy},
  pages={1419--1437},
  year={2025}
}
```

CountMamba project:

https://github.com/SJTU-dxw/CountMamba-WF

## Citation

The citation for MariIRQ-Mamba-Lite will be added after publication.

## License

A software license will be added before the public release.
