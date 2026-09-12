# MariIRQ-Mamba-Lite

Official code release for:

**MariIRQ-Mamba-Lite: Lightweight CPU Interrupt Side-Channel Fingerprinting for Maritime Websites**

The repository contains the two stages reported in the paper:

- **MariIRQ-Adapt**: CPU interrupt trace → IRQ-WTCM → CountMamba-style Mamba2 backbone.
- **MariIRQ-Mamba-Lite**: CPU interrupt trace → WICM → single-window PatchEmbed → three bidirectional Mamba2 blocks → classifier.

The public code is intentionally scoped to the 100-class closed-world, single-tab setting used in the paper. Development-only branches and long inline experiment notes were removed from the source files and consolidated here.

## Repository structure

```text
MariIRQ-Mamba-Lite/
├── mariirq/
│   ├── irq_wtcm.py                 # IRQ-WTCM construction
│   ├── wicm.py                     # WICM construction
│   ├── countmamba_backbone.py      # MariIRQ-Adapt backbone
│   ├── bidirectional_mamba2.py     # Bidirectional Mamba2 block
│   ├── mariirq_mamba_lite.py       # Final lightweight model
│   └── utils.py
├── preprocessing/
│   ├── prepare_data.py             # Raw BiggerFish-style traces → NPZ
│   └── split_dataset.py            # 81/9/10 split per class for balanced data
├── train_adapt.py
├── test_adapt.py
├── train_lite.py
├── test_lite.py
├── scripts/
├── data/
├── checkpoints/
└── results/
```


## Development-file mapping

| Development file | Clean release |
|---|---|
| `convert_biggerfish_to_npz.py` | `preprocessing/prepare_data.py` |
| `dataset_split.py` | `preprocessing/split_dataset.py` |
| `dataset_irq_wf.py` | `mariirq/irq_wtcm.py` |
| `dataset_irq_wicm.py` | `mariirq/wicm.py` |
| `model_CountMamba.py` | `mariirq/countmamba_backbone.py` |
| `model_CountMamba_Light.py` | `mariirq/mariirq_mamba_lite.py` |
| `model_mamba2_light.py` | `mariirq/bidirectional_mamba2.py` |
| `util.py` | `mariirq/utils.py` |
| `main_irq_wf.py` | `train_adapt.py` |
| `main_irq_light.py` | `train_lite.py` |

## Tested configuration

The experiments were run with Python 3.10, PyTorch 2.1.2 + CUDA 11.8, Mamba-SSM 2.2.2, and causal-conv1d 1.4.0 on an NVIDIA GeForce RTX 3080 Ti (12 GB).

Install PyTorch and the CUDA-matched Mamba/causal-conv1d wheels first, then install the remaining packages:

```bash
pip install -r requirements.txt
```

For Mamba-SSM installation details, follow the environment guidance in the original CountMamba repository and the Mamba repository for your CUDA/PyTorch combination.

## Dataset format

Each sample is stored as:

```text
X[i, :, 0] = timestamp in seconds
X[i, :, 1] = scaled CPU interrupt value
y[i]       = website class label
```

The paper setting uses:

- 100 maritime websites
- 100 traces per website
- 10,000 traces in total
- 5 s per trace
- 5000 samples per trace
- approximately 1 ms sampling interval
- 81 training / 9 validation / 10 test traces per class

The preprocessing script maps non-zero interrupt values to `[0, 1500]` using a global min-max transform, matching the provided experimental pipeline.

### Prepare data

For an NPZ file or a directory containing BiggerFish-style `.pkl` files:

```bash
python preprocessing/prepare_data.py \
  --input_path /path/to/raw_data \
  --output_dir data/processed/maritime
```

Then create the train/validation/test splits:

```bash
python preprocessing/split_dataset.py \
  --data_file data/processed/maritime/data.npz
```

Expected output:

```text
data/processed/maritime/
├── data.npz
├── train.npz
├── valid.npz
└── test.npz
```

## Stage I: MariIRQ-Adapt

### IRQ-WTCM

Final configuration:

| Parameter | Value |
|---|---:|
| Trace length | 5000 |
| Matrix size | 35 × 4998 |
| Amplitude-bin width | 75 |
| Maximum bin index | 32 |
| AMP_DELTA threshold | 75 |
| CLUSTER rolling window | 20 samples |
| Log transform | `log1p` |

The 35 rows consist of 33 amplitude-bin rows plus AMP_DELTA and CLUSTER.

### Train

```bash
python train_adapt.py
```

Equivalent explicit configuration:

```bash
python train_adapt.py \
  --data_path data/processed/maritime \
  --irq_bin_size 75 \
  --maximum_cell_number 32 \
  --irq_cluster_window 20 \
  --max_matrix_len 4998 \
  --embed_dim 256 \
  --depth 3 \
  --batch_size 128 \
  --lr 2e-3 \
  --epochs 50
```

Training uses AdamW, weight decay `0.05`, label smoothing `0.1`, cosine learning-rate decay, and seed `2024`.

### Test

```bash
python test_adapt.py
```

The default checkpoint is:

```text
checkpoints/mariirq_adapt/max_f1.pth
```

## Stage II: MariIRQ-Mamba-Lite

### WICM

Each non-overlapping 20 ms window is represented by four features:

1. `TOTAL`: sum of interrupt values.
2. `HIGH_IO`: number of samples above 800.
3. `STD`: within-window standard deviation.
4. `MAX_DELTA`: maximum adjacent absolute change.

Final representation:

```text
4 × 250 = 1000 input elements
```

### Model configuration

| Parameter | Value |
|---|---:|
| PatchEmbed kernel | `(4, 1)` |
| Embedding dimension | 192 |
| Mamba2 depth | 3 bidirectional blocks |
| Head dimension | 32 |
| DropPath | 0.1 |
| WICM window / stride | 20 / 20 |
| HIGH_IO threshold | 800 |

The final release uses the **single-window `(4, 1)` PatchEmbed**. The earlier `(4, 3)` version is an ablation variant and is not used by the final model.

### Train

```bash
python train_lite.py
```

Equivalent explicit configuration:

```bash
python train_lite.py \
  --data_path data/processed/maritime \
  --wicm_window_size 20 \
  --wicm_window_stride 20 \
  --high_io_threshold 800 \
  --embed_dim 192 \
  --headdim 32 \
  --depth 3 \
  --batch_size 128 \
  --lr 3e-3 \
  --epochs 50
```

Training uses AdamW, weight decay `0.05`, label smoothing `0.1`, cosine learning-rate decay, and seed `2024`.

### Test

```bash
python test_lite.py
```

The default checkpoint is:

```text
checkpoints/mariirq_mamba_lite/max_f1.pth
```

## Main results

| Method | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) |
|---|---:|---:|---:|---:|
| CountMamba | 24.00 | 26.54 | 24.00 | 23.42 |
| BiggerFish CNN-LSTM | 87.00 | 88.06 | 87.00 | 86.83 |
| MariIRQ-Adapt | **93.40** | **93.84** | **93.40** | **93.38** |
| MariIRQ-Mamba-Lite | 91.70 | 92.23 | 91.70 | 91.67 |

Efficiency comparison between the two proposed stages:

| Metric | MariIRQ-Adapt | MariIRQ-Mamba-Lite |
|---|---:|---:|
| Input | 35 × 4998 | 4 × 250 |
| Input elements | 174,930 | 1,000 |
| Parameters | 2.959 M | 2.279 M |
| Checkpoint size | 11.310 MiB | 8.722 MiB |
| Training time | 6.27 h | 0.48 h |
| Peak training GPU memory | 10,430.23 MiB | 2,801.78 MiB |

## Notes on the cleaned release

The GitHub-oriented version differs from the development folder only in code organization and removal of unused/development-only paths. The final paper settings were made the defaults, including:

- IRQ-WTCM bin width changed from an old development default of `100` to the final `75`.
- Lite PatchEmbed changed from the ablation `(4, 3)` configuration to the final `(4, 1)` configuration.
- The old commented-out Lite and bidirectional-Mamba implementations were removed.
- Early-stage and multi-tab CountMamba branches that are not part of this paper were removed from the public training/evaluation scripts.
- Mamba-SSM is required for reproduction; the previous GRU fallback was removed to avoid silently evaluating a different model.

## Attribution

MariIRQ-Adapt retains the CountMamba sequence-modeling backbone and adapts its input representation to CPU interrupt traces. Please also cite the original CountMamba work when using the Stage-I implementation:

```bibtex
@inproceedings{deng2025countmamba,
  title={CountMamba: A Generalized Website Fingerprinting Attack via Coarse-Grained Representation and Fine-Grained Prediction},
  author={Deng, X. and Zhao, R. and Wang, Y. and Zhan, M. and Xue, Z. and Wang, Y.},
  booktitle={2025 IEEE Symposium on Security and Privacy},
  pages={1419--1437},
  year={2025}
}
```

Project reference: https://github.com/SJTU-dxw/CountMamba-WF

## Citation

The paper citation will be added after publication.

## License

No software license has been selected in this package. Add the license chosen by the authors before public release.
