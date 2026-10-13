# MariIRQ: E0/V2 Data Processing, Training, and Testing

This repository provides the code for processing 5-second IRQ traces (5000 samples), training **MariIRQ-Lite** and **MariIRQ-Adapt**, and testing the models on closed-world and cross-environment datasets.

- **Lite:** computes WICM features online during training and testing.
- **Adapt:** uses precomputed WTCM features for training and validation; test features are computed by the test script.
- **E0 and V2:** use the same closed-world source and split, but are processed and trained separately.

## 1. Dataset layout

Place the raw `.pkl` files in the following directories after cloning the repository. Folder names for test conditions are examples and can be adjusted consistently.

```text
MariIRQ-Mamba-Lite/
├── datasets/
│   └── raw/
│       ├── closed_world/                # Source data for CW train/valid/test
│       │   ├── site_a.pkl
│       │   ├── site_b.pkl
│       │   └── ...
│       └── evaluation/                  # Independently collected test datasets
│           ├── cross_time_day1/
│           ├── cross_time_day2/
│           ├── background_wps/
│           ├── background_video/
│           ├── background_stress80/
│           ├── browser_version_92/
│           └── browser_version_120/
├── data/
│   └── processed/                       # Created by the preprocessing scripts
│       ├── chrome_cw/                   # E0 train/valid/test
│       │   └── wtcm_cache/
│       ├── robust_v2/chrome_cw/         # V2 train/valid/test
│       │   └── wtcm_cache/
│       ├── e0_tests/                    # E0 external test NPZ files
│       └── v2_tests/                    # V2 external test NPZ files
├── mariirq/
├── scripts/
├── tools/
├── train_lite.py
├── train_adapt_cached.py
├── test_lite.py
└── test_adapt.py
```

Keep each external test condition in its own subdirectory under `datasets/raw/evaluation/`. The datasets and processed files are **not** included in this repository.

## 2. Environment

Use Linux with Python 3.10, CUDA-compatible PyTorch, `mamba_ssm`, and `causal_conv1d`, then install the remaining requirements:

```bash
pip install -r requirements.txt
```

The CUDA, PyTorch, Mamba and causal-conv1d builds must be mutually compatible.

## 3. Prepare E0 and V2 datasets

Prepare the E0 closed-world data from the raw `.pkl` files:

```bash
python tools/prepare_baseline.py fit \
  --input datasets/raw/closed_world \
  --output data/processed/chrome_cw \
  --seed 2024
```

Then prepare V2 from the **same raw files** and E0's saved training reference:

```bash
bash scripts/run_e0_v2_cached.sh \
  --variant v2 --phase prepare --alpha 1.0
```

Both versions produce `train.npz`, `valid.npz`, and `test.npz`. To reproduce an existing fixed split exactly, pass its saved class mapping and split indices to `prepare_baseline.py fit` using `--mapping` and `--indices`.

## 4. Prepare additional test datasets

For example, to process `background_wps` for both E0 and V2:

```bash
COND=background_wps

python tools/prepare_baseline.py test \
  --input "datasets/raw/evaluation/$COND" \
  --reference data/processed/chrome_cw \
  --output "data/processed/e0_tests/$COND"

python tools/prepare_robust_v2.py test \
  --raw "datasets/raw/evaluation/$COND" \
  --reference data/processed/robust_v2/chrome_cw \
  --baseline-test "data/processed/e0_tests/$COND/test.npz" \
  --out "data/processed/v2_tests/$COND"
```

Replace `COND` with another test condition, such as `cross_time_day1` or `browser_version_92`. External closed-world test labels must correspond to the classes in the saved training mapping.

## 5. Prepare Adapt WTCM features

Generate and check WTCM training/validation caches for E0 and V2:

```bash
bash scripts/run_e0_v2_cached.sh \
  --variant both --phase cache --models adapt --cache-workers 8
```

**Lite uses online WICM; no Lite cache is required.**

## 6. Train E0/V2 Lite and Adapt (50 epochs, early stopping)

Run all four configurations with five random seeds:

```bash
bash scripts/run_e0_v2_cached.sh \
  --variant both \
  --phase all \
  --models lite adapt \
  --seeds 2024 2025 2026 2027 2028 \
  --epochs 50 \
  --patience 10 \
  --batch-size 32 \
  --adapt-workers 2 \
  --experiment e0_v2_50ep_p10
```

Training runs for **up to 50 epochs** and stops early if validation F1 does not improve for **10 consecutive epochs**. Each model is tested using the checkpoint with the best validation F1. The `all` phase also evaluates the held-out CW test split and writes a local summary.

## 7. Test trained models

For the held-out closed-world test set:

```bash
bash scripts/run_e0_v2_cached.sh \
  --variant both --phase eval --models lite adapt \
  --seeds 2024 2025 2026 2027 2028 \
  --batch-size 32 --adapt-workers 2 \
  --experiment e0_v2_50ep_p10
```

For prepared external test datasets:

```bash
bash scripts/run_e0_v2_cached.sh \
  --variant both --phase eval --models lite adapt \
  --conditions cross_time_day1 background_wps browser_version_92 \
  --seeds 2024 2025 2026 2027 2028 \
  --batch-size 32 --adapt-workers 2 \
  --experiment e0_v2_50ep_p10
```

Use only condition names for which the E0 and V2 test NPZ files have already been prepared. Results are saved locally under `results/e0_v2_cached/<experiment>/`; model checkpoints are saved under `checkpoints/e0_v2_cached/<experiment>/`.
