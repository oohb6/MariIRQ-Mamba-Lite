# MariIRQ — E0/V2 data preparation, training, and evaluation

**Scope:** This repository contains the source code required to prepare raw IRQ traces, train MariIRQ-Lite and MariIRQ-Adapt, and evaluate closed-world classification on both the in-domain test split and separately collected test sets. It does not distribute raw traces, saved model weights, cached matrices, or paper results.

- **Lite:** extracts WICM **online**, without precomputing a Lite feature cache.
- **Adapt:** precomputes and verifies **WTCM for training and validation**; `test_adapt.py` calculates test features online.
- **E0 / V2:** use the same closed-world raw data and frozen sample split; **V2 is independently processed and trained**. Traces are 5 seconds / 5000 samples.

## 1. Where to put your datasets

No dataset folder is tied to a collection date, device, or specific filename. Use the generic directories below, and name each **test condition** after what was collected.

```text
MariIRQ-Mamba-Lite/
├── datasets/                           # User-supplied; ignored by Git
│   ├── raw/
│   │   ├── closed_world/               # Raw known-site PKL files for fitting + CW split
│   │   │   ├── site_a.pkl
│   │   │   ├── site_b.pkl
│   │   │   └── ...
│   │   └── evaluation/                 # Separately collected known-site test sets
│   │       ├── cross_time_day1/        # Each condition contains raw site PKL files
│   │       ├── cross_time_day2/
│   │       ├── background_wps/
│   │       ├── background_video/
│   │       ├── background_stress80/
│   │       ├── browser_version_92/
│   │       ├── browser_version_120/
│   │       └── other_condition/
│   └── references/                     # Optional previously frozen mapping/split files
│       ├── mapping.json
│       └── split_indices.npz
├── data/processed/                     # Generated locally; ignored by Git
│   ├── chrome_cw/                       # E0: train/valid/test + frozen metadata
│   │   ├── train.npz
│   │   ├── valid.npz
│   │   ├── test.npz
│   │   ├── mapping.json
│   │   ├── scaler.json
│   │   ├── split_indices.npz
│   │   └── wtcm_cache/                 # Adapt TRAIN/VALID cache only
│   ├── robust_v2/chrome_cw/            # V2: train/valid/test + robust_v2.json
│   │   └── wtcm_cache/                 # Adapt TRAIN/VALID cache only
│   ├── e0_tests/                        # E0 cross-environment NPZ per condition
│   │   └── background_wps/test.npz
│   └── v2_tests/                        # V2 cross-environment NPZ per condition
│       └── background_wps/test.npz
├── mariirq/                             # Required model / feature implementations
├── tools/                               # Data preparation, cache, experiment driver
├── scripts/run_e0_v2_cached.sh
├── train_lite.py
├── train_adapt_cached.py
├── test_lite.py
├── test_adapt.py
└── README.md
```

The names `cross_time_day1`, `background_wps`, and `browser_version_92` are **examples**, not required dataset identities. Choose arbitrary simple folder names and reuse exactly the same names for raw tests, prepared E0 tests, prepared V2 tests, and `--conditions`. Keep raw known-site evaluation datasets **separate** from the raw closed-world dataset used for fitting and train/valid/CW-test splitting.

> All data and generated artifacts stay local. `.gitignore` protects against accidental `git add`, but does not prevent deliberate GitHub web uploads. Never upload raw traces or sensitive collection information to the repository.

## 2. Requirements

Use Linux, Python 3.10, a CUDA-capable PyTorch installation, compatible `mamba_ssm` and `causal_conv1d`, then install `requirements.txt`. The verified environment used PyTorch 2.2.2+cu118 and mamba_ssm 2.2.2; binary wheels must match your own GPU/CUDA/Torch versions.

## 3. Prepare the E0 and V2 closed-world train/valid/CW-test splits

First copy the raw closed-world IRQ `.pkl` files to `datasets/raw/closed_world/` (or supply a different location via `--input` / `--v2-raw`). These are the *same* raw input files for E0 and V2, including all samples used for the original CW train/valid/test split.

```bash
# E0: create frozen class mapping, train-only scaling and 81/9/10 data split
python tools/prepare_baseline.py fit \
  --input datasets/raw/closed_world \
  --output data/processed/chrome_cw \
  --seed 2024

# V2: independently transform the SAME raw data using the frozen E0 reference
bash scripts/run_e0_v2_cached.sh --variant v2 --phase prepare --alpha 1.0
```

For **exact reproduction of an existing split**, use the original saved class mapping and split indices instead of generating new ones:

```bash
python tools/prepare_baseline.py fit \
  --input datasets/raw/closed_world \
  --output data/processed/chrome_cw \
  --mapping datasets/references/mapping.json \
  --indices datasets/references/split_indices.npz
```

Use **one E0 `fit` command**, not both. Only use the second when the reference files actually exist. The original sample order must be preserved; the V2 fitting code checks that its raw source matches E0. If prepared E0/V2 NPZs already exist and have the correct frozen mapping/split, **skip preparation** rather than overwriting them. Each NPZ contains `X: (N,5000,2)` and integer `y: (N,)`.

## 4. Prepare additional test datasets (Cross-time / Background / browser versions)

Each test condition uses its **own** raw PKL folder under `datasets/raw/evaluation/<condition>/`. Run both transformations against *training-frozen* references; do not calculate new training scalers or V2 alignment anchors on the test data. The following example uses `background_wps`, but the commands apply to any valid condition name.

```bash
COND=background_wps

# E0 test: apply only the saved E0 class mapping and training scaler
python tools/prepare_baseline.py test \
  --input "datasets/raw/evaluation/$COND" \
  --reference data/processed/chrome_cw \
  --output "data/processed/e0_tests/$COND"

# V2 test: apply only the frozen E0 mapping/scaler and V2 training reference
python tools/prepare_robust_v2.py test \
  --raw "datasets/raw/evaluation/$COND" \
  --reference data/processed/robust_v2/chrome_cw \
  --baseline-test "data/processed/e0_tests/$COND/test.npz" \
  --out "data/processed/v2_tests/$COND"
```

Repeat with `COND=cross_time_day1`, `COND=background_stress80`, `COND=browser_version_92`, etc., only for test datasets you **actually have**. Each creates `test.npz` with the same sample ordering for E0 and V2. Both test transformations require known closed-world class labels consistent with the *frozen 100-class mapping*; this pipeline's accuracy/F1 test is **not** an open-world rejection evaluation. If a test dataset has subpage URLs or different label strings, establish a correct and consistent label mapping first rather than silently re-indexing it.

## 5. Generate and validate Adapt WTCM caches

```bash
bash scripts/run_e0_v2_cached.sh \
  --variant both --phase cache --models adapt --cache-workers 8

bash scripts/run_e0_v2_cached.sh \
  --variant both --phase check --models lite adapt
```

Caches are generated **only** for the CW `train` and `valid` splits under each variant's `wtcm_cache/` directory. Lite does **not** need a WICM cache. A 8100/900 sample split needs approximately 5.9 GiB of cache space **per variant**, plus free-space headroom.

## 6. Train E0/V2 Lite and Adapt (30-epoch example)

```bash
bash scripts/run_e0_v2_cached.sh \
  --variant both --phase all --models lite adapt \
  --seeds 2024 --epochs 30 --patience 31 \
  --batch-size 32 --adapt-workers 2 \
  --experiment pilot_30ep_e0_v2
```

`--phase all` prepares/checks the CW datasets, reuses valid Adapt caches, trains the four models, evaluates the *within-split* `cw_test`, and creates a local CSV. Use `--phase train` rather than `all` if independent test NPZs are not ready. Models are evaluated using the saved best-validation-F1 checkpoint, not necessarily the last epoch.

## 7. Evaluate existing checkpoints on CW and additional test conditions

```bash
# CW test (the held-out 10% split of closed-world raw samples)
bash scripts/run_e0_v2_cached.sh \
  --variant both --phase eval --models lite adapt \
  --seeds 2024 --experiment pilot_30ep_e0_v2

# Test on a separately prepared condition
bash scripts/run_e0_v2_cached.sh \
  --variant both --phase eval --models lite adapt \
  --conditions background_wps \
  --seeds 2024 --experiment pilot_30ep_e0_v2

# Multiple conditions can be evaluated together (once their NPZ files exist)
bash scripts/run_e0_v2_cached.sh \
  --variant both --phase eval --models lite adapt \
  --conditions cross_time_day1 background_wps browser_version_92 \
  --seeds 2024 --experiment pilot_30ep_e0_v2
```

Default locations for custom conditions are `data/processed/e0_tests/<condition>/test.npz` and `data/processed/v2_tests/<condition>/test.npz`. Override with `--e0-test-root` and `--v2-test-root` if needed. `cw_test` is special and reads the *variant's own* `test.npz`.

`--phase summary` also accepts the same `--conditions` to summarize previously generated JSON reports. Keep `--experiment` equal to the training experiment name. Results and checkpoints remain local. No accuracy tables are committed to the repository.

## 8. Important reproducibility notes

- The closed-world source is fixed to **100 known classes** by this preparation code; it does *not* assume any special site domain or collection date.
- Do not mix raw collection sessions into one evaluation folder when they are intended to be different conditions.
- Never refit E0 or V2 using test-data statistics. For V2, per-trace alignment uses frozen train-derived references.
- New devices, times and browsers may have fewer/missing sites. Their labels still must belong to the frozen known-class mapping for this evaluation path.
- V2 metadata stores the absolute E0 reference path for integrity checks. If moving prepared files to another server, either rerun V2 `fit` with the same original raw input and frozen E0 reference, or keep the required reference accessible at the recorded path.
- You may store datasets outside the Git clone: pass `--input` to `prepare_baseline.py`, `--v2-raw` to the experiment driver, and `--e0-test-root` / `--v2-test-root` when testing.
