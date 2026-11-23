# Cross-dataset TB screening — code to week 9

Code for the project *Diagnosing Tuberculosis in Chest X-rays from Two
Countries*: a two-directional, cross-hospital evaluation of TB screening on the
NLM Montgomery and Shenzhen chest X-ray sets, with lung-masking controls and
confidence intervals on every number.

This repository covers the work scheduled through **week 9**: Phases 2–6 of the
plan are complete, including the full intensity-correction family (CNN and
classical model, both directions) and its paired significance tests. It is built around one rule from the
methodology: the *direction* of an experiment (which set trains, which set
tests) and the *image region* (whole, lung-cropped, lung-masked) are settings in
a YAML file, not code changes. One pipeline runs every experiment.

## Status at week 9

| Phase | Weeks | Work | Status |
|-------|-------|------|--------|
| 2 | 3–4 | Data download, metadata, frozen 5-fold splits | done |
| 3 | 4–5 | EDA / shortcut audit, lung U-Net masks | done |
| 4 | 5–7 | Classical lung-field baseline, within-set CNN | done |
| 5 | 7–8 | Cross-dataset (both directions), region controls, Grad-CAM, DeLong tests | done |
| 6 | 8–9 | Intensity corrections (`histeq`, `clahe`, `zscore`) | done — CNN and classical, both directions, DeLong tests, Grad-CAM on CLAHE |

**Week 9 added:** the classical lung-field model with each correction (both
directions), the CNN Shenzhen→Montgomery direction with each correction, the
paired DeLong tests for the correction family (`phase6`), and Grad-CAM on the
CLAHE model to check whether the correction moves attention onto the lungs.

## What maps to what

| Phase | Weeks | Script | Produces |
|-------|-------|--------|----------|
| 2 | 3–4 | `00_download_data.py` | both NLM sets under `data/raw/` |
| 2 | 3–4 | `01_prepare_data.py` | `metadata.csv`, frozen 5-fold splits |
| 3 | 4–5 | `02_run_eda.py` | visual audit grid, distribution plots, metadata summary |
| 3 | 4–5 | `03_train_unet_masks.py` | lung U-Net, held-out Dice, generated Shenzhen masks |
| 4 | 5–7 | `04_build_features.py` | lung-field texture features (classical baseline) |
| 4–6 | 5–8 | `05_run_experiment.py` | results + bootstrap CIs for any experiment config |
| 4–6 | 5–8 | `06_make_results_table.py` | collated results table (CSV + Markdown) |
| 5 | 7–8 | `07_gradcam.py` | Grad-CAM grid + lung-attention share per CNN experiment |
| 5 | 7–8 | `08_compare_auc.py` | paired DeLong tests, Holm-corrected |

## Install (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -e ".[dl]"                # omit [dl] if you only need the classical path
```

On macOS / Linux use `source .venv/bin/activate` instead. The deep-learning
extra pulls in `torch`, `torchvision`, `timm` and `grad-cam`. The data-prep, EDA
and classical stages run without them. A CUDA GPU makes the CNN and U-Net runs
practical (hours on GPU instead of days on CPU).

## Data layout

Download both sets from the NLM repository (`python scripts/00_download_data.py`)
or unzip them under `data/raw/` so the tree looks like this (the code keys off
these folder names):

```
data/raw/
  MontgomerySet/
    CXR_png/              MCUCXR_0001_0.png ...
    ManualMask/leftMask/  MCUCXR_0001_0.png ...
    ManualMask/rightMask/ MCUCXR_0001_0.png ...
    ClinicalReadings/     MCUCXR_0001_0.txt ...
  ChinaSet_AllFiles/
    CXR_png/              CHNCXR_0001_0.png ...
    ClinicalReadings/     CHNCXR_0001_0.txt ...
```

The TB label is read from the last digit of each filename (`_0` normal, `_1`
TB). Age, sex and the clinical reading are parsed from the text files.

## Run order

```bash
# Phase 2 — data and frozen splits
python scripts/01_prepare_data.py --config configs/default.yaml

# Phase 3 — EDA and the shortcut audit
python scripts/02_run_eda.py --config configs/default.yaml

# Phase 3 — lung masks for Shenzhen (needs the dl extra + a GPU for speed)
python scripts/03_train_unet_masks.py --config configs/default.yaml --epochs 60

# Phase 4 — classical lung-field baseline (E0), within each set
python scripts/04_build_features.py --config configs/default.yaml
python scripts/05_run_experiment.py --config configs/E0_classical_within.yaml
python scripts/05_run_experiment.py --config configs/E0_classical_within_shenzhen.yaml

# Phase 4 — CNN within each set (E1, E2)
python scripts/05_run_experiment.py --config configs/E1_montgomery_within.yaml
python scripts/05_run_experiment.py --config configs/E2_shenzhen_within.yaml
```

Phase 5 (weeks 7–8) — cross-dataset generalisation and the shortcut audit:

```bash
# CNN, both directions
python scripts/05_run_experiment.py --config configs/E3_mc_to_sz_cross.yaml
python scripts/05_run_experiment.py --config configs/E4_sz_to_mc_cross.yaml

# classical lung-field baseline across datasets (needs 03 + 04 for Shenzhen features)
python scripts/05_run_experiment.py --config configs/E3_classical_mc_to_sz_cross.yaml
python scripts/05_run_experiment.py --config configs/E4_classical_sz_to_mc_cross.yaml

# region controls: every E1–E4 has a *_lungcrop.yaml and *_lungmask.yaml twin
python scripts/05_run_experiment.py --config configs/E3_mc_to_sz_cross_lungmask.yaml

# where does the CNN look? (run after the experiment; uses its saved weights)
python scripts/07_gradcam.py --config configs/E3_mc_to_sz_cross.yaml

# paired AUC tests across every whole-vs-region and CNN-vs-classical pair with results
python scripts/08_compare_auc.py
```

`lung_masked` blacks out the lungs and keeps everything else: if a model still
scores well on it, it is using cues outside the lung field.

Phase 6 (weeks 8–9) — intensity corrections (`correction: histeq |
clahe | zscore`), applied per image to train and test alike, using only that
image's pixels. `histeq` and `zscore` take their statistics from the image
content and leave pure-black borders black (Montgomery films have large ones,
Shenzhen films do not). The CNN applies the correction at load time, so each
run is just a config:

```bash
python scripts/05_run_experiment.py --config configs/E3_mc_to_sz_cross_histeq.yaml
python scripts/05_run_experiment.py --config configs/E3_mc_to_sz_cross_clahe.yaml
python scripts/05_run_experiment.py --config configs/E3_mc_to_sz_cross_zscore.yaml
python scripts/06_make_results_table.py --config configs/default.yaml
```

Week 9 completes the family. The classical model reads a corrected feature
table (`features_<correction>.csv`), so build those first; the CNN needs nothing
extra:

```bash
# corrected lung-field features for the classical model
python scripts/04_build_features.py --config configs/default.yaml --corrections histeq clahe zscore

# classical model, both directions, one run per correction
python scripts/05_run_experiment.py --config configs/E3_classical_mc_to_sz_cross_histeq.yaml
python scripts/05_run_experiment.py --config configs/E3_classical_mc_to_sz_cross_clahe.yaml
python scripts/05_run_experiment.py --config configs/E3_classical_mc_to_sz_cross_zscore.yaml
python scripts/05_run_experiment.py --config configs/E4_classical_sz_to_mc_cross_histeq.yaml
python scripts/05_run_experiment.py --config configs/E4_classical_sz_to_mc_cross_clahe.yaml
python scripts/05_run_experiment.py --config configs/E4_classical_sz_to_mc_cross_zscore.yaml

# CNN Shenzhen -> Montgomery, one run per correction
python scripts/05_run_experiment.py --config configs/E4_sz_to_mc_cross_histeq.yaml
python scripts/05_run_experiment.py --config configs/E4_sz_to_mc_cross_clahe.yaml
python scripts/05_run_experiment.py --config configs/E4_sz_to_mc_cross_zscore.yaml

# did CLAHE move the CNN's attention onto the lungs?
python scripts/07_gradcam.py --config configs/E3_mc_to_sz_cross_clahe.yaml

# DeLong tests: each correction vs none, Holm-corrected within the phase6 family
python scripts/08_compare_auc.py
python scripts/06_make_results_table.py --config configs/default.yaml
```

## The experiment matrix

Each config file is one row of Appendix B. The fields that matter:

- `train` / `test` — `montgomery`, `shenzhen`, or `both`
- `region` — `whole`, `lung_cropped`, `lung_masked`
- `model` — `classical`, `resnet18`, `densenet121`
- `eval_mode` — `within` (frozen 5-fold) or `cross` (train one set, test the
  other once)
- `correction` — `none` (default), `histeq`, `clahe`, `zscore`

Changing an experiment means editing those fields, never the code. The region
switch (`lung_cropped`, `lung_masked`) needs lung masks, so run Phase 3 first;
`lung_masked` is the sharp control that answers "is the model even using the
lungs?".

## Evaluation

AUC is primary (threshold-free, unaffected by the prevalence gap between the
sets). Alongside it: sensitivity and specificity at a threshold **fixed on
training data**, and specificity at 90% sensitivity as the screening operating
point. Every metric carries a 95% CI from 2,000 stratified bootstrap resamples
(`tbx.evaluation.bootstrap`), and paired AUC comparisons on the same test set
use DeLong's test (`tbx.evaluation.delong`) with a Holm correction within each
family of comparisons.

## Reproducibility

Every script seeds Python, NumPy and torch from `data.seed` (default 42). The
five-fold splits are written to disk once and reused by every experiment, so the
classical model and the CNN are always compared on identical folds and no result
can be flattered by a lucky split. Each experiment's full config is saved next
to its results.

## A quick check without the real data

`tests/smoke_test.py` builds a tiny synthetic dataset in the NLM layout and runs
the whole non-deep-learning path (metadata → splits → EDA → features → classical
experiment), plus unit checks on the bootstrap and DeLong routines. Useful for
confirming the environment before the downloads finish:

```bash
python tests/smoke_test.py
```

## Layout

```
configs/      default.yaml + one file per experiment (E0–E4, region and correction twins)
src/tbx/
  config.py        experiment schema + YAML loader
  data/            parsing, metadata, preprocessing, masks, splits
  eda/             visual audit + distribution plots
  segmentation/    lung U-Net (train on Montgomery, infer Shenzhen)
  features/        lung-field texture / intensity features
  models/          classical (LR/SVM) and CNN (ResNet-18 / DenseNet-121)
  evaluation/      metrics, bootstrap CIs, DeLong's test
  runner.py        run any experiment from its config
scripts/      numbered CLI entry points, one per stage
tests/        synthetic end-to-end smoke test
```
