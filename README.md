# Art Competition: Ideal Dorsal Butterfly Pattern

This repository contains a butterfly dorsal-pattern pipeline and generated outputs for artwork submission.

The core script is:

- `inat_dorsal_skeletons.py`

It supports:

- Fetching specimen images from GBIF-backed museum records
- Filtering to dorsal/in-view candidates
- Skeleton and mathematical reconstruction generation
- Building an idealized dorsal pattern and ideal mathematical reconstruction
- Pruning non-dorsal images from an existing local dataset

## Repository Layout

- `inat_dorsal_skeletons.py`: Main pipeline script
- `museum_butterfly_dorsal/`: Local dataset and generated intermediate/final artifacts
- `Output/`: Print-ready submission images and text description
- `artwork_description.txt`: Root-level artwork statement

## Setup

## 1. Python

Use Python 3.10+ (3.11 recommended).

## 2. Create and activate a virtual environment

```bash
cd /Users/khushalpandala/khush_fun/art_competition
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

Required:

```bash
pip install --upgrade pip
pip install numpy pillow
```

Optional (for Hugging Face embeddings instead of offline fallback):

```bash
pip install torch transformers
```

If `torch/transformers` are not installed, the script automatically falls back to offline image-stat embeddings.

## Run Instructions

## A) Use included dataset (recommended quick start)

Generate ideal math reconstruction from the existing dataset:

```bash
python3 inat_dorsal_skeletons.py \
  --build-ideal-math \
  --math-dataset-dir museum_butterfly_dorsal/mathequations_shape_fill \
  --ideal-math-output museum_butterfly_dorsal/ideal_math_reconstruction_shape_fill_rawref.png \
  --ideal-math-raw-ref-dir museum_butterfly_dorsal/raw_images \
  --ideal-top-k 80 \
  --ideal-consensus-threshold 0.20
```

## B) Prune non-dorsal images from local raw set

```bash
python3 inat_dorsal_skeletons.py \
  --prune-non-dorsal \
  --prune-input-dir museum_butterfly_dorsal/raw_images \
  --prune-threshold 0.66
```

Rejected images are moved to:

- `museum_butterfly_dorsal/raw_images_rejected_non_dorsal`

## C) Rebuild math dataset from raw images

```bash
python3 inat_dorsal_skeletons.py \
  --build-math-dataset \
  --math-dataset-input museum_butterfly_dorsal/raw_images \
  --math-dataset-dir museum_butterfly_dorsal/mathequations_shape_fill \
  --math-min-dorsal-score 0.60
```

## D) Build ideal dorsal pattern

```bash
python3 inat_dorsal_skeletons.py \
  --build-ideal \
  --ideal-input-dir museum_butterfly_dorsal/raw_images \
  --ideal-output museum_butterfly_dorsal/ideal_dorsal_pattern.png
```

## E) Build mathematical outline

```bash
python3 inat_dorsal_skeletons.py \
  --build-math-outline \
  --ideal-input-dir museum_butterfly_dorsal/raw_images \
  --math-output museum_butterfly_dorsal/mathematical_outline_shape_fill.png
```

## F) Full API fetch/filter/skeleton run

```bash
python3 inat_dorsal_skeletons.py \
  --outdir museum_butterfly_dorsal \
  --target-count 50 \
  --min-dim 1200 \
  --dorsal-threshold 0.62 \
  --workers 8 \
  --pages-per-query 2 \
  --page-limit 100 \
  --institutions nhmuk usnm am
```

## Useful Outputs

- `museum_butterfly_dorsal/ideal_math_reconstruction_shape_fill_rawref.png`
- `museum_butterfly_dorsal/ideal_math_reconstruction_shape_fill_rawref_consensus.png`
- `museum_butterfly_dorsal/ideal_math_reconstruction_shape_fill_rawref_selected.csv`
- `Output/artwork_submission_24x36_*.png`
- `Output/artwork_submission_24x36_*.jpg`

## Notes

- This repository includes dataset artifacts and generated files, so clone/push operations are larger than typical code-only repos.
- For repeatable results, keep the same thresholds and `--ideal-top-k` values used above.
