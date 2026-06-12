# Compression-Denoising Action Selection for Pixelated Images

This package reproduces the manuscript tables and figures for a
pixelate-focused ImageNet-C study. The main claim is that a lightweight
offline logistic router improves held-out pixelate classification over a
DnCNN-only default, while contrast, elastic transform, and JPEG compression
serve as controls.

The public repository is `https://github.com/AdML123/pixelate-action-selection`.
The camera-ready reproduction state is tagged as
`v2026-06-12-camera-ready`; after cloning, `git rev-parse HEAD` records the
exact commit for that public result snapshot.

The compact JSON and CSV files under `data/derived/imagenetc/` and
`data/source/` are sufficient to regenerate the numerical tables and the
aggregate figures. The case figure uses one ImageNet-C pixelate image when a
local authorized dataset copy is supplied.

## Reported Claims

- Pixelate test gain over DnCNN-only: +5.18 percentage points.
- Pixelate leave-one-corruption-out gain: +3.81 percentage points.
- Conservative four-corruption average gain: +1.39 percentage points.
- Default action: DnCNN-only.
- Training mode: full-information offline action-success labels.

## Data Boundary

ImageNet and ImageNet-C are third-party datasets and are not redistributed in
this package. The case figure generator accepts a local authorized ImageNet-C
digital root through `--case-digital-root`. The package stores code, derived
JSON/CSV artifacts, generated tables, generated figures, tests, and
`MANIFEST.sha256`.

## Quick Verification

From this package directory:

```powershell
python -m pytest tests
python code/make_tables.py --outdir tables/imagenetc
python code/make_figures.py --outdir figures/imagenetc
python code/verify_results.py --results data/derived/imagenetc --tables tables/imagenetc --figures figures/imagenetc
```

The quick figure command regenerates the aggregate figures from the compact
CSV/JSON artifacts and keeps the packaged case figure when raw ImageNet-C
images are unavailable. To redraw Fig. 1 from a local authorized ImageNet-C
copy, add `--case-digital-root /path/to/ImageNet-C/digital --case-severity 3
--case-index 7000`.

The generated files map directly to the manuscript:

| Manuscript item | Reproduced file |
|---|---|
| Table I | `tables/imagenetc/table_pixelate_primary.tex` |
| Table II | `tables/imagenetc/table_main.tex` |
| Table III | `tables/imagenetc/table_pixelate_severity.tex` |
| Table IV | `tables/imagenetc/table_ablation.tex` |
| Fig. 1 | `figures/imagenetc/figure_case_mechanism.pdf` |
| Fig. 2 | `figures/imagenetc/figure_residual_routing.pdf` |
| Fig. 3 | `figures/imagenetc/figure_pixelate_severity.pdf` |

Archive-only helpers may also emit feature, oracle, timing, or action-share
tables for inspection. Those files are not main-text manuscript tables.

## Full Rerun

Copy `configs/local.example.yaml` to `configs/local.yaml` and edit the paths to
your local resources:

- ImageNet-C digital corruption directory, extracted from `digital.tar`
- DnCNN color blind checkpoint
- ResNet-50 checkpoint
- KAIR source tree for the DnCNN model definition

Then run the pipeline:

```powershell
python code/oracle_ceiling.py --config configs/local.yaml --severity 3 --images 2000
python code/full_action_eval.py --config configs/local.yaml
python code/extract_imagenetc_features.py --config configs/local.yaml
python code/train_router.py --config configs/local.yaml
python code/evaluate_router.py --config configs/local.yaml
python code/evaluate_corruption_detector.py --features-root data/features --test-features-root data/features_test --action-csv data/action_eval/action_eval.csv --outdir data/derived/imagenetc
python code/make_pixelate_severity_table.py --action-csv data/action_eval/action_eval.csv --features-root-test data/features_test --router-checkpoint data/router/router_all.pt --outdir data/source
python code/run_loco.py --config configs/local.yaml
python code/measure_latency.py --config configs/local.yaml
python code/make_tables.py --outdir tables/imagenetc
python code/make_figures.py --outdir figures/imagenetc --case-digital-root /path/to/ImageNet-C/digital --case-severity 3 --case-index 7000
python code/verify_results.py --results data/derived/imagenetc --tables tables/imagenetc --figures figures/imagenetc
```

The public package keeps compact derived summaries and figure source data.
Large per-image CSV files, feature matrices, checkpoints, raw images, logs,
and LaTeX build caches are intentionally excluded.

## Result Summary

The held-out test split uses DnCNN-only as the default preprocessing action.
The logistic action router improves pixelate top-1 accuracy from 43.58% to
48.76%, with a paired lower confidence bound of 4.92 percentage points. When
pixelate is held out during training, the gain remains 3.81 points. Across
pixelate plus three digital controls, the conservative average gain is 1.39
points over the DnCNN-only default.

## Cleanliness

Before upload, run:

```powershell
python code/secret_scan.py .
python code/write_manifest.py --root . --output MANIFEST.sha256
```

The package should not contain local absolute paths, credentials, temporary
logs, LaTeX auxiliary files, or files larger than the normal GitHub warning
threshold.
