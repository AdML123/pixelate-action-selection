# Data Layout

The public package contains compact derived artifacts only.

```text
data/
  action_eval/
    action_eval.csv.gz
    action_eval.json
  features/
    features_<corruption>_<severity>_{search,validation}.npy
    feature_summary.json
  features_test/
    features_<corruption>_<severity>_test.npy
    feature_summary.json
  router/
    router_all.pt
    router_no_commutator.pt
    router_confidence.pt
    router_spectral.pt
    validation_report.json
    ablation_report.json
    loco_report.json
  derived/imagenetc/
    oracle_ceiling_s3_2000.json
    action_eval_summary.json
    feature_summary.json
    validation_report.json
    ablation_report.json
    loco_report.json
    detector_baseline_report.json
    timing_report.json
  source/
    figure_pixelate_severity.csv
    table_pixelate_severity.csv
```

Raw ImageNet-C images and external model checkpoints are not redistributed.
The included per-image action outputs, feature matrices, and trained router
checkpoints are derived artifacts used to recompute the reported tables and
aggregate figures without rerunning image inference.
