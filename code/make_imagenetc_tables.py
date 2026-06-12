from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


CORRUPTION_LABELS = {
    "contrast": "Contrast",
    "elastic_transform": "Elastic",
    "pixelate": "Pixelate",
    "jpeg_compression": "JPEG",
}


def format_accuracy(value: float) -> str:
    return f"{float(value):.1f}"


def format_gain(value: float) -> str:
    rounded = round(float(value), 1)
    if rounded == 0.0:
        return "0.0"
    return f"{rounded:+.1f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate ImageNet-C manuscript table snippets.")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--oracle-json", default="data/derived/imagenetc/oracle_ceiling_s3_2000.json")
    parser.add_argument("--action-json", default="data/derived/imagenetc/action_eval_summary.json")
    parser.add_argument("--feature-json", default="data/derived/imagenetc/feature_summary.json")
    parser.add_argument("--validation-json", default="data/derived/imagenetc/validation_report.json")
    parser.add_argument("--ablation-json", default="data/derived/imagenetc/ablation_report.json")
    parser.add_argument("--detector-json", default="data/derived/imagenetc/detector_baseline_report.json")
    parser.add_argument("--loco-json", default="data/derived/imagenetc/loco_report.json")
    parser.add_argument("--severity-csv", default="data/source/table_pixelate_severity.csv")
    parser.add_argument("--timing-json", default="data/derived/imagenetc/timing_report.json")
    parser.add_argument(
        "--include-archive",
        action="store_true",
        help="Also write diagnostic tables that are not used in the manuscript main text.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    action = _load(args.action_json)
    ablation = _load(args.ablation_json)
    detector = _load(args.detector_json)
    ablation = with_detector_policy(ablation, detector)
    severity_rows = _read_csv_rows(Path(args.severity_csv))

    _write(outdir / "table_pixelate_primary.tex", table_pixelate_primary(action, ablation))
    _write(outdir / "table_main.tex", table_main(action, ablation))
    _write(outdir / "table_ablation.tex", table_ablation(ablation))
    _write(outdir / "table_pixelate_severity.tex", table_pixelate_severity_from_rows(severity_rows))
    if args.include_archive:
        oracle = _load(args.oracle_json)
        feature = _load(args.feature_json)
        timing = _load(args.timing_json)
        _write(outdir / "table_oracle.tex", table_oracle(oracle, action))
        _write(outdir / "table_features.tex", table_features(feature))
        _write(outdir / "table_action_distribution.tex", table_action_distribution(ablation))
        _write(outdir / "table_timing.tex", table_timing(timing))
    print(f"wrote tables to {outdir}")
    return 0


def table_oracle(oracle: dict, action: dict) -> str:
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Corruption & D & Best & Oracle & Headroom & Ent. \\",
        r"\midrule",
    ]
    for corr in CORRUPTION_LABELS:
        s3 = oracle["summaries"][corr]
        counts = {
            action_name: int(count)
            for action_name, count in s3["oracle_winner_counts"].items()
            if action_name != "none"
        }
        lines.append(
            " & ".join(
                [
                    CORRUPTION_LABELS[corr],
                    format_accuracy(s3["dncnn_accuracy"]),
                    format_accuracy(s3["best_fixed_accuracy"]),
                    format_accuracy(s3["per_image_oracle_accuracy"]),
                    format_gain(s3["per_image_oracle_accuracy"] - s3["best_fixed_accuracy"]),
                    f"{_entropy_bits(counts):.2f}",
                ]
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def table_features(feature: dict) -> str:
    by_split = feature["aggregate"]["by_split"]["validation"]
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Corruption & HFER & $m_{\mathrm{comm}}$ & J-res. & D-res. \\",
        r"\midrule",
    ]
    for corr in CORRUPTION_LABELS:
        values = list(by_split[corr].values())
        hfer = _mean(values, "hfer_input_mean")
        comm = _mean(values, "m_comm_mean")
        jpeg = _mean(values, "hfer_jpeg20_residual_mean")
        dncnn = _mean(values, "hfer_dncnn_residual_mean")
        lines.append(
            f"{CORRUPTION_LABELS[corr]} & {hfer:.3f} & {comm:.4f} & {jpeg:.3f} & {dncnn:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def table_pixelate_primary(action: dict, ablation: dict) -> str:
    summaries = action["summaries"]["test"]["pixelate"]
    router = ablation["policies"]["logistic_all"]["test"]["per_corruption"]["pixelate"]
    dncnn = _weighted(summaries, "dncnn_accuracy")
    best = _weighted(summaries, "best_fixed_accuracy")
    oracle = _weighted(summaries, "per_image_oracle_accuracy")
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Condition & DnCNN & Best & Router & Oracle & Gain & LCB \\",
        r"\midrule",
        " & ".join(
            [
                "Pixelate test",
                format_accuracy(dncnn),
                format_accuracy(best),
                format_accuracy(router["accuracy"]),
                format_accuracy(oracle),
                format_gain(router["gain"]),
                format_gain(router["paired_lcb"]),
            ]
        )
        + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines) + "\n"


def table_main(action: dict, ablation: dict) -> str:
    router = ablation["policies"]["logistic_all"]["test"]
    lines = [
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        r"Corruption & Def. & Best & Router & Gain & Act. \\",
        r"\midrule",
    ]
    for corr in CORRUPTION_LABELS:
        summaries = action["summaries"]["test"][corr]
        default = _weighted(summaries, "dncnn_accuracy")
        best = _weighted(summaries, "best_fixed_accuracy")
        per_corr = router["per_corruption"][corr]
        act = _top_action(per_corr["action_distribution"])
        lines.append(
            " & ".join(
                [
                    CORRUPTION_LABELS[corr],
                    format_accuracy(default),
                    format_accuracy(best),
                    format_accuracy(per_corr["accuracy"]),
                    format_gain(per_corr["gain"]),
                    _action_short(act),
                ]
            )
            + r" \\"
        )
    lines.append(r"\midrule")
    all_test = action["summaries"]["test"]
    default_avg = _weighted_all(all_test, "dncnn_accuracy")
    best_avg = _weighted_all(all_test, "best_fixed_accuracy")
    lines.append(
        " & ".join(
            [
                "Average",
                format_accuracy(default_avg),
                format_accuracy(best_avg),
                format_accuracy(router["accuracy"]),
                format_gain(router["gain"]),
                _action_short(_top_action(router["action_distribution"])),
            ]
        )
        + r" \\"
    )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# Archive-only helper; the main manuscript shows these shares in Figure 2.
def table_action_distribution(ablation: dict) -> str:
    router = ablation["policies"]["logistic_all"]["test"]["per_corruption"]
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Corruption & D & J20 & A & B & Other & Gain \\",
        r"\midrule",
    ]
    for corr in CORRUPTION_LABELS:
        item = router[corr]
        dist = item["action_distribution"]
        d_share = 100.0 * float(dist.get("dncnn", {}).get("share", 0.0))
        j20_share = 100.0 * float(dist.get("jpeg20", {}).get("share", 0.0))
        a_share = 100.0 * float(dist.get("config_a", {}).get("share", 0.0))
        b_share = 100.0 * float(dist.get("config_b", {}).get("share", 0.0))
        other_share = max(0.0, 100.0 - d_share - j20_share - a_share - b_share)
        lines.append(
            f"{CORRUPTION_LABELS[corr]} & {d_share:.1f} & {j20_share:.1f} & {a_share:.1f} & {b_share:.1f} & {other_share:.1f} & {format_gain(item['gain'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def with_detector_policy(ablation: dict, detector: dict) -> dict:
    merged = json.loads(json.dumps(ablation))
    merged["policies"]["detector_per_corruption"] = {
        "policy": "detector_per_corruption",
        "feature_set": detector["feature_set"],
        "selected": detector["selected"],
        "validation": detector["validation"],
        "test": detector["test"],
    }
    merged["policies"]["true_label_per_corruption_oracle"] = {
        "policy": "true_label_per_corruption_oracle",
        "feature_set": "diagnostic",
        "selected": detector["selected"],
        "validation": detector["validation"],
        "test": detector["diagnostic_true_label_oracle"],
    }
    return merged


def table_ablation(ablation: dict) -> str:
    row_specs = [
        ("DnCNN only", "fixed", "default_dncnn"),
        ("Best fixed", "val. single", "best_fixed"),
        ("HFER rule", "single thr.", "hfer_rule"),
        ("Two-thresh.", "HFER", "two_threshold"),
        ("Router", "confidence", "logistic_confidence"),
        ("Router", "spectral", "logistic_spectral"),
        ("Router", "no comm.", "logistic_no_commutator"),
        ("Detector", "per-corr.", "detector_per_corruption"),
        ("Router", "all", "logistic_all"),
    ]
    if "true_label_per_corruption_oracle" in ablation["policies"]:
        row_specs.append(("Oracle", "true corr.", "true_label_per_corruption_oracle"))
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Policy & Features & Val. & Test & Pixelate & Contrast \\",
        r"\midrule",
    ]
    for policy, feat, key in row_specs:
        item = ablation["policies"][key]
        test = item["test"]
        per_corr = test["per_corruption"]
        lines.append(
            " & ".join(
                [
                    policy,
                    feat,
                    format_gain(item["validation"]["gain"]),
                    format_gain(test["gain"]),
                    format_gain(per_corr["pixelate"]["gain"]),
                    format_gain(per_corr["contrast"]["gain"]),
                ]
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def table_pixelate_severity_from_rows(rows: list[dict]) -> str:
    values: dict[int, dict[str, float]] = {}
    for row in rows:
        severity = int(row["severity"])
        values[severity] = {
            "dncnn": float(row["dncnn"]),
            "jpeg20": float(row["jpeg20"]),
            "config_a": float(row["config_a"]),
            "router": float(row["router"]),
            "gain": float(row["gain"]),
            "paired_lcb": float(row["paired_lcb"]),
        }

    lines = [
        r"\begin{tabular}{rrrrrrr}",
        r"\toprule",
        r"Sev. & DnCNN & J20 & Config-A & Router & Gain & LCB \\",
        r"\midrule",
    ]
    for severity in [1, 2, 3, 4, 5]:
        item = values[severity]
        lines.append(
            " & ".join(
                [
                    str(severity),
                    format_accuracy(item["dncnn"]),
                    format_accuracy(item["jpeg20"]),
                    format_accuracy(item["config_a"]),
                    format_accuracy(item["router"]),
                    format_gain(item["gain"]),
                    format_gain(item["paired_lcb"]),
                ]
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def table_timing(timing: dict) -> str:
    order = [
        ("Default D", "default_dncnn"),
        ("Config A", "config_a"),
        ("Config B", "config_b"),
        ("Oracle eval", "oracle_eval"),
        ("Router", "router"),
    ]
    lines = [
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Mode & Total (ms) \\",
        r"\midrule",
    ]
    for label, key in order:
        lines.append(f"{label} & {format_accuracy(timing['results'][key]['median_ms'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _mean(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _weighted(summaries: dict, key: str) -> float:
    total = sum(int(row["n"]) for row in summaries.values())
    return sum(float(row[key]) * int(row["n"]) for row in summaries.values()) / total


def _weighted_all(corruption_summaries: dict, key: str) -> float:
    total = 0
    value = 0.0
    for summaries in corruption_summaries.values():
        for row in summaries.values():
            n = int(row["n"])
            total += n
            value += float(row[key]) * n
    return value / total


def _entropy_bits(counts: dict[str, int]) -> float:
    values = [float(count) for count in counts.values() if count > 0]
    total = sum(values)
    if total <= 0.0:
        return 0.0
    return -sum((value / total) * math.log2(value / total) for value in values)


def _top_action(distribution: dict) -> str:
    return max(distribution.items(), key=lambda item: item[1]["share"])[0]


def _action_short(action: str) -> str:
    return {
        "identity": "I",
        "dncnn": "D",
        "jpeg20": "J20",
        "jpeg10": "J10",
        "config_a": "A",
        "config_b": "B",
        "config_a10": "A10",
    }[action]


if __name__ == "__main__":
    raise SystemExit(main())
