from collections import defaultdict


def choose_adaptive_config(hfer: float, threshold: float) -> str:
    return "config_a" if hfer > threshold else "config_b"


def summarize_accuracy(rows: list[dict]) -> list[dict]:
    buckets = defaultdict(lambda: [0, 0])
    for row in rows:
        key = (row["corruption"], row["severity"], row["config"])
        buckets[key][1] += 1
        buckets[key][0] += int(bool(row["correct"]))
    return [
        {
            "corruption": corruption,
            "severity": severity,
            "config": config,
            "accuracy": 100.0 * correct / total,
        }
        for (corruption, severity, config), (correct, total) in sorted(buckets.items())
    ]
