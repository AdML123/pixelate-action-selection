def compute_config_delta(df):
    deltas = {}
    for corruption, group in df.groupby("corruption"):
        values = dict(zip(group["config"], group["accuracy"]))
        deltas[corruption] = float(values["config_a"] - values["config_b"])
    return deltas


def make_value_map(raw_values):
    return {f"[{key}]": f"{float(value):.2f}" for key, value in raw_values.items()}
