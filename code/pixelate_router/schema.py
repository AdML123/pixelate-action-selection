REQUIRED_ACCURACY_COLUMNS = {"corruption", "severity", "config", "accuracy", "transform"}


def validate_accuracy_table(df):
    missing = REQUIRED_ACCURACY_COLUMNS.difference(df.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"missing columns: {names}")
