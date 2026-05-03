import pandas as pd


def load_and_clean(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    # Normalize all column names: lowercase + strip + replace dashes
    df.columns = [c.strip().lower().replace("-", "_") for c in df.columns]

    # CRITICAL: dynamically find and rename the no-show column
    # Dataset has "No-show" which becomes "no_show" after normalization
    rename_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if "show" in col_lower:
            rename_map[col] = "no_show"
        elif col_lower in ("hipertension", "hypertension"):
            rename_map[col] = "hypertension"
        elif col_lower in ("handcap", "handicap"):
            rename_map[col] = "handicap"
        elif col_lower == "scheduledday":
            rename_map[col] = "scheduled_day"
        elif col_lower == "appointmentday":
            rename_map[col] = "appointment_day"
        elif col_lower == "patientid":
            rename_map[col] = "patient_id"
        elif col_lower == "appointmentid":
            rename_map[col] = "appointment_id"
    df = df.rename(columns=rename_map)

    # Fix target column: "Yes" = no-show (1), "No" = showed up (0)
    df["no_show"] = df["no_show"].map({"Yes": 1, "No": 0})

    # Remove invalid ages
    df = df[(df["age"] >= 0) & (df["age"] <= 115)]

    # Parse dates (UTC-aware)
    df["scheduled_day"] = pd.to_datetime(df["scheduled_day"], utc=True)
    df["appointment_day"] = pd.to_datetime(df["appointment_day"], utc=True)

    # Feature 1: days_in_advance (required by exam spec)
    df["days_in_advance"] = (df["appointment_day"] - df["scheduled_day"]).dt.days
    df = df[df["days_in_advance"] >= 0]

    # Feature 2: day of week (0=Mon...6=Sun)
    df["appt_day_of_week"] = df["appointment_day"].dt.dayofweek

    return df


def get_features_and_target(df: pd.DataFrame):
    feature_cols = [
        "age",
        "scholarship",
        "hypertension",
        "diabetes",
        "alcoholism",
        "handicap",
        "sms_received",
        "days_in_advance",
        "appt_day_of_week",
    ]
    X = df[feature_cols]
    y = df["no_show"]
    return X, y
