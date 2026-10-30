"""Data auditing, physical domain checks, and leak-free splitting."""
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

PHYSICAL_BOUNDS = {
    "carbon_dioxide": (350.0, 5000.0),
    "temperature": (0.0, 50.0),
    "humidity": (0.0, 100.0),
    "pressure": (950.0, 1060.0),
    "particulate_matter_1": (0.0, 500.0),
    "particulate_matter_2.5": (0.0, 500.0),
    "particulate_matter_4": (0.0, 500.0),
    "particulate_matter_10": (0.0, 500.0),
}

CORE_FEATURES = [
    "temperature", "humidity", "pressure",
    "particulate_matter_1", "particulate_matter_2.5",
    "particulate_matter_4", "particulate_matter_10",
    "hour_sin", "hour_cos", "dT_dt", "dRH_dt"
]
TARGET_COL = "carbon_dioxide"

def audit_and_clean_sensor_data(df: pd.DataFrame, start_date="2024-05-01", end_date="2024-12-31"):
    data = df.copy()
    if "datetime" not in data.columns:
        data["datetime"] = pd.to_datetime(data["timestamp"], unit="s", utc=True)
    ts_utc = pd.to_datetime(data["timestamp"], unit="s", utc=True)
    valid_window = (ts_utc >= pd.Timestamp(start_date, tz="UTC")) & (ts_utc <= pd.Timestamp(end_date + " 23:59:59", tz="UTC"))
    data = data.loc[valid_window].sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)
    data = data.drop_duplicates(subset=["sensor_id", "timestamp"], keep="first").reset_index(drop=True)
    for col, (low, high) in PHYSICAL_BOUNDS.items():
        if col in data.columns:
            data.loc[~data[col].between(low, high) & data[col].notna(), col] = np.nan
    data = data.dropna(subset=[TARGET_COL, "temperature", "humidity", "pressure"]).reset_index(drop=True)
    summary = pd.DataFrame({
        "metric": ["clean_observations", "unique_sensors", "min_timestamp", "max_timestamp"],
        "value": [len(data), data["sensor_id"].nunique(), str(data["datetime"].min()), str(data["datetime"].max())]
    })
    return data, summary

def engineer_physical_features(df: pd.DataFrame):
    data = df.copy()
    dt = pd.to_datetime(data["timestamp"], unit="s", utc=True).dt.tz_convert("Australia/Melbourne")
    f_hour = dt.dt.hour + dt.dt.minute / 60.0
    data["hour_sin"] = np.sin(2.0 * np.pi * f_hour / 24.0)
    data["hour_cos"] = np.cos(2.0 * np.pi * f_hour / 24.0)
    data["dT_dt"] = 0.0
    data["dRH_dt"] = 0.0
    for _, grp in data.groupby("sensor_id"):
        idx = grp.index
        dt_h = grp["timestamp"].diff() / 3600.0
        val = (dt_h > 1e-4) & (dt_h <= 2.0)
        data.loc[idx, "dT_dt"] = np.nan_to_num(np.where(val, grp["temperature"].diff() / dt_h, 0.0))
        data.loc[idx, "dRH_dt"] = np.nan_to_num(np.where(val, grp["humidity"].diff() / dt_h, 0.0))
    return data

def split_sensor_data(df: pd.DataFrame, split_kind="time", test_size=0.25, random_state=42):
    n = len(df)
    idx = np.arange(n)
    if split_kind == "random":
        rng = np.random.default_rng(random_state)
        shf = rng.permutation(idx)
        nv = int(n * test_size)
        return shf[nv:], shf[:nv]
    elif split_kind == "grouped":
        val_sensor = df["sensor_id"].unique()[-1]
        mask = (df["sensor_id"] == val_sensor).to_numpy()
        return idx[~mask], idx[mask]
    elif split_kind == "time":
        tr_l, val_l = [], []
        for _, grp in df.groupby("sensor_id"):
            ord_i = grp.sort_values("timestamp").index.to_numpy()
            c = int(len(ord_i) * (1.0 - test_size))
            tr_l.append(ord_i[:c])
            val_l.append(ord_i[c:])
        return np.concatenate(tr_l), np.concatenate(val_l)
