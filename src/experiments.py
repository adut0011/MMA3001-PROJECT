"""Contiguous sensor dropout experiment."""
import numpy as np
import pandas as pd
from src.preprocessing import CORE_FEATURES, TARGET_COL
from src.models import interpolate_sensor_gaps, build_ml_regressors, estimate_inference_flops
from src.evaluation import compute_engineering_metrics, profile_callable

def run_gap_duration_experiment(train_df, test_df, gap_sizes_samples=[1, 3, 6, 12, 24, 48], n_gaps=15, random_state=42):
    X_tr = train_df[CORE_FEATURES].to_numpy()
    y_tr = train_df[TARGET_COL].to_numpy()
    pipes = build_ml_regressors(random_state)
    for p in pipes.values(): p.fit(X_tr, y_tr)
    
    ps = test_df["sensor_id"].value_counts().index[0]
    sub = test_df[test_df["sensor_id"] == ps].sort_values("timestamp").reset_index(drop=True)
    ts = sub["timestamp"].to_numpy(float)
    y_f = sub[TARGET_COL].to_numpy(float)
    X_f = sub[CORE_FEATURES].to_numpy(float)
    n_tot = len(sub)
    dt_m = float(np.median(np.diff(ts)) / 60.0)
    rng = np.random.default_rng(random_state)
    records = []
    
    for g_len in gap_sizes_samples:
        margin = max(10, g_len + 2)
        v_starts = np.arange(margin, n_tot - g_len - margin, g_len + 6)
        c_starts = rng.choice(v_starts, size=min(n_gaps, len(v_starts)), replace=False)
        mask = np.zeros(n_tot, bool)
        for s in c_starts: mask[s : s + g_len] = True
        
        t_k, y_k = ts[~mask], y_f[~mask]
        t_q, y_q = ts[mask], y_f[mask]
        X_q = X_f[mask]
        g_hours = round((g_len * dt_m) / 60.0, 2)
        
        # Linear
        pred_l, t_l, _ = profile_callable(interpolate_sensor_gaps, t_k, y_k, t_q, "linear", None)
        records.append({"gap_samples": g_len, "gap_duration_hours": g_hours, "method": "Linear Interpolation", **compute_engineering_metrics(y_q, pred_l), "inference_time_ms": t_l, "estimated_flops": estimate_inference_flops("Linear Interpolation", len(y_q), len(CORE_FEATURES))})
        
        # Cubic
        pred_c, t_c, _ = profile_callable(interpolate_sensor_gaps, t_k, y_k, t_q, "cubic", None)
        records.append({"gap_samples": g_len, "gap_duration_hours": g_hours, "method": "Cubic Spline", **compute_engineering_metrics(y_q, pred_c), "inference_time_ms": t_c, "estimated_flops": estimate_inference_flops("Cubic Spline", len(y_q), len(CORE_FEATURES))})
        
        # ML models
        for mname in ["Ridge Regression", "Random Forest"]:
            pred_m, t_m, _ = profile_callable(pipes[mname].predict, X_q)
            records.append({"gap_samples": g_len, "gap_duration_hours": g_hours, "method": mname, **compute_engineering_metrics(y_q, pred_m), "inference_time_ms": t_m, "estimated_flops": estimate_inference_flops(mname, len(y_q), len(CORE_FEATURES))})
            
    return pd.DataFrame(records)
