"""Evaluation metrics and profiler."""
import time, tracemalloc
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

def compute_engineering_metrics(y_true, y_pred):
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    res = yt - yp
    return {
        "MAE_ppm": float(mean_absolute_error(yt, yp)),
        "RMSE_ppm": float(root_mean_squared_error(yt, yp)),
        "R2": float(r2_score(yt, yp)),
        "Max_Error_ppm": float(np.max(np.abs(res)))
    }

def profile_callable(func, *args, **kwargs):
    tracemalloc.start()
    t0 = time.perf_counter()
    res = func(*args, **kwargs)
    elapsed = (time.perf_counter() - t0) * 1000.0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return res, float(elapsed), float(peak / 1024.0)
