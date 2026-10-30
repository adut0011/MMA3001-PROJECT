"""Numerical interpolation and ML regression models."""
import numpy as np
from scipy.interpolate import interp1d, CubicSpline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

def interpolate_sensor_gaps(t_k, y_k, t_q, method="linear", clip_bounds=(350.0, 5000.0)):
    tk, yk, tq = np.asarray(t_k, float), np.asarray(y_k, float), np.asarray(t_q, float)
    if np.any(np.diff(tk) <= 0):
        raise ValueError("time_known must be strictly increasing.")
    if method == "linear":
        pred = interp1d(tk, yk, kind="linear", bounds_error=False, fill_value=(yk[0], yk[-1]))(tq)
    elif method == "cubic":
        pred = CubicSpline(tk, yk, bc_type="not-a-knot", extrapolate=True)(tq)
    else:
        raise ValueError("Unknown method")
    if clip_bounds is not None:
        pred = np.clip(pred, clip_bounds[0], clip_bounds[1])
    return np.asarray(pred, float)

def build_ml_regressors(random_state=42):
    return {
        "Mean Baseline": Pipeline([("imp", SimpleImputer(strategy="median")), ("m", DummyRegressor(strategy="mean"))]),
        "Ridge Regression": Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("m", Ridge(alpha=10.0, random_state=random_state))]),
        "Random Forest": Pipeline([("imp", SimpleImputer(strategy="median")), ("m", RandomForestRegressor(n_estimators=60, max_depth=12, min_samples_leaf=4, n_jobs=-1, random_state=random_state))])
    }

def estimate_inference_flops(model_name, n_samples, n_features):
    if model_name == "Mean Baseline": return int(n_samples)
    if model_name == "Linear Interpolation": return int(n_samples * 4)
    if model_name == "Cubic Spline": return int(n_samples * 7)
    if model_name == "Ridge Regression": return int(n_samples * (4 * n_features + 1))
    if model_name == "Random Forest": return int(n_samples * (60 * 13 + 1))
    return int(n_samples)
