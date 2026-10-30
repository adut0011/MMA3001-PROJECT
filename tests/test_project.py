"""Unit tests for MMA3001 Project."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, pandas as pd, pytest
from src.preprocessing import split_sensor_data
from src.models import interpolate_sensor_gaps, estimate_inference_flops
from src.evaluation import compute_engineering_metrics

def test_linear_interpolation_exact_on_line():
    tk = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    yk = 3.0 * tk + 400.0
    tq = np.array([5.0, 15.0, 25.0])
    ypred = interpolate_sensor_gaps(tk, yk, tq, "linear", None)
    assert np.allclose(ypred, 3.0 * tq + 400.0)

def test_cubic_spline_exact_on_cubic_polynomial():
    tk = np.linspace(0.0, 10.0, 11)
    yk = 420.0 + 5.0 * tk - 0.8 * (tk ** 2) + 0.05 * (tk ** 3)
    tq = np.array([2.5, 5.5, 7.5])
    ypred = interpolate_sensor_gaps(tk, yk, tq, "cubic", None)
    assert np.allclose(ypred, 420.0 + 5.0 * tq - 0.8 * (tq ** 2) + 0.05 * (tq ** 3))

def test_interpolation_rejects_duplicate_timestamps():
    with pytest.raises(ValueError, match="strictly increasing"):
        interpolate_sensor_gaps(np.array([1.0, 2.0, 2.0]), np.array([1, 2, 2]), np.array([1.5]))

def test_time_split_zero_leakage():
    df = pd.DataFrame({"sensor_id": ["S1"]*10, "timestamp": np.arange(10)})
    tr, val = split_sensor_data(df, "time", 0.3)
    assert df.loc[tr, "timestamp"].max() < df.loc[val, "timestamp"].min()

def test_metrics_accuracy():
    yt = np.array([400.0, 500.0, 600.0])
    yp = np.array([410.0, 490.0, 620.0])
    m = compute_engineering_metrics(yt, yp)
    assert np.isclose(m["MAE_ppm"], 40.0 / 3.0)
