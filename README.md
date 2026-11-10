# MMA3001: Sensor Dropout Reconstruction in Smart Infrastructure

An investigation evaluating 1D numerical interpolation versus multivariate machine-learning regression for contiguous sensor dropout reconstruction in smart building environmental monitoring[cite: 21].

---

## 1. Engineering Problem & Operational Context
Modern Building Management Systems (BMS) use indoor carbon dioxide ($CO_2$) telemetry to modulate fresh-air ventilation dampers[cite: 26]. Wireless packet loss, power interruptions, and sensor hardware faults cause contiguous multi-hour signal dropouts that blind closed-loop ventilation controllers[cite: 19].

* **Intended Use:** Automated reconstruction of missing $CO_2$ time-series blocks during telemetry blackouts to maintain closed-loop air handling without saturating actuators[cite: 19].
* **Scope Limits:** 
  * Validated for contiguous outage durations spanning 10 minutes to 8 hours ($\Delta t_{\text{gap}} \in [0.17, 8.0]\text{ hr}$)[cite: 19].
  * Calibrated specifically for indoor lecture and laboratory microclimates (Monash Living Lab Room 69_G.57); models require local recalibration before spatial transfer to different ventilation zones[cite: 19, 26].
  * Inferences are bounded to the physical operating domain $[350, 5000]\text{ ppm}$ to prevent unphysical BMS control signals[cite: 15, 26].

---

## 2. Input and Output Engineering Specification

### Primary Output (Target)
| Quantity | Symbol | Format / Type | Units | Physical Domain | Engineering Meaning | Handling of Invalid / Missing |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Carbon Dioxide | $CO_2$ | `float64` | $\text{ppm}$ | $[350.0, 5000.0]$ | Indoor air quality indicator & occupancy proxy | Out-of-bounds clipped/filtered; target reconstructed during dropout[cite: 15, 26] |

### Input Features
| Feature Name | Symbol | Format / Type | Units | Physical Domain | Engineering Meaning | Handling of Invalid / Missing |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Ambient Temperature | $T$ | `float64` | $^\circ\text{C}$ | $[0.0, 50.0]$ | Indoor thermal state | Filtered if outside bounds; median imputed[cite: 15, 26] |
| Relative Humidity | $RH$ | `float64` | $\%RH$ | $[0.0, 100.0]$ | Room moisture fraction | Filtered if outside bounds; median imputed[cite: 15, 26] |
| Barometric Pressure | $P$ | `float64` | $\text{hPa}$ | $[950.0, 1060.0]$ | Atmospheric ambient pressure | Filtered if outside bounds; median imputed[cite: 15, 26] |
| Particulate Matter 1.0 | $\text{PM}_1$ | `float64` | $\mu\text{g/m}^3$ | $[0.0, 500.0]$ | Ultra-fine optical dust fraction | Filtered if outside bounds; median imputed[cite: 15, 26] |
| Particulate Matter 2.5 | $\text{PM}_{2.5}$ | `float64` | $\mu\text{g/m}^3$ | $[0.0, 500.0]$ | Fine respirable dust fraction | Filtered if outside bounds; median imputed[cite: 15, 26] |
| Particulate Matter 4.0 | $\text{PM}_4$ | `float64` | $\mu\text{g/m}^3$ | $[0.0, 500.0]$ | Intermediate aerosol fraction | Filtered if outside bounds; median imputed[cite: 15, 26] |
| Particulate Matter 10 | $\text{PM}_{10}$ | `float64` | $\mu\text{g/m}^3$ | $[0.0, 500.0]$ | Coarse resuspended dust (occupancy proxy) | Filtered if outside bounds; median imputed[cite: 15, 26] |
| Diurnal Sine Component | $\text{hour\_sin}$ | `float64` | Dimensionless | $[-1.0, 1.0]$ | $\sin(2\pi t / 24)$ 24-hr cyclic schedule | Deterministic transform from local UTC+10[cite: 15] |
| Diurnal Cosine Component | $\text{hour\_cos}$ | `float64` | Dimensionless | $[-1.0, 1.0]$ | $\cos(2\pi t / 24)$ 24-hr cyclic schedule | Deterministic transform from local UTC+10[cite: 15] |
| Temperature Rate of Change | $dT/dt$ | `float64` | $^\circ\text{C/hr}$ | Real ($\mathbb{R}$) | Backward finite-difference thermal drift | 0.0 on sequence boundaries; causal stencil[cite: 15, 19] |
| Humidity Rate of Change | $dRH/dt$ | `float64` | $\%RH\text{/hr}$ | Real ($\mathbb{R}$) | Backward finite-difference moisture drift | 0.0 on sequence boundaries; causal stencil[cite: 15, 19] |

---

## 3. Repository Structure

```text
├── src/                      # Modular Python engineering package
│   ├── preprocessing.py      # Physical bounds checking & chronological splitting
│   ├── models.py             # Numerical solvers and ML estimators
│   ├── evaluation.py         # Metrics (MAE, RMSE, R², E_max) & latency profiling
│   └── experiments.py        # Dropout duration sensitivity sweeps (10m - 8h)
├── tests/                    # Automated pytest suite (exactness & leakage verification)
├── results/                  # Benchmark tables (.csv) and publication plots (.png)
├── docs/                     # API documentation generated via pdoc
├── 3001project.ipynb         # End-to-end reproducible analysis notebook
├── LICENSE                   # MIT License
└── README.md                 # Project reproduction and technical summary
```[cite: 21]

---

## 4. Setup, Environment & Reproducibility

### Environment & Dependencies
The codebase requires Python 3.9+ and standard scientific packages[cite: 15]. Install all dependencies in one command:
```bash
git clone [https://github.com/adut0011/MMA3001-PROJECT.git](https://github.com/adut0011/MMA3001-PROJECT.git)
cd MMA3001-PROJECT
pip install pytest pdoc scikit-learn scipy pandas numpy matplotlib
```[cite: 15]

### Running Automated Verification Tests
Execute the verification test suite to mathematically confirm interpolation exactness, boundary handling, and leak-free splitting:
```bash
pytest tests/ -v
```[cite: 15]

### Running the End-to-End Pipeline
To reproduce all model benchmarks, sensitivity sweeps, and result figures:
* Open and execute the primary notebook: `3001project.ipynb`
* Or run the modular experiment suite via the CLI:
```bash
python -m src.experiments
```[cite: 15]

### Generating API Documentation
API documentation is compiled directly from NumPy-format docstrings using `pdoc`:
```bash
pdoc --math src -o docs
