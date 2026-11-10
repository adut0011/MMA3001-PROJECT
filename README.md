# MMA3001: Sensor Dropout Reconstruction in Smart Infrastructure

An investigation evaluating 1D numerical interpolation versus multivariate machine-learning regression for contiguous sensor dropout reconstruction in smart building environmental monitoring.

---

## Engineering Problem & Contract
Building Management Systems (BMS) rely on continuous indoor telemetry to modulate ventilation dampers and maintain air quality. Contiguous telemetry dropouts (from wireless packet loss or device outages) blind closed-loop control systems.

* **Target Output:** Carbon Dioxide ($CO_2$) concentration in parts-per-million ($\text{ppm}$), bounded within physical domain $[350, 5000]\text{ ppm}$.
* **Input Features:** Ambient temperature ($^\circ\text{C}$), relative humidity ($\%RH$), barometric pressure ($\text{hPa}$), particulate matter ($\text{PM}_1, \text{PM}_{2.5}, \text{PM}_4, \text{PM}_{10}$ in $\mu\text{g/m}^3$), diurnal temporal harmonics ($\sin/\cos$ hour encodings), and causal backward finite-difference rates ($dT/dt$, $dRH/dt$).
* **Methods Evaluated:** Naive mean baseline, Piecewise Linear Interpolation, Cubic Spline Interpolation, Ridge Regression, and Random Forest Regression ($B=60$).

---

## Repository Structure

```text
├── src/                      # Modular Python engineering package
│   ├── preprocessing.py      # Physical bounds checking & chronological splitting
│   ├── models.py             # Numerical solvers and ML estimators
│   ├── evaluation.py         # Metrics (MAE, RMSE, R², E_max) & latency profiling
│   └── experiments.py        # Dropout duration sensitivity sweeps (10m - 8h)
├── tests/                    # Automated pytest suite (exactness & leakage verification)
├── 3001project.ipynb         # End-to-end reproducible analysis notebook
├── docs/                     # API documentation generated via pdoc
├── LICENSE                   # MIT License
└── README.md                 # Project reproduction and technical summary
