# MMA3001 Sensor Dropout Reconstruction Project

A numerical methods and machine learning investigation evaluating 1D numerical interpolation versus multivariate machine-learning regression for sensor dropout reconstruction in smart infrastructure.

## Project Structure
- `src/`: Modular Python implementation containing data auditing, feature engineering, models, and evaluation routines.
- `tests/`: Automated unit test suite verifying interpolation accuracy and leak-free splitting.
- `docs/`: Auto-generated HTML documentation generated using `pdoc`.
- `results/`: Benchmark metrics tables (`.csv`), unit test reports, and figures (`.png`).

## Execution & Reproduction
1. Install dependencies:
   ```bash
   pip install pytest pdoc scikit-learn scipy pandas numpy matplotlib
   ```
