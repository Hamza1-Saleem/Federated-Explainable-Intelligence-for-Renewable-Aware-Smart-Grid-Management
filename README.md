# Federated Explainable Intelligence for Renewable-Aware Smart Grid Management: Energy Forecasting, Stability Assessment, and Adaptive Optimization

## Description

This project implements an end-to-end, privacy-preserving framework for renewable-aware smart grid management. It combines **federated learning (FedAvg)**, **explainable AI (SHAP)** and **risk-aware linear-programming optimization** across three tasks: day-ahead electricity load forecasting, short-term solar PV forecasting, and smart-grid stability classification.

The study systematically benchmarks centralized baselines (Persistence, Random Forest, XGBoost, Logistic Regression), a centralized neural network, local-only training and federated training under identical experimental conditions, emphasizing predictive performance, robustness to data heterogeneity, communication cost and explainability. The trained forecasting and stability models are then used inside a 24-hour micro-grid dispatch simulation to quantify the downstream benefit of stability-aware energy optimization.

## Dataset Information

Only the following three datasets are used. They are kept strictly separate throughout the pipeline.

| Dataset (file) | Description | Task | Records |
|----------------|-------------|------|---------|
| `AEP_hourly.csv` | Hourly AEP transmission load (MW), 2004-2018 | 24-hour-ahead load forecasting (regression) | 121,273 raw rows, 120,960 after feature engineering |
| `Load_Demand.csv` | One day (29 Oct 2021) of ~20 s PV and load measurements: `Ppv_actual`, `Qpv_actual`, `Pload_actual`, `Qload_actual` | 30-minute-ahead solar PV forecasting (regression) | 3,546 raw rows, 1,350 after 1-minute resampling and feature engineering |
| `smart_grid_stability_augmented.csv` | Decentralized smart-grid control simulation: `tau1-4` (reaction times), `p1-4` (power balance), `g1-4` (price elasticity), `stab` (stability margin), `stabf` (label) | Binary stability classification (positive class = *unstable*) | 60,000 rows (38,280 unstable, 21,720 stable) |

**Data splits**

| Dataset | Strategy | Train | Validation | Test |
|---------|----------|-------|------------|------|
| AEP | Chronological 70 / 15 / 15 | 84,672 | 18,144 | 18,144 |
| PV | Chronological 50 / 15 / 35 | 675 | 202 | 473 |
| Grid stability | Stratified random 70 / 15 / 15 | 42,000 | 9,000 | 9,000 |

**Leakage safeguards:** all AEP lags are at least 24 h; all PV inputs are from at least 30 minutes before the target; `stab` and `stabf` are never used as model inputs.

## Code Overview

The notebook is a single, self-contained Colab cell organised in six steps:

1. **Data Preprocessing & Feature Engineering**
   - AEP: sorting, de-duplication, hourly regularisation, time-interpolation of gaps up to 6 h; lag features (24-336 h), rolling statistics, cyclical hour/day-of-year encodings, weekend and US federal holiday flags.
   - PV: 1-minute resampling, interpolation, lagged PV and load features, rolling mean/std, 5-minute delta, clear-sky proxy from approximate sunrise/sunset.
   - Grid: duplicate removal, binary label creation, correlation analysis.
   - Standardisation uses global mean/std, equivalent to exact federated aggregation of client sums.
2. **Centralized Baselines**
   - Persistence, Random Forest (200 trees) and XGBoost (600 trees, early stopping) for forecasting.
   - Logistic Regression, Random Forest and XGBoost for stability classification.
3. **Federated Learning (FedAvg)**
   - MLP with two hidden layers of 64 units (ReLU), Adam optimizer, MSE loss (regression) or BCE-with-logits (classification).
   - 5 simulated clients under non-IID Dirichlet partitioning (alpha = 0.5), compared with a centralized NN and local-only training.
   - Convergence, rounds-to-target and communication cost tracking; heterogeneity quantified with a Jensen-Shannon index.
4. **Explainable AI (SHAP)**
   - `TreeExplainer` for XGBoost, `KernelExplainer` for neural networks.
   - Explanation consistency (Spearman rank correlation and top-5 Jaccard) across XGBoost, centralized NN, FedAvg and individual clients.
   - SHAP-guided feature pruning (top 60%) to build the **Full framework** arm with fewer parameters and lower communication.
5. **Adaptive Energy Optimization (Simulation)**
   - 24-hour hourly dispatch LP (SciPy HiGHS) over solar, battery, grid and unmet demand, with a stability-risk penalty and an inverter-share cap.
   - Compared with a rule-based, risk-blind baseline across 40 Monte-Carlo days.
6. **Ablation & Robustness Analysis**
   - Client count and non-IID level, missing values (0 / 10 / 30%), class imbalance (50 / 20 / 10 / 5% unstable), forecast-error multiplier, consolidated ablation table, and computational cost.

All analyses are implemented in Python with PyTorch, scikit-learn, XGBoost and SHAP, with plotting via Matplotlib and Seaborn.

## Usage Instructions

### 1. Dataset Preparation

Place all three CSV files in one folder (for example in Google Drive):

```
/path/to/project_folder/
├── AEP_hourly.csv
├── Load_Demand.csv
└── smart_grid_stability_augmented.csv
```

Update the paths in the configuration section of the notebook:

```python
DRIVE_DIR = '/content/drive/MyDrive/smart_grid_fl'
FILES = dict(
    aep  = f'{DRIVE_DIR}/AEP_hourly.csv',
    pv   = f'{DRIVE_DIR}/Load_Demand.csv',
    grid = f'{DRIVE_DIR}/smart_grid_stability_augmented.csv',
)
```

### 2. Run Training and Evaluation

Open the notebook in Google Colab (or Jupyter) and select **Run all**. Authorise the Google Drive mount when prompted.

For a quick smoke test, set `FAST_MODE = True` (fewer seeds, rounds, trees and simulated days), then switch back to `False` for the full experiment.

### 3. Outputs Generated

All results are saved to `DRIVE_DIR/results/`:

| Output | Description |
|--------|-------------|
| `all_results_long.csv` | Every metric from every experiment with means and 95% confidence intervals |
| `ablation_table.csv` | Centralized vs. Local-only vs. FedAvg vs. Full framework |
| `computational_cost.csv` | Wall-clock time per stage |
| `shap_importance_{aep,pv,grid}.csv` | Normalised mean absolute SHAP per feature and model |
| Step 5 policy table (CSV) | Optimization KPIs per policy |
| `figs/*.png` | Correlation heatmaps, forecast plots, confusion matrices, convergence curves, SHAP bar and beeswarm plots, policy comparisons, forecast-error sensitivity |

### 4. Key Configuration Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `FAST_MODE` | `False` | Quick smoke test when `True` |
| `SEED` | `42` | Base random seed |
| `N_SEEDS` / `N_SEEDS_SW` | `5` / `3` | Seeds for main comparison / robustness sweeps |
| `RF_TREES` | `200` | Random Forest trees |
| `PV_H` | `30` | PV forecast horizon (minutes) |
| `N_CLIENTS` | `5` | Simulated federated clients |
| `TASK_CFG` | per task | Rounds, local epochs, learning rate, batch size (AEP 20 rounds, PV 30, grid 25) |
| `XAI_ROWS`, `XAI_NSAMP`, `XAI_BG` | `100`, `150`, `20` | SHAP rows, Kernel samples, background size |
| `SIM_DAYS` | `40` | Simulated days for the optimization study |

## Requirements

| Library | Version (Recommended) |
|---------|-----------------------|
| Python | ≥ 3.9 |
| PyTorch | ≥ 2.0 |
| XGBoost | ≥ 1.7 |
| SHAP | ≥ 0.42 |
| numpy | ≥ 1.24 |
| pandas | ≥ 2.0 |
| scipy | ≥ 1.10 |
| scikit-learn | ≥ 1.2 |
| matplotlib | ≥ 3.7 |
| seaborn | ≥ 0.12 |

Install dependencies:

```bash
pip install torch xgboost shap numpy pandas scipy scikit-learn matplotlib seaborn
```

The notebook automatically installs `shap`, `xgboost`, `scipy` and `seaborn` if they are missing. A GPU is optional: CUDA is used when available, otherwise the notebook runs on CPU.

## Methodology Summary

| Component | Description |
|-----------|-------------|
| Tasks | Day-ahead load forecasting, 30-min PV forecasting, grid stability classification |
| Centralized Baselines | Persistence, Random Forest, XGBoost, Logistic Regression |
| Neural Model | MLP (64-64), Adam, MSE / BCE loss |
| Federated Strategy | FedAvg with size-weighted aggregation, 5 clients, Dirichlet non-IID partitions |
| Heterogeneity Sweep | IID, Dirichlet alpha = 1.0 / 0.5 / 0.3 / 0.1, temporal split, K = 3 / 5 / 10 clients |
| Explainability | SHAP (Tree and Kernel explainers), consistency via Spearman and top-5 Jaccard |
| Full Framework | FedAvg on top 60% SHAP-selected features plus risk-aware optimization |
| Optimization | Hourly LP (HiGHS) with stability-risk penalty and inverter-share cap |
| Evaluation Metrics | Regression: MAE, RMSE, R². Classification: Precision, Recall, F1, PR-AUC, ROC-AUC |
| Statistical Reporting | 95% t-intervals over seeds, moving-block bootstrap on test sets, intervals over simulated days |
| Robustness Analysis | Non-IID level, client count, missing values, class imbalance, forecast-error multiplier |

## Performance Summary

All federated and neural results below use 5 seeds, 5 clients and Dirichlet alpha = 0.5 (mean values).

### Day-Ahead Load Forecasting (AEP)

| Model | MAE (MW) | RMSE (MW) | R² | Notes |
|-------|----------|-----------|----|-------|
| Persistence | 926 | 1221 | 0.758 | Naive 24 h baseline |
| Random Forest | 705.5 | 933 | 0.859 | Best MAE |
| XGBoost | 714.9 | 932.6 | 0.859 | Best RMSE among centralized models |
| Central-NN | 743.4 | 972.8 | 0.847 | Reference NN |
| **FedAvg** | 757.6 | 967.0 | 0.849 | Matches centralized NN without sharing raw data |
| Full (SHAP-pruned FedAvg) | 749.8 | 964.6 | 0.849 | 10 of 16 features, lower communication |
| Local-only | 852.5 | 1094.0 | 0.805 | Weakest; limited local data |

### Grid Stability Classification

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|-------|-----------|--------|----|--------|---------|
| Logistic Regression | 0.837 | 0.880 | 0.858 | 0.931 | 0.889 |
| Random Forest | 0.951 | 0.968 | 0.959 | 0.995 | 0.991 |
| XGBoost | 0.981 | 0.990 | 0.986 | 0.999 | 0.999 |
| Central-NN | 0.981 | 0.982 | 0.982 | 0.999 | 0.998 |
| **FedAvg** | 0.970 | 0.981 | 0.975 | 0.998 | 0.997 |
| Full (SHAP-pruned FedAvg) | 0.975 | 0.985 | 0.980 | 0.999 | 0.998 |
| Local-only | 0.874 | 0.808 | 0.826 | 0.980 | 0.966 |

### Solar PV Forecasting (30 min ahead, single day, indicative only)

| Model | MAE (kW) | RMSE (kW) | Notes |
|-------|----------|-----------|-------|
| Persistence | 86.9 | 194.0 | Best overall on this single-day test set |
| Random Forest | 264.6 | 509.2 | Does not beat persistence |
| XGBoost | 327.3 | 415.2 | Does not beat persistence |
| Central-NN | 247.1 | 497.0 | |
| FedAvg | 243.0 | 462.1 | |
| Full (SHAP-pruned FedAvg) | 221.8 | 435.2 | |
| Local-only | 229.0 | 415.1 | |

> The PV file covers a single day, so the test period (afternoon and night) is not representative. Treat PV conclusions as indicative only.

### Federated Learning Efficiency

| Task | FedAvg Comm. (MB) | Full Framework (MB) | Raw-Data Upload (MB) | Rounds to within 5% of Central |
|------|-------------------|---------------------|----------------------|-------------------------------|
| AEP | 4.25 | 3.94 | 5.76 | 5 |
| PV | 6.38 | 5.92 | 0.05 | 1 |
| Grid | 5.06 | 4.80 | 2.18 | 1 |

FedAvg communication is lower than raw-data upload only for AEP. For the small PV and grid datasets, uploading raw data is cheaper in bytes, so the benefit of federation there is privacy, not bandwidth.

### Robustness to Data Heterogeneity (primary metric, FedAvg vs. Local-only)

| Setting | AEP RMSE (FedAvg / Local) | PV RMSE (FedAvg / Local) | Grid PR-AUC (FedAvg / Local) |
|---------|---------------------------|--------------------------|------------------------------|
| IID, K = 5 | 931 / 939 | 395 / 400 | 0.998 / 0.997 |
| Dirichlet 1.0, K = 5 | 937 / 986 | 414 / 412 | 0.998 / 0.994 |
| Dirichlet 0.3, K = 5 | 1056 / 1292 | 362 / 416 | 0.998 / 0.916 |
| Dirichlet 0.1, K = 5 | 1203 / 1502 | 377 / 714 | 0.976 / 0.841 |
| Temporal, K = 5 | 978 / 1109 | 367 / 555 | n/a |
| Dirichlet 0.5, K = 3 | 972 / 1070 | 411 / 436 | 0.998 / 0.962 |
| Dirichlet 0.5, K = 10 | 983 / 1120 | 376 / 397 | 0.995 / 0.956 |

### Missing-Value and Class-Imbalance Robustness

| Experiment | Setting | Result |
|------------|---------|--------|
| Missing values (AEP RMSE, FedAvg) | 0% / 10% / 30% | 957 / 1053 / 1178 |
| Missing values (Grid PR-AUC, FedAvg) | 0% / 10% / 30% | 0.999 / 0.967 / 0.911 |
| Class imbalance (Grid PR-AUC, FedAvg) | 50% / 20% / 10% / 5% unstable | 0.996 / 0.969 / 0.951 / 0.848 |
| Class imbalance (Grid PR-AUC, Central-NN) | 50% / 20% / 10% / 5% unstable | 0.998 / 0.987 / 0.974 / 0.946 |

### Explainability (SHAP)

| Task | Top Features (FedAvg) | Spearman: FedAvg vs. Central |
|------|-----------------------|------------------------------|
| AEP | `lag_24`, `is_weekend`, `same_hour_7d_mean`, `dow` | 0.859 |
| PV | `clearsky_proxy`, `Ppv_roll30_std`, `Ppv_lag40`, `Ppv_lag35` | 0.941 |
| Grid | `tau1`, `g2`, `g4`, `tau2` | 0.972 |

### Adaptive Optimization (mean over 40 simulated days)

| Policy | Total Cost ($) | RE Utilisation (%) | Unmet Energy (kWh) | Stability Incidents |
|--------|----------------|--------------------|--------------------|---------------------|
| Baseline (rule-based, risk-blind) | 6154.0 | 98.8 | 533.3 | 5.78 |
| Proposed, Central-NN | 6026.4 | 82.0 | 255.4 | 0.05 |
| Proposed, Local-only | 5932.0 | 82.2 | 199.6 | 0.15 |
| **Proposed, FedAvg** | 6021.6 | 82.0 | 248.4 | 0.05 |
| Proposed, Full framework | 5994.6 | 81.9 | 232.8 | 0.05 |
| Proposed, Oracle | 5768.4 | 86.0 | 141.0 | 0.00 |
| Ablation: FedAvg, risk term OFF | 5662.5 | 96.4 | 279.5 | 5.85 |

## Visualization Examples

- **Correlation Heatmaps**: target vs. lag features (AEP), PV feature correlations, grid feature correlations
- **Forecast Plots**: last 7 test days for AEP and the PV test period (actual vs. Persistence, Random Forest, XGBoost)
- **Confusion Matrices**: Logistic Regression, Random Forest and XGBoost for stability classification
- **Convergence Curves**: FedAvg test metric per communication round vs. Central-NN and Local-only reference lines
- **SHAP Plots**: normalised importance bar charts and FedAvg beeswarm summary plots for each task
- **Optimization Policy Comparison**: cost, renewable utilisation, unmet demand and incidents with confidence intervals
- **Forecast-Error Sensitivity**: optimizer KPIs vs. the forecast-error multiplier

## Evaluation Environment

Developed and tested on:
- Platform: Google Colab (CPU run documented; CUDA used automatically if available)
- Framework: PyTorch, scikit-learn, XGBoost, SHAP, SciPy (HiGHS LP solver)
- Libraries: NumPy, Pandas, Matplotlib, Seaborn

Wall-clock timings for every stage are saved to `computational_cost.csv`. The AEP Random Forest fit and the federated sweeps dominate the runtime.

## Results and Discussion

- **FedAvg matches centralized learning without pooling raw data.** On AEP, FedAvg reached an RMSE of 967 MW vs. 973 MW for the centralized NN; on grid stability, PR-AUC was 0.998 vs. 0.999.
- **Federation clearly beats local-only training under heterogeneity.** The gap widens as clients become more non-IID (grid PR-AUC at Dirichlet alpha = 0.1: 0.976 for FedAvg vs. 0.841 for local-only).
- **Tree ensembles remain strong on tabular data.** XGBoost achieved the best centralized scores on AEP RMSE and grid stability (F1 0.986, PR-AUC 0.999), but requires pooled data.
- **SHAP-guided pruning retains accuracy.** Keeping 10/16 AEP features, 10/16 PV features and 8/12 grid features matched or slightly improved FedAvg accuracy while reducing model size and communication.
- **Explanations are consistent across federated and centralized models.** Rank correlations of 0.86-0.97 indicate FedAvg uses its inputs similarly to the centralized model.
- **Stability-aware optimization sharply reduces risk.** Compared with the risk-blind baseline, the proposed planner reduced stability incidents from about 5.8 to about 0.05 per day and unmet demand by roughly half, at the cost of lower renewable utilisation (about 82% vs. 99%). Turning the risk term off removed the incident reduction.
- **Forecast source matters little in the simulation.** Central, federated and pruned models gave similar optimization outcomes; only the Oracle clearly improved results, showing the remaining headroom from better forecasts.
- **Robustness has limits.** Performance degrades with 30% missing values and at 5% unstable prevalence, where federated models lost more PR-AUC than the centralized NN.
- **PV findings are inconclusive.** With a single day of data, none of the learned models beat persistence on the test set.

## Limitations

- The PV dataset covers a single day, so no split contains a full diurnal cycle; the test set is afternoon plus night, which inflates R². Daylight-only metrics are printed, and PV conclusions are not generalisable.
- The three datasets come from unrelated systems. The optimization study couples them through documented synthetic scaling and a stress-to-stability mapping, so it demonstrates framework behaviour rather than validated real-grid performance.
- Federated clients are simulated partitions of a single dataset. There is no real network, stragglers, differential privacy or secure aggregation, and communication is uncompressed.
- Confidence intervals use few seeds (5 main, 3 sweeps) and are therefore wide; increase `N_SEEDS` for publication.
- SHAP explains model behaviour, not causality, and correlated features share credit.
- Grid stability labels come from a deterministic simulated control model, so PR-AUC is near ceiling; the imbalance and non-IID results are more informative.
- Optimizer assumptions (tariffs, battery size, value of lost load, share limit) are illustrative.
- Future work should include multi-day, multi-site PV data, real federated deployment with privacy mechanisms, coupled real-world datasets, richer federated algorithms (for example FedProx) and quantitative validation of explanations.

## Conclusion

The proposed framework delivers:
- Federated forecasting and stability models that match centralized neural networks while keeping raw data local
- Clear advantages over local-only training under heterogeneous (non-IID) client data
- Consistent, SHAP-based explanations across federated, centralized and client models, plus explanation-guided feature pruning
- A risk-aware optimization layer that reduces simulated stability incidents from about 5.8 to about 0.05 per day

This pipeline provides a transparent, reproducible foundation for privacy-preserving, explainable and stability-aware energy management, subject to the limitations above.

## Citation

```bibtex
@misc{federated_explainable_smart_grid,
  title  = {Federated Explainable Intelligence for Renewable-Aware Smart Grid Management:
            Energy Forecasting, Stability Assessment, and Adaptive Optimization},
  year   = {2026},
  note   = {Code and notebook}
}
```

## License

Add your preferred license (for example MIT). Check the licenses of the three datasets before redistributing them.
