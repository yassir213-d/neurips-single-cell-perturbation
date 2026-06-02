# Single-Cell Drug Perturbation Prediction (NeurIPS)

An iterative machine learning workflow developed to solve the biological challenge of predicting high-dimensional gene expression responses ($18,211$ target variables) under cellular chemical perturbations using deep learning architectures.

## 📊 Performance Progression & Benchmarks

The project followed an empirical, data-driven approach to optimize features and models systematically, achieving a highly competitive score on the Kaggle Leaderboard:

| Experiment | Model Architecture | Hyperparameters | Validation MRMSE | Kaggle Public Score |
| :--- | :--- | :--- | :--- | :--- |
| **Exp 1** | Linear Ridge Regression | `alpha=1.0` | 1.30577 | -- |
| **Exp 2** | Multi-Layer Perceptron (MLP) | `max_iter=30` (Unconverged) | 1.29345 | -- |
| **Exp 3** | High-LR MLP | `learning_rate_init=0.005`, `batch_size=32` | 1.34354 (Overshot) | -- |
| **Exp 4** | **Optimized Stable MLP (Final)** | `learning_rate_init=0.001`, `max_iter=150` | **1.29834** | **0.858** |

## 🧠 Key Insights & Methodology

1. **Dimensionality & Scaling:** Handled an extreme multi-output regression matrix of $(614 \times 18211)$ gene targets by isolating cell topologies and target dynamics cleanly.
2. **Handling Non-Linearity:** Upgrading from a linear benchmark (Ridge) to a Multi-Layer Perceptron allowed the model to effectively map complex biological, non-linear drug-to-cell signaling routes.
3. **Hyperparameter Tuning Strategy:** Addressed gradient degradation and stochastic optimization issues by stabilizing the learning rate at `0.001` and expanding maximum training iterations to ensure full convergence.

## 💻 Tech Stack
* Python 3.12
* Pandas & NumPy (Data Processing)
* Scikit-Learn (MLP Regressor, Ridge, OneHotEncoder, KFold Cross-Validation)

## 💾 How to Run
Ensure you have the data structured in your directory, then run the full production pipeline:
```bash
python single_cell_pipeline.py
