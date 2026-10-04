# Incremental Random Fourier Features Ridge Regression (iRFF-RR) for Robust sEMG Kinematics Decoding

This project is a part of my Master Thesis in the Automation Engineering faculty of the University of Bologna supervised by Prof. Roberto Meattini and co-supervised by Dr. Alex Pasquali. Presented with the title: "Implementation and Analysis of Interactive Myocontrol Algorithms Based on Incremental Ridge Regression with Random Fourier Features".   


An incremental learning and evaluation framework for surface electromyography (sEMG) continuous kinematic decoding under non-stationary conditions, physiological disturbances, and electrode faults.

This repository implements **Incremental Ridge Regression with Random Fourier Features (`iRFF_RR`)**, comparing it against standard linear **Incremental Ridge Regression (`iRR`)** and exact **Kernel Ridge Regression (`ExactKRR`)** baselines on benchmark datasets (**EMaGer** and **NinaPro DB8**).

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture & Mathematical Formulation](#architecture--mathematical-formulation)
  - [Random Fourier Features (RFF)](#random-fourier-features-rff)
  - [Incremental Updates & Downdates](#incremental-updates--downdates)
  - [Confidence Scoring](#confidence-scoring)
  - [Online Feature Standardization](#online-feature-standardization)
- [Simulated Perturbations & Disturbances](#simulated-perturbations--disturbances)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
  - [1. Running Single Experiments](#1-running-single-experiments)
  - [2. Hyperparameter Grid Search & Tuning](#2-hyperparameter-grid-search--tuning)
  - [3. Evaluating Clean vs. Shocked Scenarios](#3-evaluating-clean-vs-shocked-scenarios)
  - [4. Generating Comparison & Diagnostic Plots](#4-generating-comparison--diagnostic-plots)
- [CLI Reference](#cli-reference)
- [License](#license)

---

## Overview

Continuous kinematic estimation from sEMG signals often degrades over time due to skin impedance changes (sweat), muscle fatigue, sensor displacement (electrode shifts), and sensor failures. This project provides:

1. **Analytical incremental learning algorithms** capable of continuous sample-by-sample or block-by-block adaptation without costly retraining.
2. **Physiological & hardware fault simulation** injected directly into raw multi-channel sEMG signals before feature extraction.
3. **Rigorous benchmarking and evaluation pipelines** supporting prequential evaluation, validation splits, and multi-seed hyperparameter search.

---

## Key Features

- **Models Implemented (`iRFFRR.py`):**
  - `iRFF_RR`: Online non-linear approximation using Random Fourier Features combined with recursive Sherman-Morrison rank-1 updates.
  - `iRR`: Incremental standard linear Ridge Regression with exponential forgetting.
  - `RFF_RR` & `RR`: Analytical batch learning formulations.
  - `ExactKRR`: Non-linear baseline with RBF kernel and exponential sample weighting over finite history.
  - `RollingStandardizer`: Online channel-wise feature standardizer with exponential moving average (EMA) drift adaptation.
- **Signal Disturbances & Faults (`helper_functions.py`):**
  - Skin impedance / sweat noise (Gaussian white noise injection).
  - Muscle fatigue (Butterworth low-pass filtering and amplitude attenuation to mimic median frequency shift).
  - Electrode shift (spatial channel mixing between adjacent electrodes).
  - Catastrophic sensor failure (open-circuit noise replacement on designated channels).
- **Diagnostics & Benchmarking (`visualization_helpers.py`, `plot_*.py`):**
  - Error evolution over streams ($n\text{RMSE}$, $\text{RMSE}$, $R^2$, Pearson $r$).
  - Per-repetition Degrees of Freedom (DoF) scatter analysis.
  - Raw signal overlays and Power Spectral Density (PSD) analysis via Welch's method.
  - Execution time benchmarks (step-by-step latency vs. accuracy).

---

## Architecture & Mathematical Formulation

### Random Fourier Features (RFF)

To capture non-linear relationships without scaling quadratically with the number of samples, an RBF kernel $k(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2)$ is approximated using Bochner's theorem:

$$\mathbf{z}(\mathbf{x}) = \sqrt{\frac{2}{D}} \cos(\mathbf{\Omega} \mathbf{x} + \boldsymbol{\beta})$$

where:
- $\mathbf{\Omega} \sim \mathcal{N}(0, \sigma^2 \mathbf{I}_{D \times d})$
- $\boldsymbol{\beta} \sim \mathcal{U}(-\pi, \pi)^D$
- $D$ is the projection dimension (number of Fourier features).

### Incremental Updates

The model maintains an inverse regularized covariance matrix $\mathbf{A}^{-1} \in \mathbb{R}^{D \times D}$ and cross-covariance accumulator $\mathbf{B} \in \mathbb{R}^{D \times M}$:

- **Recursive Rank-1 Update (Sherman-Morrison):**

$$\mathbf{A}_{t}^{-1} = \frac{1}{\lambda_f} \left( \mathbf{A}_{t-1}^{-1} - \frac{\mathbf{A}_{t-1}^{-1} \mathbf{x}_t \mathbf{x}_t^\top \mathbf{A}_{t-1}^{-1}}{\lambda_f + \mathbf{x}_t^\top \mathbf{A}_{t-1}^{-1} \mathbf{x}_t} \right)$$

$$\mathbf{B}_t = \lambda_f \mathbf{B}_{t-1} + \mathbf{x}_t \mathbf{y}_t^\top$$

$$\mathbf{W}_t = \mathbf{A}_t^{-1} \mathbf{B}_t$$

where $\lambda_f \in (0, 1]$ represents the exponential forgetting factor.

- **Block Updates:**
  For batch updates over a repetition of size $N$, exponential sample weighting is applied within the block and inverted analytically.

### Confidence Scoring

For an incoming input vector $\mathbf{x}$, the predictive confidence metric correlates with the inverse covariance envelope:

$$c(\mathbf{x}) = \mathbf{x}^\top \mathbf{A}^{-1} \mathbf{x}$$

### Online Feature Standardization

The `RollingStandardizer` dynamically adjusts feature distributions using exponential moving statistics:

$$\boldsymbol{\mu}_t = (1 - \alpha) \boldsymbol{\mu}_{t-1} + \alpha \mathbf{x}_t$$

$$\boldsymbol{\sigma}^2_t = (1 - \alpha) (\boldsymbol{\sigma}^2_{t-1} + \alpha (\mathbf{x}_t - \boldsymbol{\mu}_t)^2)$$

$$\hat{\mathbf{x}}_t = \frac{\mathbf{x}_t - \boldsymbol{\mu}_t}{\sqrt{\boldsymbol{\sigma}^2_t} + \epsilon}$$

---

## Simulated Perturbations & Disturbances

All perturbations are applied to the raw EMG channels prior to filtering and feature extraction:

| Perturbation | Function | Implementation Details |
|---|---|---|
| **Skin Impedance / Sweat** | `simulate_skin_impedance_raw` | Additive zero-mean Gaussian white noise ($\sigma = 75.0$ for EMaGer, $\sigma = 10^{-5}$ for NinaPro). |
| **Muscle Fatigue** | `simulate_fatigue_raw` | Progressive amplitude linear decay to $60\%$ + 4th-order Butterworth low-pass filter ($f_c = 150\text{ Hz}$). |
| **Electrode Shift** | `simulate_electrode_shift_raw` | Convex spatial blending between adjacent electrode channels: $(1 - \alpha)\mathbf{x}_c + \alpha \bar{\mathbf{x}}_{\text{neighbours}}$. |
| **Sensor Failure** | `simulate_sensor_failure_raw` | Complete signal loss replaced with low-amplitude open-circuit hiss on designated pins. |

---

## Repository Structure

```text
.
├── helper_functions.py               # Raw EMG disturbance simulations and JSON I/O
├── iRFFRR.py                         # Core estimators: RR, RFF_RR, iRR, iRFF_RR, ExactKRR, RollingStandardizer
├── iRFFRR_test_and_hparam_tuning.py  # Pipeline driver: training, evaluation, grid search, and benchmarking
├── visualization_helpers.py          # Real-time plotting routines (scatter, errors, PSD, signal traces)
├── plot_model_comparison.py          # Aggregated metric evaluation across models
├── plot_training_scenarios.py        # Performance comparison across training sizes & scaler types
└── plot_execution_time.py            # Latency and step execution time visualizer
```

---

## Installation

### Prerequisites

Ensure you have Python 3.9+ installed along with standard scientific computing packages:

```bash
pip install numpy scipy scikit-learn matplotlib
```

### Dataset Support via `libemg`

The framework integrates with [`libemg`](https://github.com/LibEMG/libemg) for data handling and feature extraction:

```bash
pip install libemg
```

Ensure dataset files are organized in the root directory:
- `OneSubjectEMaGerDataset/`
- `NinaProDB8 s11/`
- `NinaProDB8 s12/`

---

## Usage Guide

### 1. Running Single Experiments

Run an experiment on `NinaPro_s12` using default parameters with repetition-level updates:

```bash
python iRFFRR_test_and_hparam_tuning.py --dataset NinaPro_s12 --model iRFF_RR --update-mode rep
```

Run a sample-by-sample update experiment with raw sEMG visualization:

```bash
python iRFFRR_test_and_hparam_tuning.py \
  --dataset EMaGer \
  --model iRFF_RR \
  --update-mode sample \
  --plot-channel 5 \
  --plot-rep 1 \
  --plot-psd
```

### 2. Hyperparameter Grid Search & Tuning

Perform multi-seed hyperparameter search over regularization $\lambda$, RFF kernel scale $\sigma$, and forgetting factor $\lambda_f$:

```bash
python iRFFRR_test_and_hparam_tuning.py \
  --dataset NinaPro_s12 \
  --model iRFF_RR \
  --tune \
  --seeds 42 \
  --eval-seeds 42,43,44 \
  --eval-top-k 10 \
  --objective mean_max \
  --save-best best_hparams_ninapro_s12.json
```

To load and evaluate saved optimal hyperparameters:

```bash
python iRFFRR_test_and_hparam_tuning.py \
  --dataset NinaPro_s12 \
  --model iRFF_RR \
  --use-best best_hparams_ninapro_s12.json
```

### 3. Evaluating Clean vs. Shocked Scenarios

Train on initial repetitions and evaluate on downstream repetitions without updating weights:

```bash
# Clean baseline: train on reps 0, 1; test on reps 2, 3, 4
python iRFFRR_test_and_hparam_tuning.py \
  --dataset EMaGer \
  --clean \
  --test-only-reps "2,3,4" \
  --save-run-results run_results

# Universal disturbance: inject impedance shock across all incremental repetitions
python iRFFRR_test_and_hparam_tuning.py \
  --dataset EMaGer \
  --shock-type-all-reps impedance \
  --test-only-reps "2,3,4" \
  --save-run-results run_results
```

### 4. Generating Comparison & Diagnostic Plots

#### Model Comparison ($n\text{RMSE}$ / $R^2$ / Pearson $r$)
```bash
python plot_model_comparison.py \
  --results-dir run_results \
  --dataset EMaGer \
  --condition impedance \
  --plot-metric nRMSE \
  --output-file model_comparison_impedance.png
```

#### Training Scenario Analysis (Standard vs. Rolling Scaler)
```bash
python plot_training_scenarios.py \
  --results-dir run_results_standard \
  --compare-rolling-dir run_results_rolling \
  --dataset EMaGer \
  --model iRFF_RR \
  --plot-metric R2 \
  --condition Clean
```

#### Execution Time Benchmarks
```bash
python plot_execution_time.py \
  --results-dir run_results \
  --dataset NinaPro_s12 \
  --condition Clean \
  --output-file latency_comparison.png
```

---

## CLI Reference

`iRFFRR_test_and_hparam_tuning.py` exposes several command-line flags:

| Argument | Options / Defaults | Description |
|---|---|---|
| `--dataset` | `EMaGer`, `NinaPro_s12`, `NinaPro_s11` (Default: `NinaPro_s12`) | Dataset to process. |
| `--model` | `iRFF_RR`, `iRR`, `ExactKRR` (Default: `iRFF_RR`) | Online regression model architecture. |
| `--update-mode` | `rep`, `sample` (Default: `rep`) | Batch repetition updates vs. online single-sample updates. |
| `--metric` | `nRMSE`, `RMSE` (Default: `nRMSE`) | Primary regression error metric. |
| `--tune` | Flag | Triggers hyperparameter grid search. |
| `--tune-metric` | `error`, `R2` (Default: `error`) | Optimization objective criterion for tuning. |
| `--objective` | `mean`, `max`, `mean_max` (Default: `mean_max`) | Repetition aggregation objective. |
| `--w-max` | Float $[0.0, 1.0]$ (Default: `0.5`) | Weight assigned to worst-case repetition in `mean_max`. |
| `--scaler` | `standard`, `rolling` (Default: `standard`) | Feature scaling method (`rolling` applies EMA drift mitigation). |
| `--clean` | Flag | Disables simulated disturbances on all repetitions. |
| `--shock-type-all-reps` | `impedance`, `fatigue`, `channel mixing`, `sensor failure`, etc. | Overrides dataset default mapping with a uniform shock. |
| `--test-only-reps` | Comma-separated ints (e.g. `"3,4"`) | Evaluates without updating model weights on these repetitions. |
| `--validation-reps` | Comma-separated ints | Repetitions reserved for tuning objective selection. |
| `--runs` | Integer (Default: `1`) | Repetitions for averaging computation latency metrics. |
| `--save-run-results` | Directory name (Default: `None`) | Saves run metrics and metadata to a timestamped JSON file. |

---

## License

This repository is distributed under the MIT License. See `LICENSE` for details.
