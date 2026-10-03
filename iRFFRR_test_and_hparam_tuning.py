"""
Hyperparameter search and test for iRFF_RR on the EMaGer and NinaPro DB8 pipeline.

Usage:
  python iRFFRR_test_and_hparam_tuning.py --dataset EMaGer --model iRR  # single run of traditional iRR on EMaGer with default hyperparameters
  python iRFFRR_test_and_hparam_tuning.py --dataset NinaPro_s12 --tune --model iRR # full hyperparameter tuning of traditional iRR on NinaPro DB8 s12
  python iRFFRR_test_and_hparam_tuning.py --dataset EMaGer  # single run with defaults for EMaGer
  python iRFFRR_test_and_hparam_tuning.py --dataset NinaPro_s12  # single run with defaults for NinaPro DB8 s12
  python iRFFRR_test_and_hparam_tuning.py --dataset NinaPro_s11  # single run with defaults for NinaPro DB8 s11
  python iRFFRR_test_and_hparam_tuning.py --dataset EMaGer --update-mode rep # single run with rep-by-rep updates instead of sample-by-sample (default)
  python iRFFRR_test_and_hparam_tuning.py --dataset EMaGer --update-mode rep --clean # single run with rep-by-rep updates on EMaGer using only clean reps (no shocks)
  python iRFFRR_test_and_hparam_tuning.py --plot-best  # single run with best plot (default NinaPro_s12)
  python iRFFRR_test_and_hparam_tuning.py --dataset EMaGer --metric nRMSE  # evaluate using nRMSE instead of RMSE
  python iRFFRR_test_and_hparam_tuning.py --w-max 0.5  # single run with w-max 0.5
  python iRFFRR_test_and_hparam_tuning.py --objective mean_max  # single run with mean_max objective
  python iRFFRR_test_and_hparam_tuning.py --tune    # full tune
  python iRFFRR_test_and_hparam_tuning.py --tune --seeds 42 --eval-seeds 42,43,44 --eval-top-k 10  # fast tune (1 seed) + verify top-10 on 3 seeds
  python iRFFRR_test_and_hparam_tuning.py --tune --seeds 42 --tune-on-first-n 2   # tune only on first 2 reps
  python iRFFRR_test_and_hparam_tuning.py --save-best best_hparams_*.json  # save best JSON
  python iRFFRR_test_and_hparam_tuning.py --use-best best_hparams_*.json  # single run from saved best JSON
  python iRFFRR_test_and_hparam_tuning.py --dataset EMaGer --plot-channel 5 --plot-rep 1  # plot channel 5 of rep 1 in detail for EMaGer (which has a shock on channel 5 in rep 1)
  python iRFFRR_test_and_hparam_tuning.py --dataset EMaGer --tune-metric R2 --tune --seeds 42 --eval-seeds 42,43,44 --eval-top-k 10 # tune based on R2 instead of error
  python iRFFRR_test_and_hparam_tuning.py --dataset EMaGer --plot-scatter # run and show scatter plots
  python iRFFRR_test_and_hparam_tuning.py --dataset EMaGer --test-only-reps "3,4" # Train on reps 0,1,2 and test on 3,4
  python iRFFRR_test_and_hparam_tuning.py --dataset EMaGer --tune --validation-reps "2" --test-only-reps "3,4" # Tune on rep 2, test on 3,4
  python iRFFRR_test_and_hparam_tuning.py --dataset EMaGer --shock-type-all-reps impedance --test-only-reps "2,3,4" # Train on rep 0 (clean) and 1 (impedance), test on 2,3,4 (impedance)

Clean Reps Scenarios:
    Train on reps 0,1; test on all subsequent reps
    python iRFFRR_test_and_hparam_tuning.py --dataset EMaGer --clean --test-only-reps "2,3,4"

    Train on reps 0,1,2; test on all subsequent reps
    python iRFFRR_test_and_hparam_tuning.py --dataset EMaGer --clean --test-only-reps "3,4"

Shocked Reps Scenarios:
    Train on reps 0 (clean), 1 (shocked); test on all subsequent reps
    python iRFFRR_test_and_hparam_tuning.py --dataset EMaGer --shock-type-all-reps impedance --test-only-reps "2,3,4"

    Train on reps 0 (clean), 1 (shocked), 2 (shocked); test on all subsequent reps
    python iRFFRR_test_and_hparam_tuning.py --dataset EMaGer --shock-type-all-reps impedance --test-only-reps "3,4"
"""

from __future__ import annotations
import argparse
import itertools
import json
import time
import random
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from libemg.datasets import *
from libemg.data_handler import RegexFilter
from libemg.datasets import OneSubjectEMaGerDataset, NinaproDB8
from libemg.filtering import *
from libemg.feature_extractor import FeatureExtractor
from sklearn.preprocessing import StandardScaler
from visualization_helpers import plot_dof_scatter_by_repetition, plot_semg_and_predictions, plot_experiment, plot_psd_comparison

from helper_functions import (
    simulate_electrode_shift_raw,
    simulate_fatigue_raw,
    simulate_sensor_failure_raw,
    simulate_skin_impedance_raw,
    save_best_hparams_json,
    load_best_hparams_json,
)
from iRFFRR import iRFF_RR, iRR, ExactKRR
from iRFFRR import iRFF_RR, iRR, ExactKRR, RollingStandardizer

EMAGER_ROOT = "OneSubjectEMaGerDataset"
NINAPRO_ROOT = "NinaProDB8"

DEFAULT_HPARAMS = {
    "lambda_val": 0.1,  # 0.1
    "sigma": 1.5, # 1.5
    "forgetting_factor": 1.0, # 1.0
}

TUNE_GRID = {
    "lambda_val": [1e-4, 1e-3, 1e-2, 0.1, 1.0],
    "sigma": [0.01, 0.05, 0.1, 0.25, 0.5], 
    "forgetting_factor": [1.0, 0.9995, 0.999, 0.995, 0.99],
}

def load_and_prepare_data(dataset_name: str = "NinaPro_s12", clean: bool = False, shock_type_all_reps: str | None = None) -> dict:
    
    if dataset_name == "EMaGer":
        ds = OneSubjectEMaGerDataset(dataset_folder=EMAGER_ROOT)
        data = ds.prepare_data()
        odh_list = [data["All"]]
        ds_fs = 1010
        shock_mapping = {1: "impedance", 2: "fatigue", 3: "channel mixing", 4: "sensor failure"}

        # Dataset-specific shock parameters
        impedance_noise_std = 75.0  # matches EMaGer's scale (~1500 amplitude) for realistic sweat impedance
        heavy_shift_strength = 0.2
        sensor_failure_noise_std = 300.0  # mimics massive open-circuit floating pin hiss
        failure_channels_dict = {"sensor failure": [1, 9, 15, 28, 60]} # 5 channels failure

    elif dataset_name in ["NinaPro_s12", "NinaPro_s11"]:
        subject_num = 12 if dataset_name == "NinaPro_s12" else 11
        folder_name = f"NinaProDB8 s{subject_num}"
        
        if not os.path.exists(folder_name):
            raise FileNotFoundError(f"\n[Error] Directory '{folder_name}' was not found. Please ensure the CSV files are placed inside it.\n")
            
        odh = OfflineDataHandler()
        odh.get_data(
            folder_location=folder_name,
            regex_filters=[
                RegexFilter(left_bound="C_", right_bound="_R", values=[str(i) for i in range(100)], description="labels"),
                RegexFilter(left_bound="R_", right_bound=".csv", values=[str(i) for i in range(20)], description="reps")
            ]
        )
            
        # Temporarily separate the 18 DOF kinematics from the 16 EMG channels
        # so shocks and filters only apply to the EMG.
        odh.temp_kinematics = []
        for i in range(len(odh.data)):
            odh.temp_kinematics.append(odh.data[i][:, 16:])
            odh.data[i] = odh.data[i][:, :16]
            
        odh_list = [odh]
        ds_fs = 2000
        if dataset_name == "NinaPro_s11":
            # Rep 1 is warm, Reps 2-3 are clean, shocks start at Rep 4
            shock_mapping = {4: "impedance", 5: "fatigue", 6: "light channel mixing", 7: "channel mixing", 
                             8: "sensor_failure 1", 9: "sensor_failure 2", 10: "sensor_failure 3"}
        else:
            # For s12: Rep 0 is warm, Reps 1-2 are clean, shocks start at Rep 3
            shock_mapping = {3: "impedance", 4: "fatigue", 5: "light channel mixing", 6: "channel mixing", 
                             7: "sensor_failure 1", 8: "sensor_failure 2", 9: "sensor_failure 3"}
        
        impedance_noise_std = 0.00001  # ~5% noise masking for realistic sweat (can be made 0.00002)
        light_shift_strength = 0.1
        heavy_shift_strength = 0.2
        sensor_failure_noise_std = 0.001  # massive noise to completely drown the 0.0002 signal
        failure_channels_dict = {
            "sensor_failure 1": [1, 5, 9],
            "sensor_failure 2": [2, 6, 10],
            "sensor_failure 3": [3, 7, 11]
        }
    
    else:
        raise ValueError("dataset_name must be 'EMaGer', 'NinaPro_s12', or 'NinaPro_s11'")

    # --- Determine warm-up and incremental reps before processing ---
    all_rep_labels = []
    for odh in odh_list:
        # This gets all rep numbers from the file names
        all_rep_labels.extend([int(np.unique(np.atleast_1d(r))[0]) for r in odh.reps])
    unique_reps = sorted(list(set(all_rep_labels)))

    if dataset_name == "NinaPro_s11":
        warm_rep_index = 1
        inc_reps = [r for r in unique_reps if 2 <= r <= 10]
    else:
        warm_rep_index = int(unique_reps[0])
        temp_inc_reps = [r for r in unique_reps if r != warm_rep_index]
        inc_reps = temp_inc_reps[:9]  # Cap at 9 incremental reps for a total of 10

    # --- Override shock_mapping if requested ---
    if clean:
        shock_mapping = {}

    if shock_type_all_reps:
        shock_mapping = {rep: shock_type_all_reps for rep in inc_reps}
        print(f"[Info] Overriding shocks. Applying '{shock_type_all_reps}' to all incremental reps: {inc_reps}")

    raw_emg_by_rep = {}

    # --- Apply shocks to raw EMG data ---
    for odh in odh_list:
        raw_data_list = odh.data
        rep_labels = [int(np.unique(np.atleast_1d(r))[0]) for r in odh.reps]

        for idx, rep_label in enumerate(rep_labels):
            raw_emg = raw_data_list[idx]
            shock_type = shock_mapping.get(rep_label)

            if shock_type == "impedance":
                raw_emg = simulate_skin_impedance_raw(raw_emg, noise_std=impedance_noise_std)
            elif shock_type == "fatigue":
                raw_emg = simulate_fatigue_raw(raw_emg, ds_fs)
            elif shock_type == "light channel mixing":
                raw_emg = simulate_electrode_shift_raw(raw_emg, shift_strength=light_shift_strength)
            elif shock_type == "channel mixing":
                raw_emg = simulate_electrode_shift_raw(raw_emg, shift_strength=heavy_shift_strength)
            elif shock_type in failure_channels_dict:
                raw_emg = simulate_sensor_failure_raw(raw_emg, failed_channels=failure_channels_dict[shock_type], noise_std=sensor_failure_noise_std)

            raw_data_list[idx] = raw_emg
            if rep_label not in raw_emg_by_rep:
                raw_emg_by_rep[rep_label] = []
            raw_emg_by_rep[rep_label].append(raw_emg.copy())
            
    # Stitch the classes together so the plotting tools can view the entire repetition
    for rep in raw_emg_by_rep:
        raw_emg_by_rep[rep] = np.vstack(raw_emg_by_rep[rep])

    # --- Filtering ---
    fil = Filter(sampling_frequency=ds_fs)
    # 50 Hz for DB8 and 60 Hz for EMaGer
    notch_freq = 60 if dataset_name == "EMaGer" else 50
    filter_dict = [
        {"name": "notch", "cutoff": notch_freq, "bandwidth": 3, "fs": ds_fs, "order": 4},
        {"name": "lowpass", "cutoff": 500, "fs": ds_fs, "order": 4},
        {"name": "highpass", "cutoff": 20, "fs": ds_fs, "order": 4},
    ]
    for f in filter_dict:
        fil.install_filters(f)

    for odh in odh_list:
        fil.filter(odh)

    # --- Windowing & Feature extraction ---
    window_size = int(0.2 * ds_fs)
    window_inc = int(0.05 * ds_fs)
    fe = FeatureExtractor()
    
    X_all, Y_all, reps_all = [], [], []

    for odh in odh_list:
        if dataset_name in ["NinaPro_s12", "NinaPro_s11"]:
            # Recombine EMG and Kinematics before windowing
            for i in range(len(odh.data)):
                odh.data[i] = np.hstack((odh.data[i], odh.temp_kinematics[i]))
                
        windows, meta = odh.parse_windows(window_size, window_inc, metadata_operations={"labels": "last_sample", "reps": "last_sample"})
        
        if dataset_name in ["NinaPro_s12", "NinaPro_s11"]:
            # Split windows back into EMG (16 channels) and Kinematics (18 channels)
            emg_windows = windows[:, :, :16]
            y_targets = windows[:, -1, 16:]  # Use last sample of kinematics as regression target
            X = fe.extract_features(["RMS"], emg_windows, array=True)
            Y_all.append(y_targets)
        else:
            X = fe.extract_features(["RMS"], windows, array=True)
            y_labels = meta["labels"]
            if y_labels.ndim == 1:
                y_labels = np.expand_dims(y_labels, axis=1)
            Y_all.append(y_labels)
            
        X_all.append(X)
        reps_all.append(meta["reps"])

    # Combining train/test splits into one continuous array
    X_concat = np.concatenate(X_all, axis=0)
    Y_concat = np.concatenate(Y_all, axis=0)
    reps_concat = np.concatenate(reps_all, axis=0)

    # Isolating the warm-up repetition
    rep0_mask = reps_concat == warm_rep_index
    X_rep0 = X_concat[rep0_mask]
    Y_rep0 = Y_concat[rep0_mask]

    X_reps = [X_concat[reps_concat == r] for r in inc_reps]
    Y_reps = [Y_concat[reps_concat == r] for r in inc_reps]

    # --- Standard scaling the features ---
    scaler = StandardScaler()
    scaler.fit(X_rep0)  # scale only on the batch data

    # Pre-scale once
    X_rep0_scaled = scaler.transform(X_rep0)
    X_reps_scaled = [scaler.transform(x) if x.shape[0] else x for x in X_reps]

    return {
        "X_rep0": X_rep0,
        "Y_rep0": Y_rep0,
        "X_reps": X_reps,
        "Y_reps": Y_reps,
        "inc_reps": inc_reps,
        "scaler": scaler,
        "X_rep0_scaled": X_rep0_scaled,
        "X_reps_scaled": X_reps_scaled,
        "input_dim": X_rep0.shape[1],
        "output_dim": Y_rep0.shape[1],
        "shock_mapping": shock_mapping,
        "raw_emg_by_rep": raw_emg_by_rep,
        "warm_rep_index": warm_rep_index,
        "ds_fs": ds_fs,
    }


def composite_objective(rep_metrics, mode: str = "mean_max", w_max: float = 0.5, higher_is_better: bool = False) -> float:
    r = np.asarray(rep_metrics, dtype=float)
    if r.size == 0:
        return float("-inf") if higher_is_better else float("inf")
    if mode == "mean":
        return float(np.mean(r))
    if mode == "max":
        return float(np.min(r)) if higher_is_better else float(np.max(r))
    w_max = float(np.clip(w_max, 0.0, 1.0))
    if higher_is_better:
        return float((1.0 - w_max) * np.mean(r) + w_max * np.min(r))
    return float((1.0 - w_max) * np.mean(r) + w_max * np.max(r))


def run_incremental_model(
    data: dict,
    model_type: str,
    lambda_val: float,
    forgetting_factor: float,
    sigma: float = 0.5,
    D: int = 1000,
    seed: int = 42,
    verbose: bool = False,
    update_mode: str = "rep",
    metric: str = "nRMSE",
    scaler_type: str = "standard",
    test_only_reps: set[int] | None = None,
    validation_reps: set[int] | None = None,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    X_rep0 = data["X_rep0"]
    Y_rep0 = data["Y_rep0"]
    X_reps = data["X_reps"]
    Y_reps = data["Y_reps"]
    inc_reps = data["inc_reps"]
    scaler = data["scaler"]
    input_dim = data["input_dim"]
    output_dim = data["output_dim"]
    test_only_reps = test_only_reps or set()
    validation_reps = validation_reps or set()
    non_training_reps = test_only_reps.union(validation_reps)

    if scaler_type == "rolling":
        standardizer = RollingStandardizer(d=input_dim, alpha=0.001)
        standardizer.fit_batch(X_rep0)
        X_batch_scaled = standardizer.transform(X_rep0)
    else:
        # Use pre-scaled data for faster tuning
        X_batch_scaled = data.get("X_rep0_scaled")
        if X_batch_scaled is None:
            X_batch_scaled = scaler.transform(X_rep0)

    # Calculate target range for nRMSE normalization
    Y_concat_all = np.concatenate([Y_rep0] + Y_reps, axis=0)
    y_range = Y_concat_all.max() - Y_concat_all.min()
    norm_factor = y_range if metric == "nRMSE" and y_range > 0 else 1.0

    if model_type == "iRFF_RR":
        model = iRFF_RR(
            d=input_dim,
            M=output_dim,
            lambda_val=lambda_val,
            sigma=sigma,
            D=int(D),
            forgetting_factor=forgetting_factor,
        )
    elif model_type == "iRR":
        model = iRR(
            d=input_dim,
            M=output_dim,
            lambda_val=lambda_val,
            forgetting_factor=forgetting_factor,
        )
    elif model_type == "ExactKRR":
        model = ExactKRR(
            d=input_dim,
            M=output_dim,
            lambda_val=lambda_val,
            sigma=sigma,
            forgetting_factor=forgetting_factor,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if verbose:
        print("--- Phase 1: Warm Starting on Batch Data ---")
    t0_phase1 = time.perf_counter()
    model.start_batch(X_batch_scaled, Y_rep0)
    t1_phase1 = time.perf_counter()
    phase1_time = t1_phase1 - t0_phase1

    if verbose:
        print("--- Phase 2: Switching to Incremental Mode ---")
    t0_phase2 = time.perf_counter()

    confidences = []
    predictions = []
    errors = []
    rep_metric_history = {}

    X_reps_scaled = data.get("X_reps_scaled")
    if X_reps_scaled is None:
        X_reps_scaled = [None] * len(X_reps)

    for idx, (rep_label, X_rep, Y_rep) in enumerate(zip(inc_reps, X_reps, Y_reps)):
        if verbose:
            print(f"--- Incremental repetition {rep_label} ---")

        if X_rep.shape[0] == 0:
            continue

        if scaler_type == "rolling":
            X_rep_proc_scaled = standardizer.update_and_transform_batch(X_rep)
        else:
            X_rep_proc_scaled = (
                X_reps_scaled[idx] if idx < len(X_reps_scaled) and X_reps_scaled[idx] is not None else None
            )
            if X_rep_proc_scaled is None or X_rep_proc_scaled.shape[0] == 0:
                X_rep_proc_scaled = scaler.transform(X_rep)

        sum_mse = 0.0
        count = 0
        rep_errors = []

        if update_mode == "rep":
            # Predict the entire repetition at once using the current model state
            y_pred_batch = model.predict(X_rep_proc_scaled)

            # Confidence for the entire batch
            batch_conf = model.confidence_batch(X_rep_proc_scaled)

            # Compute step metrics efficiently
            residuals = Y_rep - y_pred_batch
            step_mses = np.mean(residuals**2, axis=1)

            cumulative_mses = np.cumsum(step_mses)
            counts = np.arange(1, len(step_mses) + 1)
            rep_metrics_history = np.sqrt(cumulative_mses / counts) / norm_factor

            confidences.extend(batch_conf)
            predictions.extend(y_pred_batch)
            errors.extend(rep_metrics_history.tolist())
            rep_errors = rep_metrics_history.tolist()

            # Update the model with the entire repetition batch at once
            if rep_label not in non_training_reps:
                model.update_batch(X_rep_proc_scaled, Y_rep)
        else:
            for i in range(len(X_rep_proc_scaled)):
                x_sample = X_rep_proc_scaled[i : i + 1]
                y_sample = Y_rep[i : i + 1]

                y_pred = model.predict(x_sample).flatten()
                y_true = y_sample.flatten()

                conf = model.confidence(x_sample)

                residuals = y_true - y_pred
                step_mse = np.mean(residuals**2)
                sum_mse += step_mse
                count += 1
                current_rmse = np.sqrt(sum_mse / count) 
                current_metric = current_rmse / norm_factor

                confidences.append(conf)
                predictions.append(y_pred)
                errors.append(current_metric)
                rep_errors.append(current_metric)

                if rep_label not in non_training_reps:
                    model.update_model(x_sample, y_sample)
        rep_metric_history[rep_label] = rep_errors

    t1_phase2 = time.perf_counter()
    phase2_time = t1_phase2 - t0_phase2

    Y_concat = np.concatenate(Y_reps, axis=0)
    predictions_arr = np.vstack(predictions)
    stream_ranges = []
    current_start = 0

    for X_rep in X_reps:
        n = X_rep.shape[0]
        stream_ranges.append((current_start, current_start + n))
        current_start += n

    rep_metrics = []
    rep_r2s = []
    rep_pearsons = []
    is_test_rep_mask: list[bool] = []
    is_validation_rep_mask: list[bool] = []
    for i, (start, end) in enumerate(stream_ranges):
        y_true_rep = Y_concat[start:end]
        y_pred_rep = predictions_arr[start:end]
        if y_true_rep.shape[0] == 0:
            continue
        mse = np.mean((y_true_rep - y_pred_rep) ** 2)
        rep_metrics.append(np.sqrt(mse) / norm_factor)
        
        # Calculate R-squared 
        var_true = np.mean(np.var(y_true_rep, axis=0))
        r2 = 1.0 - (mse / var_true) if var_true > 1e-8 else 0.0
        rep_r2s.append(r2)

        # Calculate Pearson Correlation Coefficient
        correlations = []
        for dof in range(y_true_rep.shape[1]):
            mean_true = np.mean(y_true_rep[:, dof])
            mean_pred = np.mean(y_pred_rep[:, dof])
            std_true = np.std(y_true_rep[:, dof])
            std_pred = np.std(y_pred_rep[:, dof])
            
            if std_true > 1e-8 and std_pred > 1e-8:
                cov = np.mean((y_true_rep[:, dof] - mean_true) * (y_pred_rep[:, dof] - mean_pred))
                corr = cov / (std_true * std_pred)
            else:
                corr = 0.0
            correlations.append(corr)
        avg_corr = np.mean(correlations)
        rep_pearsons.append(avg_corr)

        is_validation_rep_mask.append(inc_reps[i] in validation_reps)
        is_test_rep_mask.append(inc_reps[i] in test_only_reps)

    avg_r2 = float(np.mean(rep_r2s)) if rep_r2s else 0.0
    std_r2 = float(np.std(rep_r2s)) if rep_r2s else 0.0
    avg_pearson = float(np.mean(rep_pearsons)) if rep_pearsons else 0.0
    std_pearson = float(np.std(rep_pearsons)) if rep_pearsons else 0.0

    return {
        "rep_metrics": rep_metrics,  # Total metric per repetition (after all incremental reps)
        "rep_r2s": rep_r2s,  # R-squared 
        "rep_pearsons": rep_pearsons,
        "avg_r2": avg_r2,
        "std_r2": std_r2,
        "avg_pearson": avg_pearson,
        "std_pearson": std_pearson,
        "is_validation_rep_mask": is_validation_rep_mask,
        "is_test_rep_mask": is_test_rep_mask,
        "rep_metric_history": rep_metric_history, 
        "errors": errors,  # metric per sample (during incremental reps)
        "confidences": confidences,
        "predictions_arr": predictions_arr,
        "stream_ranges": stream_ranges,  # Start and end indices of each repetition
        "phase1_time": phase1_time,
        "phase2_time": phase2_time,
    }

def grid_search_tuning(
    data: dict,
    grid: dict,
    seeds: list[int],
    objective_mode: str = "mean_max",
    w_max: float = 0.5,
    tune_on_first_n: int | None = None,
    verbose: bool = True,
    update_mode: str = "rep",
    model_type: str = "iRFF_RR",
    metric: str = "nRMSE",
    tune_metric: str = "error",
    scaler_type: str = "standard",
    test_only_reps: set[int] | None = None,
    validation_reps: set[int] | None = None,
) -> list[dict]:
    if model_type == "iRR":
        grid = {k: v for k, v in grid.items() if k not in ["sigma", "D"]}
    elif model_type == "ExactKRR":
        grid = {k: v for k, v in grid.items() if k not in ["D"]}
        if "lambda_val" in grid:
            grid["lambda_val"] = [1e-5, 1e-4, 1e-3, 1e-2]  # Scale down to match D/2 kernel equivalence

    keys = list(grid.keys())
    combinations = list(itertools.product(*(grid[k] for k in keys))) # creates all possible combinations of hyperparameters
    results: list[dict] = []

    for values in combinations:
        hparams = dict(zip(keys, values))
        obj_vals: list[float] = []
        rep_metric_runs: list[list[float]] = []
        rep_r2_runs: list[list[float]] = []
        rep_pearson_runs: list[list[float]] = []

        for seed in seeds:
            out = run_incremental_model(data, model_type=model_type, seed=seed, verbose=False, update_mode=update_mode, metric=metric, scaler_type=scaler_type, test_only_reps=test_only_reps, validation_reps=validation_reps, **hparams)
            
            all_vals = np.array(out["rep_r2s"] if tune_metric == "R2" else out["rep_metrics"])

            # If validation_reps are provided, use them for the objective.
            # Otherwise, fall back to the old logic (use test_reps or all reps).
            if validation_reps and len(validation_reps) > 0:
                is_validation_mask = np.array(out["is_validation_rep_mask"])
                if len(all_vals) == len(is_validation_mask) and np.any(is_validation_mask):
                    vals_for_objective = all_vals[is_validation_mask]
                else:
                    vals_for_objective = np.array([]) # No validation reps found
            elif test_only_reps and len(test_only_reps) > 0:
                is_test_mask = np.array(out["is_test_rep_mask"])
                if len(all_vals) == len(is_test_mask) and np.any(is_test_mask):
                    vals_for_objective = all_vals[is_test_mask]
                else:
                    vals_for_objective = np.array([])
            else:
                vals_for_objective = all_vals
            
            if tune_on_first_n is not None:
                vals_for_objective = vals_for_objective[: int(tune_on_first_n)]
            
            obj_vals.append(composite_objective(vals_for_objective.tolist(), mode=objective_mode, w_max=w_max, higher_is_better=(tune_metric == "R2")))
            rep_metric_runs.append(out["rep_metrics"])
            rep_r2_runs.append(out["rep_r2s"])
            rep_pearson_runs.append(out["rep_pearsons"])

        mean_obj = float(np.mean(obj_vals))
        std_obj = float(np.std(obj_vals)) if len(obj_vals) > 1 else 0.0
        mean_rep_metrics = np.mean(np.asarray(rep_metric_runs), axis=0)
        mean_rep_r2s = np.mean(np.asarray(rep_r2_runs), axis=0)
        mean_rep_pearsons = np.mean(np.asarray(rep_pearson_runs), axis=0)

        results.append(
            {
                **hparams,
                "objective_mean": mean_obj,
                "objective_std": std_obj,
                "objective_per_seed": obj_vals,
                "rep_metrics_mean": mean_rep_metrics.tolist(),
                "rep_r2s_mean": mean_rep_r2s.tolist(),
                "rep_pearsons_mean": mean_rep_pearsons.tolist(),
            }
        )

    results.sort(key=lambda row: row["objective_mean"], reverse=(tune_metric == "R2"))
    if verbose:
        print(
            f"Grid search: {len(combinations)} combos x {len(seeds)} seeds "
            f"(objective={objective_mode}, w_max={w_max})\n"
        )
        for i, row in enumerate(results[:15]):
            if tune_metric == "R2":
                rep_str = ", ".join(f"{x:.3f}" for x in row["rep_r2s_mean"])
                metric_label = "R2"
            else:
                rep_str = ", ".join(f"{x:.3f}" for x in row["rep_metrics_mean"])
                metric_label = metric
            pearson_str = ", ".join(f"{x:.3f}" for x in row["rep_pearsons_mean"])
            if model_type == "iRR":
                hparam_str = f"λ={row['lambda_val']!s} ff={row['forgetting_factor']}"
            elif model_type == "ExactKRR":
                hparam_str = f"λ={row['lambda_val']!s} ff={row['forgetting_factor']} σ={row['sigma']}"
            else:
                hparam_str = f"λ={row['lambda_val']!s} ff={row['forgetting_factor']} σ={row['sigma']}"
                if 'D' in row:
                    hparam_str += f" D={row['D']}"
            print(
                f"{i + 1:2d}. obj={row['objective_mean']:.4f} ± {row['objective_std']:.4f} | "
                f"{hparam_str} | "
                f"rep {metric_label}=[{rep_str}] | "
                f"rep Pearson=[{pearson_str}]"
            )
        if len(results) > 15:
            print(f"... ({len(results) - 15} more)")
    return results


def rerank_top_k_on_eval_seeds(
    data: dict,
    ranked: list[dict],
    eval_seeds: list[int],
    k: int,
    objective_mode: str,
    w_max: float,
    eval_on_first_n: int | None,
    verbose: bool = True,
    update_mode: str = "rep",
    model_type: str = "iRFF_RR",
    metric: str = "nRMSE",
    tune_metric: str = "error",
    scaler_type: str = "standard",
    validation_reps: set[int] | None = None,
    test_only_reps: set[int] | None = None,
) -> list[dict]:
    """Re-evaluate only top-k configs on eval seeds for robustness."""
    k = int(max(1, k))
    top = ranked[:k]
    rescored: list[dict] = []

    if model_type == "iRR":
        hparam_keys = ("lambda_val", "forgetting_factor")
    elif model_type == "ExactKRR":
        hparam_keys = ("lambda_val", "sigma", "forgetting_factor")
    else:
        hparam_keys = ("lambda_val", "sigma", "forgetting_factor")
    for row in top:
        hparams = {k: row[k] for k in hparam_keys}
        obj_vals: list[float] = []
        rep_metric_runs: list[list[float]] = []
        rep_r2_runs: list[list[float]] = []
        rep_pearson_runs: list[list[float]] = []
        for seed in eval_seeds:
            out = run_incremental_model(data, model_type=model_type, seed=seed, verbose=False, update_mode=update_mode, metric=metric, scaler_type=scaler_type, test_only_reps=test_only_reps, validation_reps=validation_reps, **hparams)
            
            all_vals = np.array(out["rep_r2s"] if tune_metric == "R2" else out["rep_metrics"])

            if validation_reps and len(validation_reps) > 0:
                is_validation_mask = np.array(out["is_validation_rep_mask"])
                if len(all_vals) == len(is_validation_mask) and np.any(is_validation_mask):
                    vals_for_objective = all_vals[is_validation_mask]
                else:
                    vals_for_objective = np.array([])
            elif test_only_reps and len(test_only_reps) > 0:
                is_test_mask = np.array(out["is_test_rep_mask"])
                if len(all_vals) == len(is_test_mask) and np.any(is_test_mask):
                    vals_for_objective = all_vals[is_test_mask]
                else:
                    vals_for_objective = np.array([])
            else:
                vals_for_objective = all_vals

            if eval_on_first_n is not None:
                vals_for_objective = vals_for_objective[: int(eval_on_first_n)]
            
            obj_vals.append(composite_objective(vals_for_objective.tolist(), mode=objective_mode, w_max=w_max, higher_is_better=(tune_metric == "R2")))
            rep_metric_runs.append(out["rep_metrics"])
            rep_r2_runs.append(out["rep_r2s"])
            rep_pearson_runs.append(out["rep_pearsons"])

        mean_obj = float(np.mean(obj_vals))
        std_obj = float(np.std(obj_vals)) if len(obj_vals) > 1 else 0.0
        mean_rep_metrics = np.mean(np.asarray(rep_metric_runs), axis=0)
        mean_rep_r2s = np.mean(np.asarray(rep_r2_runs), axis=0)
        mean_rep_pearsons = np.mean(np.asarray(rep_pearson_runs), axis=0)

        rescored.append(
            {
                **row,
                "objective_mean": mean_obj,
                "objective_std": std_obj,
                "objective_per_seed": obj_vals,
                "rep_metrics_mean": mean_rep_metrics.tolist(),
                "rep_r2s_mean": mean_rep_r2s.tolist(),
                "rep_pearsons_mean": mean_rep_pearsons.tolist(),
            }
        )

    rescored.sort(key=lambda r: r["objective_mean"], reverse=(tune_metric == "R2"))
    if verbose:
        print(
            f"\nRe-ranked top-{k} on eval seeds ({len(eval_seeds)} seeds): {eval_seeds}\n"
        )
        for i, row in enumerate(rescored[: min(15, len(rescored))]):
            if tune_metric == "R2":
                rep_str = ", ".join(f"{x:.3f}" for x in row["rep_r2s_mean"])
                metric_label = "R2"
            else:
                rep_str = ", ".join(f"{x:.3f}" for x in row["rep_metrics_mean"])
                metric_label = metric
            pearson_str = ", ".join(f"{x:.3f}" for x in row["rep_pearsons_mean"])
            if model_type == "iRR":
                hparam_str = f"λ={row['lambda_val']!s} ff={row['forgetting_factor']}"
            elif model_type == "ExactKRR":
                hparam_str = f"λ={row['lambda_val']!s} ff={row['forgetting_factor']} σ={row['sigma']}"
            else:
                hparam_str = f"λ={row['lambda_val']!s} ff={row['forgetting_factor']} σ={row['sigma']}"
                if 'D' in row:
                    hparam_str += f" D={row['D']}"
            print(
                f"{i + 1:2d}. obj={row['objective_mean']:.4f} ± {row['objective_std']:.4f} | "
                f"{hparam_str} | "
                f"rep {metric_label}=[{rep_str}] | "
                f"rep Pearson=[{pearson_str}]"
            )
    return rescored

def parse_seeds(s: str) -> list[int]:
    return [int(p.strip()) for p in s.split(",") if p.strip()]

#---------------------------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid search / plots for iRFF_RR and iRR models."
    )
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter grid search.")
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44",
        help="Comma-separated RNG seeds. With --tune, consider using a single seed and "
        "re-ranking only top-K with --eval-seeds.",
    )
    parser.add_argument(
        "--eval-seeds",
        type=str,
        default=None,
        help="With --tune, optionally re-evaluate only the top-K configs on these seeds (comma-separated).",
    )
    parser.add_argument(
        "--eval-top-k",
        type=int,
        default=10,
        help="With --tune and --eval-seeds, how many top configs to re-evaluate.",
    )
    parser.add_argument("--objective", choices=["mean", "max", "mean_max"], default="mean_max")
    parser.add_argument(
        "--w-max",
        type=float,
        default=0.5,
        help="Weight on max rep metric for mean_max objective.",
    )
    parser.add_argument(
        "--tune-on-first-n",
        type=int,
        default=None,
        help="Objective uses first N incremental reps only.",
    )
    parser.add_argument(
        "--eval-on-first-n",
        type=int,
        default=None,
        help="With --tune and --eval-seeds, re-ranking objective uses first N incremental reps only. "
        "Default is all repetitions.",
    )
    parser.add_argument(
        "--plot-best",
        action="store_true",
        help="With --tune, plot best config using the first seed.",
    )
    parser.add_argument(
        "--save-best",
        type=str,
        default="best_hparams.json",
        help="With --tune, save best hyperparameters to this JSON file.",
    )
    parser.add_argument(
        "--use-best",
        type=str,
        default=None,
        help="Skip tuning and run a single experiment using hyperparameters loaded from this JSON file.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Single run: skip plots.")
    parser.add_argument("--dataset", choices=["EMaGer", "NinaPro_s12", "NinaPro_s11"], default="NinaPro_s12", help="Select which dataset to run the pipeline on.")
    parser.add_argument("--plot-channel", type=int, default=None, help="Channel number to plot in the detailed view.")
    parser.add_argument("--plot-rep", type=str, default=None, help="Comma-separated target repetitions to plot in the detailed view (e.g., '1,2').")
    parser.add_argument("--update-mode", choices=["sample", "rep"], default="rep", help="Update the model sample-by-sample or rep-by-rep.")
    parser.add_argument("--clean", action="store_true", help="Run the experiment using only clean repetitions (no shocks applied).")
    parser.add_argument("--model", choices=["iRFF_RR", "iRR", "ExactKRR"], default="iRFF_RR", help="Choose between iRFF_RR, traditional iRR, and exact KRR baseline.")
    parser.add_argument("--metric", choices=["RMSE", "nRMSE"], default="nRMSE", help="Evaluation metric: RMSE or normalized RMSE (nRMSE).")
    parser.add_argument("--save-plots", type=str, default=None, help="Directory to save plot figures.")
    parser.add_argument("--plot-psd", action="store_true", help="Plot Power Spectral Density comparison for the target rep(s).")
    parser.add_argument("--plot-scatter", action="store_true", help="Plot the DoF scatter plots for each repetition.")
    parser.add_argument("--runs", type=int, default=1, help="Number of identical runs to average execution time metrics.")
    parser.add_argument("--tune-metric", choices=["error", "R2"], default="error", help="Metric to optimize during tuning: 'error' (minimizes RMSE/nRMSE) or 'R2' (maximizes R2 score).")
    parser.add_argument("--scaler", choices=["standard", "rolling"], default="standard", help="Scaling method to use.")
    parser.add_argument("--test-only-reps", type=str, default=None, help="Comma-separated rep numbers to use for testing only (no model updates).")
    parser.add_argument("--validation-reps", type=str, default=None, help="Comma-separated rep numbers to use for hyperparameter tuning objective. Model is not trained on these.")
    shock_types = ["impedance", "fatigue", "light channel mixing", "channel mixing", "sensor failure", "sensor_failure 1", "sensor_failure 2", "sensor_failure 3"]
    parser.add_argument("--shock-type-all-reps", type=str, default=None, choices=shock_types, help="Apply a single shock type to all incremental reps. Overrides default shock_mapping and --clean.")
    parser.add_argument(
        "--save-run-results",
        nargs='?',
        const="run_results",
        default=None,
        help="Save final run results to a timestamped JSON file. Optionally specify a directory."
    )
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    
    # Seed the generators before loading data so that simulated shocks (noise) are deterministic
    np.random.seed(seeds[0])
    random.seed(seeds[0])
    
    data = load_and_prepare_data(dataset_name=args.dataset, clean=args.clean, shock_type_all_reps=args.shock_type_all_reps)
    eval_seeds = parse_seeds(args.eval_seeds) if args.eval_seeds else None
    test_only_reps = set(parse_seeds(args.test_only_reps)) if args.test_only_reps else set()
    validation_reps = set(parse_seeds(args.validation_reps)) if args.validation_reps else set()
    
    if args.model == "iRR":
        hparam_keys = ("lambda_val", "forgetting_factor")
    elif args.model == "ExactKRR":
        hparam_keys = ("lambda_val", "sigma", "forgetting_factor")
    else:
        hparam_keys = ("lambda_val", "sigma", "forgetting_factor")

    if args.tune:
        grid = deepcopy(TUNE_GRID)
        ranked = grid_search_tuning(
            data,
            grid,
            seeds=seeds,
            objective_mode=args.objective,
            w_max=args.w_max,
            tune_on_first_n=args.tune_on_first_n,
            verbose=True,
            update_mode=args.update_mode,
            model_type=args.model,
            metric=args.metric,
            tune_metric=args.tune_metric,
            scaler_type=args.scaler,
            validation_reps=validation_reps,
            test_only_reps=test_only_reps,
        )

        if eval_seeds is not None:
            ranked = rerank_top_k_on_eval_seeds(
                data=data,
                ranked=ranked,
                eval_seeds=eval_seeds,
                k=args.eval_top_k,
                objective_mode=args.objective,
                w_max=args.w_max,
                eval_on_first_n=args.eval_on_first_n,
                verbose=True,
                update_mode=args.update_mode,
                model_type=args.model,
                metric=args.metric,
                tune_metric=args.tune_metric,
                scaler_type=args.scaler,
                validation_reps=validation_reps,
                test_only_reps=test_only_reps,
            )
        best = ranked[0]
        if args.tune_metric == "R2":
            print(f"\nBest (highest mean objective over seeds) for {args.model}:")
        else:
            print(f"\nBest (lowest mean objective over seeds) for {args.model}:")
        if args.model == "iRR":
            hparam_str = f"  lambda_val={best['lambda_val']}, forgetting_factor={best['forgetting_factor']}"
        elif args.model == "ExactKRR":
            hparam_str = f"  lambda_val={best['lambda_val']}, forgetting_factor={best['forgetting_factor']}, sigma={best['sigma']}"
        else:
            hparam_str = f"  lambda_val={best['lambda_val']}, forgetting_factor={best['forgetting_factor']}, sigma={best['sigma']}"
            if 'D' in best:
                hparam_str += f", D={best['D']}"
        print(hparam_str)
        print(f"  objective_mean={best['objective_mean']:.4f}, std={best['objective_std']:.4f}")
        if args.tune_metric == "R2":
            print(f"  rep_R2s (mean over seeds)={best['rep_r2s_mean']}")
        else:
            print(f"  rep_{args.metric}s (mean over seeds)={best['rep_metrics_mean']}")

        save_payload = {
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": args.dataset,
            "seeds": seeds,
            "eval_seeds": eval_seeds,
            "eval_top_k": args.eval_top_k if eval_seeds is not None else None,
            "eval_on_first_n": args.eval_on_first_n if eval_seeds is not None else None,
            "objective_mode": args.objective,
            "w_max": args.w_max,
            "metric": args.metric,
            "tune_metric": args.tune_metric,
            "scaler_type": args.scaler,
            "validation_reps": list(validation_reps) if validation_reps else None,
            "test_only_reps": list(test_only_reps) if test_only_reps else None,
            "tune_on_first_n": args.tune_on_first_n,
            "hparams": {k: best[k] for k in hparam_keys},
            "metrics": {
                "objective_mean": best["objective_mean"],
                "objective_std": best["objective_std"],
                "rep_metrics_mean": best["rep_metrics_mean"],
                "rep_r2s_mean": best["rep_r2s_mean"],
                "objective_per_seed": best["objective_per_seed"],
            },
            "ranked_results": ranked,
        }
        save_best_hparams_json(args.save_best, save_payload)
        print(f"Saved best hyperparameters to: {args.save_best}")

        if args.plot_best:
            h = {k: best[k] for k in hparam_keys}
            result = run_incremental_model(data, model_type=args.model, seed=seeds[0], verbose=True, update_mode=args.update_mode, metric=args.metric, scaler_type=args.scaler, validation_reps=validation_reps, test_only_reps=test_only_reps, **h)
            total_samples = sum(r.shape[0] for r in data["X_reps"])
            avg_step_ms = (result['phase2_time'] / total_samples) * 1000 if total_samples > 0 else 0.0
            print(f"Incremental execution time ({args.model}): {result['phase2_time']:.4f} seconds")
            print(f"Average time per prediction/update step: {avg_step_ms:.4f} ms")
            show_plot = not args.no_plots
            plot_experiment(data, result, update_mode=args.update_mode, metric_name=args.metric, model_type=args.model, save_dir=args.save_plots, show_plot=show_plot, test_only_reps=test_only_reps)
            rep_titles = [
                f"Rep {r} ({data['shock_mapping'][r]})" if r in data.get("shock_mapping", {}) else f"Rep {r}"
                for r in data["inc_reps"]
            ]
            if args.plot_scatter:
                plot_dof_scatter_by_repetition(data, result, rep_titles=rep_titles, save_dir=args.save_plots, show_plot=show_plot)
            if args.plot_rep is not None:
                plot_reps = [int(r.strip()) for r in args.plot_rep.split(",") if r.strip()]
                for rep in plot_reps:
                    plot_semg_and_predictions(
                        data,
                        result,
                        target_rep=rep,
                        channel=args.plot_channel,
                        clean_rep=data.get("warm_rep_index", 0),
                        model_type=args.model,
                        save_dir=args.save_plots,
                        show_plot=show_plot,
                    )
                    if args.plot_psd:
                        plot_psd_comparison(
                            data,
                            target_rep=rep,
                            channel=args.plot_channel if args.plot_channel is not None else 0,
                            clean_rep=data.get("warm_rep_index", 0),
                            save_dir=args.save_plots,
                            show_plot=show_plot,
                        )
        return

    if args.use_best is not None:
        h, payload = load_best_hparams_json(args.use_best)
        print(f"Loaded hyperparameters from: {args.use_best}")
        saved_std = payload.get("metrics", {}).get("objective_std", 0.0)
    else:
        h = {k: DEFAULT_HPARAMS[k] for k in hparam_keys}
        saved_std = 0.0

    phase1_times = []
    phase2_times = []
    
    print(f"\nRunning experiment {args.runs} time(s) to compute stable execution metrics...")
    for i in range(args.runs):
        print(f"\n=== Starting Run {i + 1}/{args.runs} ===")
        result = run_incremental_model(data, model_type=args.model, seed=seeds[0], verbose=True, update_mode=args.update_mode, metric=args.metric, scaler_type=args.scaler, validation_reps=validation_reps, test_only_reps=test_only_reps, **h)
        phase1_times.append(result["phase1_time"])
        phase2_times.append(result["phase2_time"])

    avg_phase1 = float(np.mean(phase1_times))
    avg_phase2 = float(np.mean(phase2_times))
    std_phase2 = float(np.std(phase2_times)) if args.runs > 1 else 0.0
    
    total_samples = sum(r.shape[0] for r in data["X_reps"])
    if total_samples > 0:
        step_times_ms = [(t / total_samples) * 1000 for t in phase2_times]
        avg_step_ms = float(np.mean(step_times_ms))
        std_step_ms = float(np.std(step_times_ms)) if args.runs > 1 else 0.0
    else:
        avg_step_ms = 0.0
        std_step_ms = 0.0

    rep_titles = [
        f"Rep {r} ({data['shock_mapping'][r]})" if r in data.get("shock_mapping", {}) else f"Rep {r}"
        for r in data["inc_reps"]
    ]
    
    all_metrics = np.array(result["rep_metrics"])
    all_r2s = np.array(result.get("rep_r2s", []))
    all_pearsons = np.array(result.get("rep_pearsons", []))
    is_test_mask = np.array(result["is_test_rep_mask"])
    
    obj_val = None # Will be populated for saving
    if test_only_reps:
        print(f"\n--- Training Reps (Prequential) ---")
        if np.any(~is_test_mask):
            print(f"rep_{args.metric}s:", all_metrics[~is_test_mask].tolist())
            print(f"rep_R2s:", [round(x, 3) for x in all_r2s[~is_test_mask]])
            print(f"rep_Pearsons:", [round(x, 3) for x in all_pearsons[~is_test_mask]])
        else:
            print("No training reps in the incremental phase.")
        
        print(f"\n--- Test-Only Reps ---")
        if np.any(is_test_mask):
            print(f"rep_{args.metric}s:", all_metrics[is_test_mask].tolist())
            print(f"rep_R2s:", [round(x, 3) for x in all_r2s[is_test_mask]])
            print(f"rep_Pearsons:", [round(x, 3) for x in all_pearsons[is_test_mask]])
            
            obj_vals = all_r2s[is_test_mask] if args.tune_metric == "R2" else all_metrics[is_test_mask]
            obj_val = composite_objective(obj_vals, mode="mean_max", w_max=0.5, higher_is_better=(args.tune_metric == "R2"))
            print(f"objective on test reps (mean_max, w_max=0.5): {obj_val:.4f} ± {saved_std:.4f}")
        else:
            print("No test-only reps found.")

    else:
        print(f"rep_{args.metric}s:", result["rep_metrics"])
        print(f"rep_R2s:", [round(x, 3) for x in result.get("rep_r2s", [])])
        print(f"rep_Pearsons:", [round(x, 3) for x in result.get("rep_pearsons", [])])
        obj_vals = result["rep_r2s"] if args.tune_metric == "R2" else result["rep_metrics"]
        obj_val = composite_objective(obj_vals, mode="mean_max", w_max=0.5, higher_is_better=(args.tune_metric == "R2"))
        print(f"objective (mean_max, w_max=0.5): {obj_val:.4f} ± {saved_std:.4f}")

    print(f"average R2: {result.get('avg_r2', 0.0):.3f} ± {result.get('std_r2', 0.0):.3f}")
    print(f"average Pearson: {result.get('avg_pearson', 0.0):.3f} ± {result.get('std_pearson', 0.0):.3f}")

    # --- Save run results if requested ---
    if args.save_run_results:
        results_to_save = {}
        if test_only_reps:
            train_mask = ~is_test_mask
            if np.any(train_mask):
                results_to_save["training_reps_metrics"] = {
                    f"rep_{args.metric}s": all_metrics[train_mask].tolist(),
                    "rep_r2s": [round(x, 3) for x in all_r2s[train_mask]],
                    "rep_pearsons": [round(x, 3) for x in all_pearsons[train_mask]],
                }
            if np.any(is_test_mask):
                results_to_save["test_only_reps_metrics"] = {
                    f"rep_{args.metric}s": all_metrics[is_test_mask].tolist(),
                    "rep_r2s": [round(x, 3) for x in all_r2s[is_test_mask]],
                    "rep_pearsons": [round(x, 3) for x in all_pearsons[is_test_mask]],
                    "objective_on_test_reps": obj_val,
                    "objective_std": saved_std,
                }
        else:
            results_to_save["all_reps_metrics"] = {
                f"rep_{args.metric}s": result["rep_metrics"],
                "rep_r2s": [round(x, 3) for x in result.get("rep_r2s", [])],
                "rep_pearsons": [round(x, 3) for x in result.get("rep_pearsons", [])],
                "objective": obj_val,
                "objective_std": saved_std,
            }

        results_to_save["average_r2_overall"] = result.get('avg_r2', 0.0)
        results_to_save["std_r2_overall"] = result.get('std_r2', 0.0)
        results_to_save["average_pearson_overall"] = result.get('avg_pearson', 0.0)
        results_to_save["std_pearson_overall"] = result.get('std_pearson', 0.0)

        execution_times_to_save = {
            "phase1_batch_training_time_s": avg_phase1,
            "phase2_incremental_time_s_mean": avg_phase2,
            "phase2_incremental_time_s_std": std_phase2,
            "avg_time_per_step_ms_mean": avg_step_ms,
            "avg_time_per_step_ms_std": std_step_ms,
        }

        # Sanitize args for JSON serialization
        args_dict = {k: str(v) if isinstance(v, (set, Path)) else v for k, v in vars(args).items()}

        run_payload = {
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "args": args_dict,
            "hyperparameters_used": h,
            "results": results_to_save,
            "execution_times": execution_times_to_save,
        }

        os.makedirs(args.save_run_results, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"run_results_{args.dataset}_{args.model}_{timestamp}.json"
        filepath = os.path.join(args.save_run_results, filename)
        save_best_hparams_json(filepath, run_payload)
        print(f"Saved full run results to: {filepath}")

    print(f"\n--- Execution Time Metrics ({args.model}, averaged over {args.runs} runs) ---")
    print(f"Phase 1 (Batch Training) Time: {avg_phase1:.4f} seconds")
    print(f"Phase 2 (Incremental) Time: {avg_phase2:.4f} ± {std_phase2:.4f} seconds")
    print(f"Average time per prediction/update step: {avg_step_ms:.4f} ± {std_step_ms:.4f} ms")
    print("----------------------------------------------------------------\n")
    
    if not args.no_plots or args.save_plots:
        show_plot = not args.no_plots
        plot_experiment(data, result, update_mode=args.update_mode, metric_name=args.metric, model_type=args.model, save_dir=args.save_plots, show_plot=show_plot, test_only_reps=test_only_reps) # validation reps don't need special plotting
        if args.plot_scatter:
            plot_dof_scatter_by_repetition(data, result, rep_titles=rep_titles, save_dir=args.save_plots, show_plot=show_plot)
        if args.plot_rep is not None:
            plot_reps = [int(r.strip()) for r in args.plot_rep.split(",") if r.strip()]
            for rep in plot_reps:
                plot_semg_and_predictions(
                    data,
                    result,
                    target_rep=rep,
                    channel=args.plot_channel,
                    clean_rep=data.get("warm_rep_index", 0),
                    model_type=args.model,
                    save_dir=args.save_plots,
                    show_plot=show_plot,
                )
                if args.plot_psd:
                    plot_psd_comparison(
                        data,
                        target_rep=rep,
                        channel=args.plot_channel if args.plot_channel is not None else 0,
                        clean_rep=data.get("warm_rep_index", 0),
                        save_dir=args.save_plots,
                        show_plot=show_plot,
                    )


if __name__ == "__main__":
    main()
