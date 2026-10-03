import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def get_total_inc_reps(dataset_name: str) -> int:
    """Returns the total number of incremental repetitions for a given dataset."""
    if dataset_name == "EMaGer":
        # inc_reps = [1, 2, 3, 4]
        return 4
    elif dataset_name in ["NinaPro_s12", "NinaPro_s11"]:
        # inc_reps = [1..9] for s12, [2..10] for s11
        return 9
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def plot_training_scenarios(results_dir: str, dataset: str, model: str, plot_metric: str, output_file: str | None = None, compare_rolling_dir: str | None = None, condition_filter: str | None = None):
    """
    Generates a plot comparing model performance across different training scenarios and shock conditions.
    If compare_rolling_dir is provided for the iRFF_RR model, it will also compare standard vs. rolling scaling.

    Args:
        results_dir: The directory containing the JSON run result files (standard scaling).
        dataset: The name of the dataset to generate the plot for (e.g., 'EMaGer').
        model: The name of the model to generate the plot for (e.g., 'iRFF_RR').
        plot_metric: The metric to display on the Y-axis ('nRMSE' or 'R2').
        output_file: Optional path to save the generated plot image.
        compare_rolling_dir: Optional. Path to results from rolling scaler to compare against. Only used for iRFF_RR model.
        condition_filter: Optional. If provided, plots only data for this specific condition (e.g., 'Clean', 'Impedance').
    """
    # {condition: {scaling_type: [(num_train_reps, metric), ...], ...}}
    results_by_condition_and_scaling = defaultdict(lambda: defaultdict(list))

    dirs_to_process = {'Standard': results_dir}
    if compare_rolling_dir and model == 'iRFF_RR':
        dirs_to_process['Rolling'] = compare_rolling_dir

    try:
        total_inc_reps = get_total_inc_reps(dataset)
    except ValueError as e:
        print(e)
        return

    print(f"Processing for dataset '{dataset}' and model '{model}'...")

    for scaling_type, current_dir in dirs_to_process.items():
        json_files = glob.glob(os.path.join(current_dir, f"run_results_{dataset}_{model}_*.json"))
        if not json_files:
            print(f"Warning: No result files found for '{scaling_type}' scaling for model '{model}' in '{current_dir}'.")
            continue

        print(f"Found {len(json_files)} files for '{scaling_type}' scaling. Processing...")

        for file_path in json_files:
            with open(file_path, 'r') as f:
                payload = json.load(f)

            args = payload.get("args", {})

            # --- 1. Determine the condition (Clean, Impedance, etc.) ---
            condition = None
            if args.get("clean"):
                condition = "Clean"
            elif args.get("shock_type_all_reps"):
                condition = args["shock_type_all_reps"].replace("_", " ").title()

            if not condition:
                continue

            # If a condition filter is specified, skip files that don't match
            if condition_filter and condition.lower() != condition_filter.lower():
                continue

            # --- 2. Determine the number of training repetitions (X-axis) ---
            test_reps_str = args.get("test_only_reps")
            if not test_reps_str:
                continue

            num_test_reps = len(test_reps_str.split(','))
            num_training_reps = 1 + (total_inc_reps - num_test_reps)

            # --- 3. Get the final repetition's metric value (Y-axis) ---
            results = payload.get("results", {})
            test_metrics = results.get("test_only_reps_metrics", {})

            if plot_metric.lower() == 'nrmse':
                metric_key = f'rep_{args.get("metric", "nRMSE")}s'
                y_label = "nRMSE on Final Repetition"
            elif plot_metric.lower() == 'r2':
                metric_key = 'rep_r2s'
                y_label = "R² Score on Final Repetition"
            elif plot_metric.lower() == 'pearson':
                metric_key = 'rep_pearsons'
                y_label = "Pearson Correlation on Final Repetition"
            else:
                raise ValueError(f"Unsupported plot_metric: {plot_metric}")

            if metric_key not in test_metrics or not test_metrics[metric_key]:
                continue

            final_rep_metric_value = test_metrics[metric_key][-1]
            results_by_condition_and_scaling[condition][scaling_type].append((num_training_reps, final_rep_metric_value))

    if not results_by_condition_and_scaling:
        print("No valid data points found to plot. Ensure you have run scenarios with '--save-run-results'.")
        return

    # --- 4. Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    style_map = {
        "Clean": {"color": "green", "marker": "o"}, "Impedance": {"color": "blue", "marker": "s"},
        "Fatigue": {"color": "red", "marker": "^"}, "Channel Mixing": {"color": "purple", "marker": "D"},
        "Light Channel Mixing": {"color": "orange", "marker": "P"}, "Sensor Failure": {"color": "black", "marker": "X"},
        "Sensor Failure 1": {"color": "black", "marker": "X"}, "Sensor Failure 2": {"color": "dimgray", "marker": "x"},
        "Sensor Failure 3": {"color": "darkgray", "marker": "*"},
    }

    scaling_style_map = {
        "Standard": {"linestyle": '-', "markersize": 8},
        "Rolling": {"linestyle": '--', "markersize": 6},
    }

    for condition, scaling_data in sorted(results_by_condition_and_scaling.items()):
        base_style = style_map.get(condition, {"color": "gray", "marker": "."})

        for scaling_type, data_points in sorted(scaling_data.items()):
            if not data_points: continue

            sorted_points = sorted(data_points, key=lambda item: item[0])
            x_vals, y_vals = zip(*sorted_points)

            plot_style = scaling_style_map.get(scaling_type, {"linestyle": ':', "markersize": 8})

            label = f"{condition}"
            if len(dirs_to_process) > 1:
                label += f" ({scaling_type})"

            ax.plot(x_vals, y_vals,
                    label=label,
                    color=base_style["color"],
                    marker=base_style["marker"],
                    linestyle=plot_style["linestyle"],
                    markersize=plot_style["markersize"])

    ax.set_xlabel("Number of Repetitions in Training Set", fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    title = f"Model Performance vs. Training Set Size ({dataset} - {model})"
    if condition_filter:
        title = f"Model Performance for '{condition_filter.title()}' Condition ({dataset} - {model})"
    if len(dirs_to_process) > 1:
        title += "\n(Standard vs. Rolling Scaling)"
    ax.set_title(title, fontsize=16)
    ax.legend(title="Condition", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)

    x_ticks = np.unique([p[0] for points_dict in results_by_condition_and_scaling.values() for points in points_dict.values() for p in points])
    if len(x_ticks) > 0:
        ax.set_xticks(np.arange(min(x_ticks), max(x_ticks) + 1, 1))

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot model performance across different training scenarios.")
    parser.add_argument("--results-dir", type=str, default="run_results", help="Directory with JSON run results.")
    parser.add_argument("--dataset", type=str, required=True, choices=["EMaGer", "NinaPro_s12", "NinaPro_s11"], help="The dataset to plot for.")
    parser.add_argument("--model", type=str, required=True, choices=["iRFF_RR", "iRR", "ExactKRR"], help="The model to plot for.")
    parser.add_argument("--plot-metric", type=str, default="nRMSE", choices=["nRMSE", "R2", "Pearson"], help="The metric to plot on the Y-axis.")
    parser.add_argument("--output-file", type=str, default=None, help="Optional path to save the plot image.")
    parser.add_argument("--compare-rolling-dir", type=str, default=None, help="For iRFF_RR, path to results from rolling scaler to compare against standard.")
    parser.add_argument("--condition", type=str, default=None, help="Optional. Plot only for a specific condition (e.g., 'Clean', 'Impedance').")
    args = parser.parse_args()
    plot_training_scenarios(args.results_dir, args.dataset, args.model, args.plot_metric, args.output_file, args.compare_rolling_dir, args.condition)