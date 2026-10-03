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


def plot_model_comparison(results_dir: str, dataset: str, condition: str, plot_metric: str, output_file: str | None = None):
    """
    Generates a plot comparing model performance for a specific shock condition.

    Args:
        results_dir: The directory containing the JSON run result files.
        dataset: The name of the dataset to generate the plot for.
        condition: The specific condition to plot (e.g., 'Clean', 'impedance').
        plot_metric: The metric to display on the Y-axis ('nRMSE' or 'R2').
        output_file: Optional path to save the generated plot image.
    """
    models_to_plot = ["iRFF_RR", "iRR", "ExactKRR"]
    # {model_name: [(num_train_reps, final_rep_metric_value), ...]}
    results_by_model = defaultdict(list)

    try:
        total_inc_reps = get_total_inc_reps(dataset)
    except ValueError as e:
        print(e)
        return

    print(f"Processing files for dataset '{dataset}' and condition '{condition}'...")

    for model in models_to_plot:
        json_files = glob.glob(os.path.join(results_dir, f"run_results_{dataset}_{model}_*.json"))

        for file_path in json_files:
            with open(file_path, 'r') as f:
                payload = json.load(f)

            args = payload.get("args", {})

            # --- 1. Check if the file matches the desired condition ---
            file_condition = None
            if args.get("clean"):
                file_condition = "Clean"
            elif args.get("shock_type_all_reps"):
                file_condition = args["shock_type_all_reps"]

            if not file_condition or file_condition.lower() != condition.lower():
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
            results_by_model[model].append((num_training_reps, final_rep_metric_value))

    if not results_by_model:
        print(f"No valid data points found for condition '{condition}'. Ensure you have run scenarios with '--save-run-results'.")
        return

    # --- 4. Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    style_map = {"iRFF_RR": {"color": "blue", "marker": "o"}, "iRR": {"color": "green", "marker": "s"}, "ExactKRR": {"color": "red", "marker": "^"}}

    for model, data_points in sorted(results_by_model.items()):
        if not data_points: continue
        sorted_points = sorted(data_points, key=lambda item: item[0])
        x_vals, y_vals = zip(*sorted_points)
        style = style_map.get(model, {"color": "gray", "marker": "."})
        ax.plot(x_vals, y_vals, label=model.replace("_", "-"), color=style["color"], marker=style["marker"], linestyle='-', markersize=8)

    ax.set_xlabel("Number of Repetitions in Training Set", fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_title(f"Model Comparison for '{condition.title()}' Condition ({dataset})", fontsize=16)
    ax.legend(title="Model", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)

    x_ticks = np.unique([p[0] for points in results_by_model.values() for p in points])
    if len(x_ticks) > 0:
        ax.set_xticks(np.arange(min(x_ticks), max(x_ticks) + 1, 1))

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot model performance comparison for a specific condition.")
    parser.add_argument("--results-dir", type=str, default="run_results", help="Directory with JSON run results.")
    parser.add_argument("--dataset", type=str, required=True, choices=["EMaGer", "NinaPro_s12", "NinaPro_s11"], help="The dataset to plot for.")
    parser.add_argument("--condition", type=str, required=True, help="The condition to plot (e.g., 'Clean', 'impedance', 'channel mixing'). Case-insensitive.")
    parser.add_argument("--plot-metric", type=str, default="nRMSE", choices=["nRMSE", "R2", "Pearson"], help="The metric to plot on the Y-axis.")
    parser.add_argument("--output-file", type=str, default=None, help="Optional path to save the plot image.")
    args = parser.parse_args()
    plot_model_comparison(args.results_dir, args.dataset, args.condition, args.plot_metric, args.output_file)