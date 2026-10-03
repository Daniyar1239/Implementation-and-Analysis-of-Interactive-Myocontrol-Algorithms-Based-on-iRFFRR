import argparse
import glob
import json
import os

import matplotlib.pyplot as plt


def plot_execution_time(results_dir: str, dataset: str, condition: str, output_file: str | None = None):
    """
    Generates a bar plot comparing the execution time of different models for a specific condition.

    Args:
        results_dir: The directory containing the JSON run result files.
        dataset: The name of the dataset to plot for.
        condition: The specific condition to plot (e.g., 'Clean', 'Impedance').
        output_file: Optional path to save the generated plot image.
    """
    models_to_plot = ["iRFF_RR", "iRR", "ExactKRR"]
    model_times = {}
    model_stds = {}
    num_runs = None

    print(f"Processing files for dataset '{dataset}' and condition '{condition}'...")

    for model in models_to_plot:
        json_files = glob.glob(os.path.join(results_dir, f"run_results_{dataset}_{model}_*.json"))

        candidate_files = []
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

            # --- 2. Check if the file has timing data and a valid scenario ---
            test_reps_str = args.get("test_only_reps")
            if not test_reps_str:
                continue

            execution_times = payload.get("execution_times", {})
            time_mean = execution_times.get("avg_time_per_step_ms_mean")

            if time_mean is not None:
                candidate_files.append({
                    "path": file_path,
                    "num_test_reps": len(test_reps_str.split(',')),
                    "payload": payload
                })

        if not candidate_files:
            print(f"Warning: No result file with execution times found for model '{model}' under condition '{condition}'.")
            continue

        # --- 3. Select the best candidate (most training data = fewest test reps) ---
        best_candidate = min(candidate_files, key=lambda x: x['num_test_reps'])
        
        print(f"  > For model '{model}', using timing from scenario with {best_candidate['num_test_reps']} test rep(s): {os.path.basename(best_candidate['path'])}")

        # --- 4. Extract data from the best candidate ---
        payload = best_candidate['payload']
        execution_times = payload.get("execution_times", {})
        model_times[model] = execution_times.get("avg_time_per_step_ms_mean")
        model_stds[model] = execution_times.get("avg_time_per_step_ms_std", 0)
        if num_runs is None:
            num_runs = payload.get("args", {}).get("runs", "N/A")

    if not model_times:
        print(f"No execution time data found for any model under condition '{condition}'. Cannot generate plot.")
        return

    # --- Plotting ---
    style_map = {"iRFF_RR": {"color": "blue"}, "iRR": {"color": "green"}, "ExactKRR": {"color": "red"}}
    plot_labels, plot_means, plot_stds, plot_colors = [], [], [], []

    for model_name in style_map:
        if model_name in model_times:
            plot_labels.append(model_name.replace("_", "-"))
            plot_means.append(model_times[model_name])
            plot_stds.append(model_stds.get(model_name, 0))
            plot_colors.append(style_map[model_name]["color"])

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))

    bars = ax.bar(plot_labels, plot_means, yerr=plot_stds, capsize=5, color=plot_colors)
    ax.bar_label(bars, fmt='%.3f', padding=3)

    ax.set_ylabel("Average Time per Step (ms) (log scale)", fontsize=14)
    ax.set_xlabel("Model", fontsize=14)
    title = f"Execution Time Comparison for '{condition.title()}' Condition ({dataset})\n"
    if num_runs:
        title += f"(Time averaged over {num_runs} runs)"
    ax.set_title(title, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_yscale('log')

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot model execution time comparison for a specific condition.")
    parser.add_argument("--results-dir", type=str, default="run_results", help="Directory with JSON run results.")
    parser.add_argument("--dataset", type=str, required=True, choices=["EMaGer", "NinaPro_s12", "NinaPro_s11"], help="The dataset to plot for.")
    parser.add_argument("--condition", type=str, required=True, help="The condition to plot (e.g., 'Clean', 'impedance'). Case-insensitive.")
    parser.add_argument("--output-file", type=str, default=None, help="Optional path to save the plot image.")
    args = parser.parse_args()
    plot_execution_time(args.results_dir, args.dataset, args.condition, args.output_file)