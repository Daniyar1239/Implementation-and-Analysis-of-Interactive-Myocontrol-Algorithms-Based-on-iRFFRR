import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

FIGURE_COUNTER = 1

def handle_plot_output(save_dir: str | None = None, show_plot: bool = True) -> None:
    global FIGURE_COUNTER
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f"Figure_{FIGURE_COUNTER}.png")
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        print(f"Saved figure to {filepath}")
        FIGURE_COUNTER += 1
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_dof_scatter_by_repetition(
    data: dict,
    result: dict,
    max_samples_per_rep: int = 1500,
    rng_seed: int = 0,
    rep_titles: list[str] | None = None,
    save_dir: str | None = None,
    show_plot: bool = True,
) -> None:
    """
    For each incremental repetition, plot two scatterplots (DoF 0 and DoF 1):
    true target vs predicted target, using a subsample of windows for readability.
    """
    inc_reps = list(data["inc_reps"])
    stream_ranges = result["stream_ranges"]
    Y_concat = np.concatenate(data["Y_reps"], axis=0)
    y_hat = np.asarray(result["predictions_arr"], dtype=float)

    if Y_concat.ndim != 2 or y_hat.ndim != 2:
        raise ValueError("Expected Y and predictions to be 2D arrays shaped (n_samples, n_dofs).")
    if Y_concat.shape[0] != y_hat.shape[0] or Y_concat.shape[1] != y_hat.shape[1]:
        raise ValueError(f"Shape mismatch: Y {Y_concat.shape} vs predictions {y_hat.shape}.")
    if Y_concat.shape[1] < 2:
        raise ValueError(
            f"Expected at least 2 DoFs for plotting, got output_dim={Y_concat.shape[1]}.")
    if Y_concat.shape[1] > 2:
        y_hat = y_hat[:, :2]
        Y_concat = Y_concat[:, :2]

    if rep_titles is None:
        shock_mapping = data.get("shock_mapping", {})
        rep_titles = [
            f"Rep {r} ({shock_mapping[r]})" if r in shock_mapping else f"Rep {r}"
            for r in inc_reps
        ]

    rng = np.random.default_rng(int(rng_seed))
    max_samples_per_rep = int(max(1, max_samples_per_rep))

    for rep_idx, ((start, end), rep_id) in enumerate(zip(stream_ranges, inc_reps), start=1):
        if start == end:
            continue

        y_true = Y_concat[start:end]
        y_pred = y_hat[start:end]
        n = y_true.shape[0]
        if n == 0:
            continue

        idx = np.arange(n, dtype=int)
        if n > max_samples_per_rep:
            idx = rng.choice(idx, size=max_samples_per_rep, replace=False)
            idx.sort()

        title = rep_titles[rep_idx - 1] if rep_idx - 1 < len(rep_titles) else f"Rep {rep_id}"

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=False, sharey=False)
        fig.suptitle(f"{title} (dataset rep id={rep_id}, n_plot={idx.size}/{n})")

        for dof, ax in enumerate(axes):
            t = y_true[idx, dof]
            p = y_pred[idx, dof]
            ax.scatter(t, p, s=10, alpha=0.35, edgecolors="none")
            lo = float(min(t.min(), p.min()))
            hi = float(max(t.max(), p.max()))
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                pad = 1.0 if lo == hi else 0.05 * (abs(hi) + 1.0)
                lo -= pad
                hi += pad
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.6)
            ax.set_title(f"DoF {dof}")
            ax.set_xlabel("True")
            ax.set_ylabel("Predicted")
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.set_aspect("equal", adjustable="box")

        plt.tight_layout()
        handle_plot_output(save_dir, show_plot)


def plot_experiment(data: dict, result: dict, update_mode: str = "rep", metric_name: str = "RMSE", model_type: str = "iRFF_RR", save_dir: str | None = None, show_plot: bool = True, test_only_reps: set[int] | None = None) -> None:
    errors = result["errors"]
    rep_metric_history = result.get("rep_metric_history", result.get("rep_rmse_history", {}))
    stream_ranges = result["stream_ranges"]
    confidences = result["confidences"]
    rep_metrics = result.get("rep_metrics", result.get("rep_rmses", []))
    rep_pearsons = result.get("rep_pearsons")
    X_reps = data["X_reps"]

    if update_mode == "sample":
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax_global = axes[0]

        err_offset = 0
        stream_x0 = 0
        first_seg = True
        for X_rep in X_reps:
            n = X_rep.shape[0]
            if n == 0:
                continue
            if not first_seg:
                ax_global.axvline(stream_x0 + 0.5, color="k", linestyle="--", alpha=0.3)
            first_seg = False
            seg = errors[err_offset : err_offset + n]
            x = np.arange(stream_x0 + 1, stream_x0 + n + 1)
            ax_global.plot(x, seg, "b-")
            err_offset += n
            stream_x0 += n
        ax_global.set_title(f"Global Incremental {metric_name}")
        ax_global.set_xlabel("Sample Index in Stream")
        ax_global.set_ylabel(metric_name)
        ax_global.grid(True)

        for r, (start_idx, end_idx) in enumerate(stream_ranges, start=1):
            if start_idx == end_idx:
                continue
            ax_global.text(
                (start_idx + end_idx) / 2 + 0.5,
                ax_global.get_ylim()[1],
                f"Rep {r}",
                ha="center",
                va="top",
                fontsize=8,
                color="k",
                alpha=0.7,
            )

        ax_rep = axes[1]
        for rep_label in sorted(rep_metric_history.keys()):
            metric_curve = rep_metric_history[rep_label]
            ax_rep.plot(range(1, len(metric_curve) + 1), metric_curve, label=f"Rep {rep_label}")
        ax_rep.set_title(f"Per-Repetition {metric_name} Evolution")
        ax_rep.set_xlabel("Sample Index within Repetition")
        ax_rep.set_ylabel(metric_name)
        ax_rep.grid(True)
        ax_rep.legend()
        
        fig.tight_layout()
        handle_plot_output(save_dir, show_plot)

    plt.figure(figsize=(12, 7))

    # Fetch the data we need
    inc_reps = data["inc_reps"]
    shock_mapping = data.get("shock_mapping", {})
    test_only_reps = test_only_reps or set()

    # Identify non-empty repetitions and build dynamic labels for them
    non_empty_rep_ids = [
        inc_reps[i] for i, (start, end) in enumerate(stream_ranges) if start != end
    ]
    labels = []
    for r in non_empty_rep_ids:
        shock_info = shock_mapping.get(r)
        is_test = r in test_only_reps
        desc = shock_info if shock_info else "clean"
        
        if is_test:
            labels.append(f"Rep {r}\n({desc}, Test)")
        else:
            labels.append(f"Rep {r}\n({desc})")

    if len(labels) != len(rep_metrics):
        print(f"Warning: Mismatch between number of labels ({len(labels)}) and {metric_name} values ({len(rep_metrics)}). Plot may be incorrect.")

    # Generate a dynamic list of colors so we never run out
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / len(rep_metrics)) for i in range(len(rep_metrics))]
    
    # Plot the bars
    bars = plt.bar(labels, rep_metrics, color=colors)
    
    # Rotate labels slightly so they don't overlap if there are 9+ reps
    plt.xticks(rotation=45, ha='right')

    for bar, val in zip(bars, rep_metrics):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.ylabel(f"Total {metric_name} per Repetition")
    plt.title("Decreasing/Increasing Tendency of Error Across Shocks")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    handle_plot_output(save_dir, show_plot)

    if rep_pearsons:
        plt.figure(figsize=(12, 7))
        
        # We can reuse labels and colors from the previous plot
        bars = plt.bar(labels, rep_pearsons, color=colors)
        
        plt.xticks(rotation=45, ha='right')

        for bar, val in zip(bars, rep_pearsons):
            height = bar.get_height()
            # Adjust text position based on bar height (positive or negative)
            v_offset = 0.01 if height >= 0 else -0.05
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + v_offset,
                f"{val:.3f}",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=10,
            )
        plt.ylabel("Average Pearson Correlation per Repetition")
        plt.title("Model Correlation Across Shocks")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        # Set y-axis limits to be appropriate for Pearson correlation
        y_min = min(min(rep_pearsons) - 0.1, 0) if rep_pearsons else 0
        plt.ylim(bottom=y_min, top=1.05)
        plt.tight_layout()
        handle_plot_output(save_dir, show_plot)

    if update_mode == "sample":
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(confidences) + 1), confidences, "r-", label="Confidence")
        plt.title("Confidence During Stream Phase")
        plt.xlabel("Window Index")
        plt.grid(True)
        plt.legend()

        for r, (start_idx, end_idx) in enumerate(stream_ranges, start=1):
            ax = plt.gca()
            ax.axvline(start_idx + 1, color="k", linestyle="--", alpha=0.3)
            ax.axvline(end_idx, color="k", linestyle="--", alpha=0.3)
            ax.text(
                (start_idx + end_idx) / 2 + 0.5,
                ax.get_ylim()[1],
                f"Rep {r}",
                ha="center",
                va="top",
                fontsize=8,
                color="k",
                alpha=0.7,
            )

        plt.tight_layout()
        handle_plot_output(save_dir, show_plot)
    
    elif update_mode == "rep":
        plt.figure(figsize=(12, 7))
        
        avg_confidences = []
        for start, end in stream_ranges:
            if start == end:
                continue 
            rep_confs = confidences[start:end]
            avg_confidences.append(np.mean(rep_confs) if rep_confs else 0)

        cmap = plt.get_cmap("viridis")
        colors = [cmap(i / len(avg_confidences)) for i in range(len(avg_confidences))]
        
        # The `labels` list was generated for the other bar plot; we need to regenerate it
        # for the confidence plot if it's different (e.g. different non-empty reps)
        # For simplicity, we assume the non-empty reps are the same. If not, this needs adjustment.
        conf_labels = labels # Reuse labels from the error plot

        bars = plt.bar(conf_labels, avg_confidences, color=colors)
        
        plt.xticks(rotation=45, ha='right')
        for bar, conf in zip(bars, avg_confidences):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2, height, f"{conf:.4f}",
                ha="center", va="bottom", fontsize=10,
            )
        
        plt.ylabel("Average Confidence per Repetition")
        plt.title("Block-Level Confidence Across Shocks")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        handle_plot_output(save_dir, show_plot)



def plot_semg_and_predictions(
    data: dict, 
    result: dict, 
    target_rep: int, 
    channel: int | None = None, 
    clean_rep: int = 0, 
    model_type: str = "iRFF_RR",
    save_dir: str | None = None,
    show_plot: bool = True
) -> None:
    """
    Plots the raw clean sEMG, the raw shocked sEMG, and the model's predictions
    to compare the signals and the model's performance on a specific repetition.
    """
    if "raw_emg_by_rep" not in data:
        print("Raw EMG data not found. Ensure 'raw_emg_by_rep' is in the data dictionary.")
        return
        
    raw_clean = data["raw_emg_by_rep"].get(clean_rep)
    raw_shocked = data["raw_emg_by_rep"].get(target_rep)
    
    if raw_clean is None or raw_shocked is None:
        print(f"Data for clean rep ({clean_rep}) or target rep ({target_rep}) not found.")
        return
        
    inc_reps = list(data["inc_reps"])
    if target_rep not in inc_reps:
        print(f"Repetition {target_rep} is not in the tested incremental reps.")
        return
        
    rep_idx = inc_reps.index(target_rep)
    start_idx, end_idx = result["stream_ranges"][rep_idx]
    
    y_true = np.concatenate(data["Y_reps"], axis=0)[start_idx:end_idx]
    y_pred = result["predictions_arr"][start_idx:end_idx]
    num_dofs = min(2, y_true.shape[1])
    
    if channel is not None:
        fig1, axes1 = plt.subplots(2, 1, figsize=(12, 6))
        
        axes1[0].plot(raw_clean[:, channel], color="green", alpha=0.7)
        axes1[0].set_title(f"Clean raw sEMG (Rep {clean_rep}) - Channel {channel}")
        axes1[0].set_ylabel("Amplitude")
        axes1[0].grid(True)
        
        shock_name = data.get("shock_mapping", {}).get(target_rep)
        axes1[1].plot(raw_shocked[:, channel], color="red", alpha=0.7)
        if shock_name:
            axes1[1].set_title(f"Shocked raw sEMG (Rep {target_rep}, {shock_name}) - Channel {channel}")
        else:
            axes1[1].set_title(f"Target raw sEMG (Rep {target_rep}, clean) - Channel {channel}")
        axes1[1].set_ylabel("Amplitude")
        axes1[1].grid(True)
        axes1[1].set_xlabel("Sample Index")
        
        fig1.tight_layout()
        handle_plot_output(save_dir, show_plot)
    
    fig2, axes2 = plt.subplots(num_dofs, 1, figsize=(12, 3 * num_dofs), squeeze=False)
    
    for dof in range(num_dofs):
        ax = axes2[dof, 0]
        ax.plot(y_true[:, dof], label=f"True DoF {dof}", linestyle="--")
        ax.plot(y_pred[:, dof], label=f"Predicted DoF {dof}", alpha=0.7)
        ax.set_title(f"{model_type} Predictions (Rep {target_rep}) - DoF {dof}")
        ax.set_ylabel("Kinematics")
        if dof == num_dofs - 1:
            ax.set_xlabel("Window Index")
        ax.legend()
        ax.grid(True)
    
    fig2.tight_layout()
    handle_plot_output(save_dir, show_plot)


def plot_psd_comparison(
    data: dict,
    target_rep: int,
    channel: int = 0,
    clean_rep: int = 0,
    save_dir: str | None = None,
    show_plot: bool = True
) -> None:
    """
    Plots the Power Spectral Density (PSD) comparison between a clean repetition
    and a target repetition to visualize frequency shifts (e.g., muscle fatigue).
    """
    if "raw_emg_by_rep" not in data or "ds_fs" not in data:
        print("Required data ('raw_emg_by_rep' or 'ds_fs') for PSD plot not found.")
        return

    raw_clean = data["raw_emg_by_rep"].get(clean_rep)
    raw_target = data["raw_emg_by_rep"].get(target_rep)
    fs = data["ds_fs"]

    if raw_clean is None or raw_target is None:
        print(f"Data for clean rep ({clean_rep}) or target rep ({target_rep}) not found.")
        return

    sig_clean = raw_clean[:, channel]
    sig_target = raw_target[:, channel]

    f_clean, pxx_clean = signal.welch(sig_clean, fs, nperseg=1024)
    f_target, pxx_target = signal.welch(sig_target, fs, nperseg=1024)

    plt.figure(figsize=(10, 5))
    plt.semilogy(f_clean, pxx_clean, label=f"Clean (Rep {clean_rep})", color="green", alpha=0.7)
    plt.semilogy(f_target, pxx_target, label=f"Target (Rep {target_rep})", color="red", alpha=0.7)

    shock_name = data.get("shock_mapping", {}).get(target_rep, "clean")
    plt.title(f"PSD Comparison - Channel {channel} (Rep {clean_rep} vs Rep {target_rep} [{shock_name}])")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (V^2/Hz)")
    plt.xlim(left=0, right=fs/2)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    handle_plot_output(save_dir, show_plot)