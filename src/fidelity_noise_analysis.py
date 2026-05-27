import argparse
import glob
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

warnings.filterwarnings("ignore")


def prediction_fidelity(delta_pred, delta_true):
    delta_pred = np.asarray(delta_pred)
    delta_true = np.asarray(delta_true)

    mean_pred = np.mean(delta_pred)
    mean_true = np.mean(delta_true)

    var_pred = np.var(delta_pred)
    var_true = np.var(delta_true)

    covariance = np.mean(delta_pred * delta_true) - mean_pred * mean_true

    if var_pred * var_true == 0:
        return 0.0

    return np.abs(covariance) / np.sqrt(var_pred * var_true)


def load_ensemble_models(model_dir_pattern):
    model_dirs = sorted(glob.glob(model_dir_pattern))

    print(f"Found {len(model_dirs)} models")

    models = []
    seeds = []
    test_data_list = []

    for model_dir in model_dirs:
        seed = int(model_dir.split("_")[-1])
        seeds.append(seed)

        model_path = os.path.join(model_dir, "model.keras")
        test_path = os.path.join(model_dir, "test_data.npz")

        if not os.path.exists(model_path):
            print(f"Skipping {model_dir}: missing model.keras")
            continue

        model = tf.keras.models.load_model(model_path)
        models.append(model)

        if os.path.exists(test_path):
            test_data = np.load(test_path)
            test_data_list.append(
                {
                    "X_test": test_data["X_test"],
                    "y_test": test_data["y_test"],
                }
            )

        print(f"  Loaded model with seed {seed}")

    return models, seeds, test_data_list


def compute_fidelity_vs_noise(
    models,
    X_test,
    y_test,
    noise_levels,
    n_repetitions=10,
):
    n_models = len(models)
    n_noise = len(noise_levels)

    all_fidelities = np.zeros((n_models, n_noise, n_repetitions, 3))

    for m_idx, model in enumerate(models):
        print(f"Model {m_idx + 1}/{n_models}")

        for n_idx, eta in enumerate(noise_levels):
            for r_idx in range(n_repetitions):
                if eta > 0:
                    noise = np.random.uniform(-eta, eta, size=X_test.shape)
                    X_noisy = X_test + noise
                else:
                    X_noisy = X_test

                pred = model.predict(X_noisy, verbose=0)

                f_j1 = prediction_fidelity(pred[:, 0], y_test[:, 0])
                f_j2 = prediction_fidelity(pred[:, 1], y_test[:, 1])
                f_tot = prediction_fidelity(pred, y_test)

                all_fidelities[m_idx, n_idx, r_idx] = [f_j1, f_j2, f_tot]

    f_model_avg = np.mean(all_fidelities, axis=2)

    fidelity_mean = {
        "j1": np.mean(f_model_avg[:, :, 0], axis=0),
        "j2": np.mean(f_model_avg[:, :, 1], axis=0),
        "total": np.mean(f_model_avg[:, :, 2], axis=0),
    }

    fidelity_std = {
        "j1": np.std(f_model_avg[:, :, 0], axis=0),
        "j2": np.std(f_model_avg[:, :, 1], axis=0),
        "total": np.std(f_model_avg[:, :, 2], axis=0),
    }

    raw_data = {
        "all_fidelities": all_fidelities,
        "f_model_avg": f_model_avg,
        "noise_levels": noise_levels,
    }

    return fidelity_mean, fidelity_std, raw_data


def plot_fidelity_curves(
    noise_levels,
    fidelity_mean,
    fidelity_std,
    save_path,
    title,
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = [
        ("j1", "J1", axes[0]),
        ("j2", "J2", axes[1]),
        ("total", "Total", axes[2]),
    ]

    for metric, label, ax in metrics:
        ax.errorbar(
            noise_levels,
            fidelity_mean[metric],
            yerr=fidelity_std[metric],
            fmt="o-",
            capsize=5,
            label="Ensemble mean",
        )

        ax.set_xlabel("Noise level η")
        ax.set_ylabel("Fidelity")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {save_path}")

    return fig


def export_fidelity_table(
    noise_levels,
    fidelity_mean,
    fidelity_std,
    save_path,
):
    with open(save_path, "w") as f:
        f.write(
            "eta\t"
            "J1_mean\tJ1_std\t"
            "J2_mean\tJ2_std\t"
            "J_total_mean\tJ_total_std\n"
        )

        for i, eta in enumerate(noise_levels):
            f.write(
                f"{eta:.4f}\t"
                f"{fidelity_mean['j1'][i]:.4f}\t{fidelity_std['j1'][i]:.4f}\t"
                f"{fidelity_mean['j2'][i]:.4f}\t{fidelity_std['j2'][i]:.4f}\t"
                f"{fidelity_mean['total'][i]:.4f}\t{fidelity_std['total'][i]:.4f}\n"
            )

    print(f"Saved table: {save_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["theory", "enhanced"],
        default="theory",
        help="Analyze either the theory ensemble or the experimental-like enhanced ensemble.",
    )

    parser.add_argument(
        "--models-dir",
        default="models/trained_models",
        help="Directory containing trained model folders.",
    )

    parser.add_argument(
        "--output-dir",
        default="results/metrics",
        help="Directory where fidelity results are saved.",
    )

    parser.add_argument(
        "--n-repetitions",
        type=int,
        default=10,
        help="Number of noise realizations per noise level.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_pattern = os.path.join(args.models_dir, f"{args.mode}_seed_*")

    print(f"Loading {args.mode} ensemble from:")
    print(model_pattern)

    models, seeds, test_data_list = load_ensemble_models(model_pattern)

    if len(models) == 0:
        raise RuntimeError(f"No models found for pattern: {model_pattern}")

    if len(test_data_list) == 0:
        raise RuntimeError("No test_data.npz found in the model folders.")

    X_test = test_data_list[0]["X_test"]
    y_test = test_data_list[0]["y_test"]

    print(f"Loaded test data: {X_test.shape}")

    alpha = np.array([0.0, 0.05, 0.10, 0.15, 0.20])

    if args.mode == "enhanced":
        A = 1.2
    else:
        A = 0.25

    noise_levels = alpha * A

    print(f"Using A = {A} for mode = {args.mode}")
    print("Noise levels:", noise_levels)

    fidelity_mean, fidelity_std, raw_data = compute_fidelity_vs_noise(
        models=models,
        X_test=X_test,
        y_test=y_test,
        noise_levels=noise_levels,
        n_repetitions=args.n_repetitions,
    )

    plot_path = os.path.join(
        args.output_dir,
        f"fidelity_vs_noise_{args.mode}.png",
    )

    table_path = os.path.join(
        args.output_dir,
        f"fidelity_table_{args.mode}.txt",
    )

    data_path = os.path.join(
        args.output_dir,
        f"fidelity_analysis_results_{args.mode}.npz",
    )

    plot_fidelity_curves(
        noise_levels=noise_levels,
        fidelity_mean=fidelity_mean,
        fidelity_std=fidelity_std,
        save_path=plot_path,
        title=f"Fidelity vs noise - {args.mode} ensemble",
    )

    export_fidelity_table(
        noise_levels=noise_levels,
        fidelity_mean=fidelity_mean,
        fidelity_std=fidelity_std,
        save_path=table_path,
    )

    np.savez(
        data_path,
        mode=args.mode,
        A=A,
        alpha=alpha,
        noise_levels=noise_levels,
        seeds=np.array(seeds),
        fidelity_mean_j1=fidelity_mean["j1"],
        fidelity_mean_j2=fidelity_mean["j2"],
        fidelity_mean_total=fidelity_mean["total"],
        fidelity_std_j1=fidelity_std["j1"],
        fidelity_std_j2=fidelity_std["j2"],
        fidelity_std_total=fidelity_std["total"],
        raw_fidelities=raw_data["all_fidelities"],
    )

    print("\nAnalysis completed.")
    print(f"Saved plot:  {plot_path}")
    print(f"Saved table: {table_path}")
    print(f"Saved data:  {data_path}")


if __name__ == "__main__":
    main()
