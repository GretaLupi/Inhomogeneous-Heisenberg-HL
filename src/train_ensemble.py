import argparse
import json
import os
from datetime import datetime

import keras
import numpy as np
import tensorflow as tf
from scipy.integrate import cumulative_trapezoid
from scipy.signal import convolve
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, regularizers
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential


def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def prediction_fidelity(pred, true):
    pred, true = np.asarray(pred), np.asarray(true)
    cov = np.mean(pred * true) - np.mean(pred) * np.mean(true)
    return np.abs(cov) / np.sqrt(np.var(pred) * np.var(true))


def build_model(input_dim=600, output_dim=2, lr=3e-4, l2=3e-4, dropout=0.25):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(512, activation="relu", kernel_regularizer=regularizers.l2(l2)),
        BatchNormalization(),
        Dropout(dropout),
        Dense(256, activation="relu", kernel_regularizer=regularizers.l2(l2)),
        BatchNormalization(),
        Dropout(dropout),
        Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2)),
        Dense(output_dim, activation="linear"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.Huber(delta=0.02),
        metrics=[keras.metrics.MeanAbsoluteError()],
    )
    return model


def compute_didv_from_flat(Z_flat, X_flat, n_sites=12, n_bias=200):
    out = []

    for s in range(Z_flat.shape[0]):
        Z = np.asarray(Z_flat[s]).reshape(n_sites, n_bias)
        X = np.asarray(X_flat[s]).reshape(n_sites, n_bias)
        bias = X[0]

        didv = cumulative_trapezoid(Z, bias, axis=1, initial=0.0)
        out.append(didv.astype(np.float32).ravel())

    return np.stack(out, axis=0)


def make_windows(didv, labels, n=12, res=200):
    X_all, y_all = [], []

    for i in range(n - 2):
        X_all.append(didv[:, i * res:(i + 3) * res])
        y_all.append(labels[:, i:i + 2])

    return np.concatenate(X_all, axis=0), np.concatenate(y_all, axis=0)


def split_data_grouped(didv, labels, n=12, res=200, test_size=0.1, val_size=0.2, seed=42):
    idx = np.arange(didv.shape[0])

    idx_train, idx_test = train_test_split(idx, test_size=test_size, random_state=seed)
    idx_train, idx_val = train_test_split(idx_train, test_size=val_size, random_state=seed)

    X_train, y_train = make_windows(didv[idx_train], labels[idx_train], n=n, res=res)
    X_val, y_val = make_windows(didv[idx_val], labels[idx_val], n=n, res=res)
    X_test, y_test = make_windows(didv[idx_test], labels[idx_test], n=n, res=res)

    return X_train, X_val, X_test, y_train, y_val, y_test


def cut_and_resample(y, bias_mev, cut_mev=50.0, n_out=200):
    y = np.asarray(y, dtype=np.float32)
    b = np.asarray(bias_mev, dtype=np.float32)

    mask = (b >= 0) & (b <= cut_mev)
    b_cut = b[mask]
    y_cut = y[mask]

    b_out = np.linspace(0.0, cut_mev, n_out, dtype=np.float32)
    y_out = np.interp(b_out, b_cut, y_cut).astype(np.float32)

    return y_out


def mev_to_idx(mev, res=200, full_mev=100):
    return int(round((mev / full_mev) * res))


def baseline_and_bandscale(y, res=200, full_mev=50, baseline_mev=3, scale_band_mev=(40, 50), eps=1e-12):
    y = y.astype(np.float32).copy()

    b1 = max(1, min(res, mev_to_idx(baseline_mev, res, full_mev)))
    y -= np.mean(y[:b1])

    s0 = max(0, min(res - 1, mev_to_idx(scale_band_mev[0], res, full_mev)))
    s1 = max(s0 + 1, min(res, mev_to_idx(scale_band_mev[1], res, full_mev)))

    scale = np.mean(np.abs(y[s0:s1])) + eps
    y /= scale

    return np.nan_to_num(y)


def add_linear_drift(y, strength_range=(3.2e-5, 5.1e-5), mult_range=(1.5, 3.0)):
    y = y.copy()
    res = len(y)

    threshold = np.mean(y) * np.random.uniform(0.1, 1.2)
    start = int(np.argmax(y > threshold))

    if y[start] <= threshold:
        start = res // 3

    slope = np.random.uniform(*strength_range) * np.random.uniform(*mult_range)
    y[start:] += np.arange(res - start, dtype=np.float32) * slope

    return y


def lorentzian_kernel(size, gamma):
    x = np.arange(size) - (size - 1) / 2
    kernel = 1.0 / (1.0 + (x / gamma) ** 2)
    return (kernel / np.sum(kernel)).astype(np.float32)


def experimental_like_spectrum(y, cut_mev=50.0, offset_range=(0.009, 0.016), noise=0.002):
    y = np.abs(y).astype(np.float32)

    y += np.random.uniform(*offset_range)

    gamma = np.random.uniform(0.5, 2.0)
    kernel_size = int(np.ceil(gamma * 12))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(kernel_size, 5)

    y = convolve(y, lorentzian_kernel(kernel_size, gamma), mode="same")

    y = cut_and_resample(y, np.linspace(0, 100, 200), cut_mev=cut_mev, n_out=200)

    weight = np.linspace(0.6, 1.2, len(y)).astype(np.float32)
    y += np.random.randn(len(y)).astype(np.float32) * noise * weight

    y = add_linear_drift(y)

    y = baseline_and_bandscale(
        y,
        res=200,
        full_mev=cut_mev,
        baseline_mev=3,
        scale_band_mev=(40, cut_mev),
    )

    return y


def enhance_dataset(didv_flat, labels, L=12, res=200, n_enhance=1, cut_mev=50.0, noise=0.002, seed=None):
    if seed is not None:
        np.random.seed(seed)

    N = didv_flat.shape[0]
    didv_exp = np.zeros((N * n_enhance, L, res), dtype=np.float32)

    labels_scaled = (labels - labels.min()) / (labels.max() - labels.min() + 1e-12)
    labels_exp = np.tile(labels_scaled, (n_enhance, 1))

    for a in range(n_enhance):
        for i in range(N):
            for s in range(L):
                y = didv_flat[i, s * res:(s + 1) * res]
                didv_exp[a * N + i, s, :] = experimental_like_spectrum(
                    y,
                    cut_mev=cut_mev,
                    noise=noise,
                )

    return didv_exp.reshape(N * n_enhance, L * res), labels_exp


def load_npz_dataset(paths):
    X_all, Z_all, J_all = [], [], []

    for path in paths:
        data = np.load(path)
        X_all.append(data["X"])
        Z_all.append(data["Z"])
        J_all.append(data["J"])

    return np.concatenate(X_all), np.concatenate(Z_all), np.concatenate(J_all)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["theory", "enhanced"], default="theory")
    parser.add_argument("--dataset", nargs="+", required=True)
    parser.add_argument("--output-dir", default="models/trained_models")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--cut-mev", type=float, default=50.0)
    parser.add_argument("--n-enhance", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0.002)

    args = parser.parse_args()

    job_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))
    seed = args.base_seed + job_id

    set_seed(seed)

    model_dir = os.path.join(args.output_dir, f"{args.mode}_seed_{seed}")
    os.makedirs(model_dir, exist_ok=True)

    X_data, Z, target = load_npz_dataset(args.dataset)

    didv = compute_didv_from_flat(
        Z_flat=Z,
        X_flat=X_data,
        n_sites=12,
        n_bias=200,
    )

    if args.mode == "theory":
        labels = (target - target.min()) / (target.max() - target.min() + 1e-12)
        didv_input = didv

    elif args.mode == "enhanced":
        didv_input, labels = enhance_dataset(
            didv,
            target,
            L=12,
            res=200,
            n_enhance=args.n_enhance,
            cut_mev=args.cut_mev,
            noise=args.noise,
            seed=seed,
        )

    X_train, X_val, X_test, y_train, y_val, y_test = split_data_grouped(
        didv_input,
        labels,
        n=12,
        res=200,
        test_size=0.1,
        val_size=0.2,
        seed=seed,
    )

    model = build_model()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(os.path.join(model_dir, "training_log.csv")),
    ]

    history = model.fit(
        X_train,
        y_train,
        batch_size=128,
        epochs=100,
        callbacks=callbacks,
        validation_data=(X_val, y_val),
        verbose=1,
    )

    predictions = model.predict(X_test)

    metrics = {
        "mode": args.mode,
        "job_id": job_id,
        "seed": seed,
        "cut_mev": args.cut_mev,
        "fidelity_j1": float(prediction_fidelity(predictions[:, 0], y_test[:, 0])),
        "fidelity_j2": float(prediction_fidelity(predictions[:, 1], y_test[:, 1])),
        "fidelity_total": float(prediction_fidelity(predictions, y_test)),
        "val_loss": float(min(history.history["val_loss"])),
        "train_loss": float(min(history.history["loss"])),
        "epochs_completed": len(history.history["loss"]),
    }

    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    model.save(os.path.join(model_dir, "model.keras"))

    np.savez(
        os.path.join(model_dir, "test_data.npz"),
        X_test=X_test,
        y_test=y_test,
        predictions=predictions,
    )

    with open(os.path.join(model_dir, "report.txt"), "w") as f:
        f.write(f"Training report - seed {seed}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(json.dumps(metrics, indent=4))

    print(f"Training completed: {model_dir}")
    print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    main()
