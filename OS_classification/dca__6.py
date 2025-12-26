#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# PATHS (EDIT THESE)
# -------------------------
OUT_DIR = "./outputs"
ROOT_DIR = "./final_data_aug_bin"

TEST_PROBS_PATH = os.path.join(OUT_DIR, "test_probs.npy")
TEST_LABELS_PATH = os.path.join(ROOT_DIR, "y_test.npy")     # must be 0/1 (or finite maskable)

OOF_PROBS_PATH = os.path.join(OUT_DIR, "oof_probs.npy")
OOF_LABELS_PATH = os.path.join(OUT_DIR, "oof_labels.npy")

# -------------------------
# DCA SETTINGS
# -------------------------
THRESHOLDS = np.linspace(0.01, 0.99, 99)

POS_LABEL = 1


# -------------------------
# Core DCA utilities
# -------------------------
def _load_probs_labels(probs_path: str, labels_path: str):
    if not os.path.exists(probs_path):
        raise FileNotFoundError(f"Missing probs file: {probs_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Missing labels file: {labels_path}")

    p = np.asarray(np.load(probs_path), dtype=float).reshape(-1)
    y = np.asarray(np.load(labels_path), dtype=float).reshape(-1)

    if p.shape[0] != y.shape[0]:
        raise ValueError(f"Length mismatch: probs={p.shape[0]} labels={y.shape[0]}")

    # mask finite labels and probs
    m = np.isfinite(p) & np.isfinite(y)
    p = p[m]
    y = y[m]

    # binarize labels safely
    y = (y == POS_LABEL).astype(int)

    # clip probs into [0,1]
    p = np.clip(p, 0.0, 1.0)

    if p.size == 0:
        raise ValueError("After masking, no samples remain.")

    return p, y


def decision_curve(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray):
    """
    Net Benefit (NB) definition (Vickers & Elkin):
      NB(pt) = TP/N - FP/N * (pt / (1-pt))

    Also returns Treat-All and Treat-None baselines:
      NB_all(pt)  = prevalence - (1-prevalence) * (pt/(1-pt))
      NB_none(pt) = 0
    """
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)
    thresholds = np.asarray(thresholds, dtype=float).reshape(-1)

    N = y_true.size
    prev = float(y_true.mean())

    nb_model = np.zeros_like(thresholds, dtype=float)
    nb_all = np.zeros_like(thresholds, dtype=float)
    nb_none = np.zeros_like(thresholds, dtype=float)

    for i, pt in enumerate(thresholds):
        # avoid numerical blowups
        if pt <= 0.0 or pt >= 1.0:
            nb_model[i] = np.nan
            nb_all[i] = np.nan
            nb_none[i] = np.nan
            continue

        w = pt / (1.0 - pt)

        pred = (y_prob >= pt).astype(int)
        tp = float(((pred == 1) & (y_true == 1)).sum())
        fp = float(((pred == 1) & (y_true == 0)).sum())

        nb_model[i] = (tp / N) - (fp / N) * w
        nb_all[i] = prev - (1.0 - prev) * w
        nb_none[i] = 0.0

    return nb_model, nb_all, nb_none


def save_csv(path: str, thresholds: np.ndarray, nb_model: np.ndarray, nb_all: np.ndarray, nb_none: np.ndarray):
    arr = np.column_stack([thresholds, nb_model, nb_all, nb_none])
    header = "threshold,net_benefit_model,net_benefit_treat_all,net_benefit_treat_none"
    np.savetxt(path, arr, delimiter=",", header=header, comments="")


def plot_dca(title: str, thresholds: np.ndarray, nb_model: np.ndarray, nb_all: np.ndarray, nb_none: np.ndarray, out_png: str):
    plt.figure()
    plt.plot(thresholds, nb_model, label="Model")
    plt.plot(thresholds, nb_all, label="Treat-all")
    plt.plot(thresholds, nb_none, label="Treat-none")
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.ylim(-1.0, 1.0)
    plt.title(title)
    plt.legend()
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# -------------------------
# Main
# -------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) TEST DCA
    p_test, y_test = _load_probs_labels(TEST_PROBS_PATH, TEST_LABELS_PATH)
    nb_test, nb_all_test, nb_none_test = decision_curve(y_test, p_test, THRESHOLDS)
    plot_dca(
        title=f"DCA (TEST) — N={y_test.size}, prevalence={y_test.mean():.3f}",
        thresholds=THRESHOLDS,
        nb_model=nb_test,
        nb_all=nb_all_test,
        nb_none=nb_none_test,
        out_png=os.path.join(OUT_DIR, "dca_test.png"),
    )
    save_csv(
        os.path.join(OUT_DIR, "dca_values_test.csv"),
        THRESHOLDS, nb_test, nb_all_test, nb_none_test
    )
    print(f"[OK] TEST DCA saved: {os.path.join(OUT_DIR, 'dca_test.png')}")

    # 2) OOF (CV) DCA
    p_oof, y_oof = _load_probs_labels(OOF_PROBS_PATH, OOF_LABELS_PATH)
    nb_oof, nb_all_oof, nb_none_oof = decision_curve(y_oof, p_oof, THRESHOLDS)
    plot_dca(
        title=f"DCA (OOF-CV) — N={y_oof.size}, prevalence={y_oof.mean():.3f}",
        thresholds=THRESHOLDS,
        nb_model=nb_oof,
        nb_all=nb_all_oof,
        nb_none=nb_none_oof,
        out_png=os.path.join(OUT_DIR, "dca_oof.png"),
    )
    save_csv(
        os.path.join(OUT_DIR, "dca_values_oof.csv"),
        THRESHOLDS, nb_oof, nb_all_oof, nb_none_oof
    )
    print(f"[OK] OOF DCA saved: {os.path.join(OUT_DIR, 'dca_oof.png')}")

    print("[DONE] Generated two DCA plots + CSV tables in OUT_DIR.")


if __name__ == "__main__":
    main()

