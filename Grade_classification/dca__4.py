#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# PATHS (EDIT THESE)
# -------------------------
OUT_DIR  = "./outputs"
ROOT_DIR = "./final_data_aug"

TEST_PROB_PATH = os.path.join(OUT_DIR, "test_prob.npy")
OOF_PROB_PATH  = os.path.join(OUT_DIR, "oof_prob.npy")
OOF_TRUE_PATH  = os.path.join(OUT_DIR, "oof_true.npy")
Y_TEST_PATH    = os.path.join(ROOT_DIR, "y_test.npy")

# -------------------------
# EVENT DEFINITION (EDIT THESE)
# -------------------------
# Binary DCA between two classes only; others are ignored.
POS_CLASS = 0
NEG_CLASS = 2

# -------------------------
# LABEL REMAP FOR y_test.npy
# -------------------------
LABEL_VALUES = [2, 3, 4]

# -------------------------
# DCA SETTINGS
# -------------------------
THRESHOLDS = np.linspace(0.01, 0.99, 99)


# -------------------------
# Helpers
# -------------------------
def remap_labels(y_raw: np.ndarray, label_values):
    """
    Map raw labels in label_values -> {0,1,...,K-1}.
    Keeps NaNs as NaN.
    """
    y = np.asarray(y_raw, dtype=float).copy()
    mapping = {int(v): i for i, v in enumerate(label_values)}
    m = ~np.isnan(y)
    # vectorized map
    y[m] = np.vectorize(lambda z: mapping[int(z)])(y[m])
    return y


def load_probs(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    arr = np.asarray(np.load(path), dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D probs array at {path}, got shape {arr.shape}")
    return np.clip(arr, 0.0, 1.0)


def load_labels(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    y = np.asarray(np.load(path), dtype=float).reshape(-1)
    return y


def filter_posneg(y_class: np.ndarray, p_pos: np.ndarray, pos_class: int, neg_class: int):
    """
    Keep only samples whose y is either pos_class or neg_class.
    Return binary y (1=pos, 0=neg) and corresponding p_pos.
    """
    y = np.asarray(y_class, dtype=float).reshape(-1)
    p = np.asarray(p_pos, dtype=float).reshape(-1)

    m = np.isfinite(y) & np.isfinite(p)
    y = y[m].astype(int)
    p = p[m]

    keep = (y == int(pos_class)) | (y == int(neg_class))
    y = y[keep]
    p = p[keep]

    if y.size == 0:
        raise ValueError("After filtering to POS/NEG classes, no samples remain.")

    y_bin = (y == int(pos_class)).astype(int)
    return y_bin, p


def decision_curve(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray):
    """
    Net Benefit (NB):
      NB(pt) = TP/N - FP/N * (pt/(1-pt))
    Treat-all:
      NB_all(pt) = prev - (1-prev) * (pt/(1-pt))
    Treat-none:
      NB_none = 0
    """
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)
    thresholds = np.asarray(thresholds, dtype=float).reshape(-1)

    N = y_true.size
    prev = float(y_true.mean())

    nb_model = np.zeros_like(thresholds, dtype=float)
    nb_all   = np.zeros_like(thresholds, dtype=float)
    nb_none  = np.zeros_like(thresholds, dtype=float)

    for i, pt in enumerate(thresholds):
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
        nb_all[i]   = prev - (1.0 - prev) * w
        nb_none[i]  = 0.0

    return nb_model, nb_all, nb_none


def save_csv(path: str, thresholds, nb_model, nb_all, nb_none):
    arr = np.column_stack([thresholds, nb_model, nb_all, nb_none])
    header = "threshold,net_benefit_model,net_benefit_treat_all,net_benefit_treat_none"
    np.savetxt(path, arr, delimiter=",", header=header, comments="")


def plot_dca(title: str, thresholds, nb_model, nb_all, nb_none, out_png: str,
             ylim=None, xlim=(0.0, 1.0)):
    fig, ax = plt.subplots()
    ax.plot(thresholds, nb_model, label="Model")
    ax.plot(thresholds, nb_all, label="Treat-all")
    ax.plot(thresholds, nb_none, label="Treat-none")

    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title(title)

    ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.legend()
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# -------------------------
# Main
# -------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---------- LOAD PROBS ----------
    test_prob = load_probs(TEST_PROB_PATH)  # (N_test, K)
    oof_prob  = load_probs(OOF_PROB_PATH)   # (N_oof,  K)
    oof_true  = np.asarray(np.load(OOF_TRUE_PATH), dtype=float).reshape(-1)

    # ---------- LOAD & MAP TEST LABELS ----------
    y_test_raw = load_labels(Y_TEST_PATH)
    if LABEL_VALUES is not None:
        y_test = remap_labels(y_test_raw, LABEL_VALUES)
    else:
        y_test = y_test_raw

    # ---------- BASIC SHAPE CHECKS ----------
    K = test_prob.shape[1]
    if POS_CLASS < 0 or POS_CLASS >= K or NEG_CLASS < 0 or NEG_CLASS >= K:
        raise ValueError(f"POS_CLASS/NEG_CLASS must be within [0, {K-1}]. Got POS={POS_CLASS}, NEG={NEG_CLASS}")

    if y_test.shape[0] != test_prob.shape[0]:
        raise ValueError(f"TEST length mismatch: y_test={y_test.shape[0]} vs test_prob={test_prob.shape[0]}")
    if oof_true.shape[0] != oof_prob.shape[0]:
        raise ValueError(f"OOF length mismatch: oof_true={oof_true.shape[0]} vs oof_prob={oof_prob.shape[0]}")

    # ---------- DEFINE EVENT PROBABILITY ----------
    # For pos-vs-neg DCA, use P(pos_class).
    p_test_pos = test_prob[:, POS_CLASS]
    p_oof_pos  = oof_prob[:, POS_CLASS]

    # ---------- FILTER TO POS/NEG ONLY ----------
    y_test_bin, p_test_pos = filter_posneg(y_test, p_test_pos, POS_CLASS, NEG_CLASS)
    y_oof_bin,  p_oof_pos  = filter_posneg(oof_true, p_oof_pos, POS_CLASS, NEG_CLASS)

    # ---------- DCA ----------
    nb_test, nb_all_test, nb_none_test = decision_curve(y_test_bin, p_test_pos, THRESHOLDS)
    nb_oof,  nb_all_oof,  nb_none_oof  = decision_curve(y_oof_bin,  p_oof_pos,  THRESHOLDS)

    # ---------- SAVE ----------
    tag = f"pos{POS_CLASS}_vs_neg{NEG_CLASS}"

    out_test_png = os.path.join(OUT_DIR, f"dca_test_{tag}.png")
    out_oof_png  = os.path.join(OUT_DIR, f"dca_oof_{tag}.png")
    out_test_csv = os.path.join(OUT_DIR, f"dca_values_test_{tag}.csv")
    out_oof_csv  = os.path.join(OUT_DIR, f"dca_values_oof_{tag}.csv")

    plot_dca(
        title=f"DCA (TEST) [{tag}] — N={y_test_bin.size}, prev={y_test_bin.mean():.3f}",
        thresholds=THRESHOLDS,
        nb_model=nb_test, nb_all=nb_all_test, nb_none=nb_none_test,
        out_png=out_test_png,
        ylim=(-0.5, 0.5),
        xlim=(0.0, 1.0)
    )
    plot_dca(
        title=f"DCA (OOF) [{tag}] — N={y_oof_bin.size}, prev={y_oof_bin.mean():.3f}",
        thresholds=THRESHOLDS,
        nb_model=nb_oof, nb_all=nb_all_oof, nb_none=nb_none_oof,
        out_png=out_oof_png,
        ylim=(-1.0, 1.0),
        xlim=(0.0, 1.0)
    )

    save_csv(out_test_csv, THRESHOLDS, nb_test, nb_all_test, nb_none_test)
    save_csv(out_oof_csv,  THRESHOLDS, nb_oof,  nb_all_oof,  nb_none_oof)

    print(f"[OK] Saved TEST DCA: {out_test_png}")
    print(f"[OK] Saved  OOF DCA: {out_oof_png}")
    print(f"[OK] Saved CSVs:\n  {out_test_csv}\n  {out_oof_csv}")
    print("[DONE]")


if __name__ == "__main__":
    main()

