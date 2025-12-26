#!/usr/bin/env python3
import os, json, math, pathlib, csv
import numpy as np
import gc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from tensorflow.keras.utils import register_keras_serializable
from math import gcd
import nibabel as nib
from nibabel.processing import resample_from_to
from scipy.ndimage import gaussian_filter
from sklearn.metrics import f1_score

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.keras.utils.set_random_seed(42)

# -----------------------
# Utilities / Params
# -----------------------
# ========= IG UTILS FOR MIXED INPUT (IMAGE + TABULAR) =========
def _pick_target_scalar(y, target_class):
    """Return a scalar target for IG.
       - If model has 1 sigmoid output, use y for class-1, (1 - y) for class-0.
       - If model has K outputs, slice the requested class.
    """
    k = int(y.shape[-1])
    if k == 1:  # binary sigmoid
        return y if target_class == 1 else (1.0 - y)
    else:
        return y[..., target_class:target_class+1]

def _best_threshold(y_true: np.ndarray, y_prob: np.ndarray, mode: str = "youden"):
    """
    Pick a probability threshold on OOF to maximize a metric.
    mode: "youden" (sensitivity + specificity - 1) or "f1".
    Returns (best_thr, best_score).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    thrs = np.linspace(0.0, 1.0, 1001)
    best_thr, best_score = 0.5, -1.0
    for t in thrs:
        pred = (y_prob >= t).astype(int)
        tp = float(((pred == 1) & (y_true == 1)).sum())
        tn = float(((pred == 0) & (y_true == 0)).sum())
        fp = float(((pred == 1) & (y_true == 0)).sum())
        fn = float(((pred == 0) & (y_true == 1)).sum())
        if mode == "f1":
            denom = (2*tp + fp + fn)
            score = (2*tp / denom) if denom > 0 else 0.0
        else:  # youden
            sens = tp / (tp + fn + 1e-8)
            spec = tn / (tn + fp + 1e-8)
            score = sens + spec - 1.0
        if score > best_score:
            best_score, best_thr = score, float(t)
    return best_thr, best_score

def _ensure_5d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3:
        x = x[..., None]
    assert x.ndim == 4, f"Expected (D,H,W,C), got {x.shape}"
    return x[None, ...]  # (1,D,H,W,C)

def _ig_single_mixed(model, vol_4d, tab_vec, target_class,
                     m_steps=64, n_smooth=4, noise_std=0.02):
    """
    Integrated Gradients for a model with TWO inputs:
      - image volume (D,H,W,C)
      - tabular vector (T,)
    Returns: (ig_img: D,H,W,C), (ig_tab: T,)
    """
    x_img_np = _ensure_5d(vol_4d)            # (1,D,H,W,C)
    x_tab_np = np.asarray(tab_vec, np.float32)[None, ...]  # (1,T)

    x_img_tf = tf.convert_to_tensor(x_img_np)
    x_tab_tf = tf.convert_to_tensor(x_tab_np)

    # Baselines: zeros
    b_img = tf.zeros_like(x_img_tf)
    b_tab = tf.zeros_like(x_tab_tf)

    alphas = tf.linspace(0.0, 1.0, m_steps)
    repeats = max(1, int(n_smooth))

    ig_img_total = np.zeros_like(x_img_np, dtype=np.float32)
    ig_tab_total = np.zeros_like(x_tab_np, dtype=np.float32)

    for _ in range(repeats):
        # SmoothGrad noise on image only (tabular noise usually unnecessary)
        if n_smooth > 0:
            x_use_img_np = x_img_np.copy()
            std = float(x_use_img_np.std())
            if std > 0:
                x_use_img_np += np.random.normal(0.0, noise_std * std, size=x_use_img_np.shape).astype(np.float32)
            x_use_img = tf.convert_to_tensor(x_use_img_np)
        else:
            x_use_img = x_img_tf

        x_use_tab = x_tab_tf

        d_img = x_use_img - b_img
        d_tab = x_use_tab - b_tab

        g_acc_img = tf.zeros_like(x_use_img)
        g_acc_tab = tf.zeros_like(x_use_tab)

        for a in tf.unstack(alphas):
            x_img_step = b_img + a * d_img
            x_tab_step = b_tab + a * d_tab
            with tf.GradientTape(persistent=True) as tape:
                tape.watch([x_img_step, x_tab_step])
                y = model([x_img_step, x_tab_step], training=False)
                y_tar = _pick_target_scalar(y, target_class)

            g_img = tape.gradient(y_tar, x_img_step)
            g_tab = tape.gradient(y_tar, x_tab_step)
            del tape

            g_acc_img += g_img
            g_acc_tab += g_tab

        avg_g_img = g_acc_img / float(m_steps)
        avg_g_tab = g_acc_tab / float(m_steps)

        ig_img = (x_use_img - b_img) * avg_g_img
        ig_tab = (x_use_tab - b_tab) * avg_g_tab

        ig_img_total += ig_img.numpy()
        ig_tab_total += ig_tab.numpy()

    ig_img_total = ig_img_total / float(repeats)
    ig_tab_total = ig_tab_total / float(repeats)

    return ig_img_total[0], ig_tab_total[0, :]  # (D,H,W,C), (T,)

def _ig_tab_only(model, vol_4d, tab_vec, target_class,
                 tab_baseline=None, m_steps=64):
    x_img = tf.convert_to_tensor(_ensure_5d(vol_4d))
    x_tab = tf.convert_to_tensor(np.asarray(tab_vec, np.float32)[None, ...])
    b_tab = tf.zeros_like(x_tab) if tab_baseline is None else tf.convert_to_tensor(
        np.asarray(tab_baseline, np.float32)[None, ...]
    )

    d_tab = x_tab - b_tab
    alphas = tf.linspace(0.0, 1.0, m_steps)
    g_acc = tf.zeros_like(x_tab)

    for a in tf.unstack(alphas):
        x_tab_step = b_tab + a * d_tab
        with tf.GradientTape() as tape:
            tape.watch(x_tab_step)
            y = model([x_img, x_tab_step], training=False)
            y_tar = y[..., target_class:target_class+1]
        g = tape.gradient(y_tar, x_tab_step)
        g_acc += g

    avg_g = g_acc / float(m_steps)
    ig_tab = (x_tab - b_tab) * avg_g
    return ig_tab.numpy()[0]  # (T,)

def _ig_img_only(model, vol_4d, tab_vec, target_class,
                 m_steps=64, n_smooth=4, noise_std=0.02):
    x_img_np = _ensure_5d(vol_4d)
    x_tab = tf.convert_to_tensor(np.asarray(tab_vec, np.float32)[None, ...])
    b_img = tf.zeros_like(tf.convert_to_tensor(x_img_np))

    alphas = tf.linspace(0.0, 1.0, m_steps)
    repeats = max(1, int(n_smooth))
    ig_img_total = np.zeros_like(x_img_np, dtype=np.float32)

    for _ in range(repeats):
        if n_smooth > 0:
            xi = x_img_np.copy()
            std = float(xi.std())
            if std > 0:
                xi += np.random.normal(0.0, noise_std * std, size=xi.shape).astype(np.float32)
            x_img = tf.convert_to_tensor(xi)
        else:
            x_img = tf.convert_to_tensor(x_img_np)

        d_img = x_img - b_img
        g_acc = tf.zeros_like(x_img)

        for a in tf.unstack(alphas):
            x_img_step = b_img + a * d_img
            with tf.GradientTape() as tape:
                tape.watch(x_img_step)
                y = model([x_img_step, x_tab], training=False)
                y_tar = _pick_target_scalar(y, target_class)

            g = tape.gradient(y_tar, x_img_step)
            g_acc += g

        avg_g = g_acc / float(m_steps)
        ig_img = (x_img - b_img) * avg_g
        ig_img_total += ig_img.numpy()

    return (ig_img_total / float(repeats))[0]  # (D,H,W,C)

def _channel_weights_from_ig(ig_vol: np.ndarray, eps: float = 1e-8):
    ch = np.sum(np.abs(ig_vol), axis=(0,1,2)).astype(np.float32)  # (C,)
    s = float(np.sum(ch))
    if s < eps:
        return np.ones_like(ch, dtype=np.float32) / max(1, ch.size)
    return ch / s

def _tabular_weights_from_ig(ig_tab: np.ndarray, eps: float = 1e-8):
    w = np.abs(np.asarray(ig_tab, np.float32))
    s = float(np.sum(w))
    if s < eps:
        return np.ones_like(w, dtype=np.float32) / max(1, w.size)
    return w / s

def _postprocess_ig(ig_vol, vol_4d, sigma=1.0, air_pct=10.0,
                    p_low=1.0, p_high=99.5, bg_factor=0.3):
    vol_4d = np.asarray(vol_4d, dtype=np.float32)
    ig_abs = np.sum(np.abs(ig_vol), axis=-1).astype(np.float32)  # (D,H,W)

    padmask = (np.max(vol_4d, axis=-1) > 0)
    mean_img = np.mean(vol_4d, axis=-1)
    if padmask.any():
        t_air = np.percentile(mean_img[padmask], air_pct)
        brainmask = padmask & (mean_img > t_air)
    else:
        brainmask = padmask

    ig_abs[~brainmask] = 0.0
    if sigma > 0:
        ig_abs = gaussian_filter(ig_abs, sigma=sigma, mode="nearest")

    vals = ig_abs[brainmask]
    if vals.size > 0:
        lo = np.percentile(vals, p_low)
        hi = np.percentile(vals, p_high)
        hi = max(hi, lo + 1e-6)
        ig_norm = np.clip((ig_abs - lo) / (hi - lo), 0, 1)
    else:
        ig_norm = np.zeros_like(ig_abs, dtype=np.float32)

    m = ig_norm[brainmask].mean() if brainmask.any() else 0.0
    ig_norm[ig_norm < bg_factor * m] = 0.0
    return ig_norm, brainmask

def _save_ig_with_affine(
    ig_xyz: np.ndarray, src_path: str, box_xyz: tuple, pad_offsets: tuple,
    out_path: str, tight: bool = False, t1_path: str | None = None,
):
    src_img = nib.load(src_path)
    src_aff = src_img.affine.copy()
    x0,x1,y0,y1,z0,z1 = map(int, box_xyz)
    sox,soy,soz = map(int, pad_offsets)

    if tight:
        xs, ys, zs = x1-x0, y1-y0, z1-z0
        core = ig_xyz[sox:sox+xs, soy:soy+ys, soz:soz+zs]
        arr_xyz = core.astype(np.float32, copy=False)
        new_aff = src_aff.copy()
        new_aff[:3,3] = (src_aff @ np.array([x0, y0, z0, 1.0]))[:3]
    else:
        arr_xyz = ig_xyz.astype(np.float32, copy=False)
        ox, oy, oz = (x0 - sox, y0 - soy, z0 - soz)
        new_aff = src_aff.copy()
        new_aff[:3,3] = (src_aff @ np.array([ox, oy, oz, 1.0]))[:3]

    img = nib.Nifti1Image(arr_xyz, new_aff, header=src_img.header)
    img.set_sform(new_aff, code=1)
    img.set_qform(new_aff, code=1)
    nib.save(img, out_path)

    if t1_path is not None:
        try:
            t1_img = nib.load(t1_path)
            ig_on_t1 = resample_from_to(img, t1_img, order=1)
            p = out_path.replace(".nii.gz", "_onT1.nii.gz")
            nib.save(ig_on_t1, p)
        except Exception as e:
            print(f"[HEATMAP] T1 resample failed: {e}")

def aggregate_usage_mixed(model, X_mem, T_mem, indices, probs=None,
                          steps=16, smooth_repeats=3, noise_std=0.02,
                          post_sigma=1.0, verbose_every=2,
                          tab_names=None, tab_baseline=None):
    """
    For each sample:
      - run IG w.r.t. predicted class
      - return:
          * image channel weights (C, sum=1)
          * tabular feature weights (T, sum=1)
          * modality share: img_share, tab_share (sum=1)
    Aggregates mean/std across the set, and returns per-sample rows for CSV.
    """
    ch_rows = []
    tb_rows = []
    md_rows = []
    combined_rows = []
    
    for c, i in enumerate(indices):
        vol = np.array(X_mem[i], dtype=np.float32)              # (D,H,W,C)
        tab = np.array(T_mem[i], dtype=np.float32).ravel()      # (T,)
        # pick class
        if probs is not None:
            pi = probs[i]
            target_class = int(pi >= 0.5) if np.ndim(pi) == 0 else int(np.argmax(pi))
        else:
            p = model.predict([vol[None, ...], tab[None, ...]], verbose=0)[0]
            target_class = int(np.argmax(p))

        # FAIR modality shares: vary one input at a time
        ig_img_raw = _ig_img_only(model, vol, tab, target_class,
                                  m_steps=steps, n_smooth=smooth_repeats, noise_std=noise_std)
        ig_tab_raw = _ig_tab_only(model, vol, tab, target_class,
                                  tab_baseline=tab_baseline, m_steps=steps)

        # Optional smoothing ONLY for display of channel mix
        ig_img_vis = ig_img_raw
        if post_sigma and post_sigma > 0:
            ig_img_vis = gaussian_filter(ig_img_raw, sigma=(post_sigma, post_sigma, post_sigma, 0), mode="nearest")

        # inside-branch weights
        w_img_ch = _channel_weights_from_ig(ig_img_vis)  # C
        w_tab    = _tabular_weights_from_ig(ig_tab_raw)  # T

        # modality shares from RAW (no smoothing)
        img_abs = float(np.sum(np.abs(ig_img_raw)))
        tab_abs = float(np.sum(np.abs(ig_tab_raw)))
        denom   = max(img_abs + tab_abs, 1e-8)
        img_share = img_abs / denom
        tab_share = tab_abs / denom

        ch_rows.append(w_img_ch.tolist())
        tb_rows.append(w_tab.tolist())
        md_rows.append([img_share, tab_share])

        # 7-way vector (sum=1)
        comb_img = img_share * w_img_ch
        comb_tab = tab_share * w_tab
        comb_all = np.concatenate([comb_img, comb_tab])
        s = comb_all.sum()
        if s > 0:
            comb_all = comb_all / s
        combined_rows.append(comb_all.tolist())

        if verbose_every and ((c+1) % verbose_every == 0):
            print(f"[MIXED-IG] processed {c+1} samples...")

    ch_arr = np.asarray(ch_rows, dtype=np.float32) if ch_rows else np.zeros((0, X_mem.shape[-1]), np.float32)
    tb_arr = np.asarray(tb_rows, dtype=np.float32) if tb_rows else np.zeros((0, T_mem.shape[-1]), np.float32)
    md_arr = np.asarray(md_rows, dtype=np.float32) if md_rows else np.zeros((0, 2), np.float32)
    comb_arr = (np.asarray(combined_rows, np.float32)
                if combined_rows else
                np.zeros((0, X_mem.shape[-1] + T_mem.shape[-1]), np.float32))
    out = {
        "image_channels": {
            "mean": ch_arr.mean(axis=0).tolist() if ch_arr.size else [],
            "std":  ch_arr.std(axis=0).tolist()  if ch_arr.size else [],
            "n":    int(ch_arr.shape[0]),
            "per_sample": ch_rows,
        },
        "tabular": {
            "names": (tab_names if tab_names is not None else [f"f{i}" for i in range(T_mem.shape[-1])]),
            "mean": tb_arr.mean(axis=0).tolist() if tb_arr.size else [],
            "std":  tb_arr.std(axis=0).tolist()  if tb_arr.size else [],
            "n":    int(tb_arr.shape[0]),
            "per_sample": tb_rows,
        },
        "modality_share": {
            "mean": md_arr.mean(axis=0).tolist() if md_arr.size else [],
            "std":  md_arr.std(axis=0).tolist()  if md_arr.size else [],
            "n":    int(md_arr.shape[0]),
            "per_sample": md_rows,
        },
        "combined_7": {  # <-- NEW
            "names": [f"ch{i}" for i in range(X_mem.shape[-1])] + (
                list(tab_names) if tab_names is not None else [f"f{i}" for i in range(T_mem.shape[-1])]
            ),
            "mean": comb_arr.mean(axis=0).tolist() if comb_arr.size else [],
            "std":  comb_arr.std(axis=0).tolist()  if comb_arr.size else [],
            "n":    int(comb_arr.shape[0]),
            "per_sample": combined_rows,
        },
    }

    return out

# ========= HEATMAP & FEATURE DUMPS AT TEST =========
def compute_box_for_path(src_path: str, side_mm: float = 70.0, top_pct: float = 0.01):
    """
    Return (x0,x1,y0,y1,z0,z1) in voxel indices for a cube of physical side 'side_mm',
    centered in X/Y, and placed in Z so that its top sits at (1 - top_pct) of the volume.
    This matches the logic used when reconstructing pad offsets for IG re-projection.
    """
    img = nib.load(src_path)
    X, Y, Z = img.shape[:3]
    zx, zy, zz = img.header.get_zooms()[:3]

    # side in voxels per axis (respect physical mm)
    sx = int(round(side_mm / float(zx)))
    sy = int(round(side_mm / float(zy)))
    sz = int(round(side_mm / float(zz)))

    sx = max(1, min(sx, X))
    sy = max(1, min(sy, Y))
    sz = max(1, min(sz, Z))

    # X/Y: center crop
    x0 = (X - sx) // 2
    y0 = (Y - sy) // 2
    x1 = x0 + sx
    y1 = y0 + sy

    # Z: place cube so that its TOP is at (1 - top_pct) of the volume
    z_top = int(round((1.0 - float(top_pct)) * Z))
    z1 = max(sz, min(Z, z_top))  # clamp to [sz, Z]
    z0 = z1 - sz

    # Final safety clamps
    x0 = max(0, min(x0, X - sx)); x1 = x0 + sx
    y0 = max(0, min(y0, Y - sy)); y1 = y0 + sy
    z0 = max(0, min(z0, Z - sz)); z1 = z0 + sz

    return (x0, x1, y0, y1, z0, z1)

def save_one_test_heatmap_mixed(
    model, Xt, Tt, test_prob, OUT, ROOT, P,
    pick_class=1,                        # positive class (dead=1)
    sigma=1.2, bg_factor=0.4,
    tab_names=("Age","Sex","EOR")
):
    IDS_TEST_PATH = os.path.join(ROOT, "ids_test.npy")
    PATHS_TEST_PATH = os.path.join(ROOT, "paths_test.npy")
    ids_test = np.load(IDS_TEST_PATH, allow_pickle=True) if os.path.exists(IDS_TEST_PATH) \
               else np.array([f"sample_{i}" for i in range(Xt.shape[0])])
    paths_test = np.load(PATHS_TEST_PATH, allow_pickle=True) if os.path.exists(PATHS_TEST_PATH) \
                 else np.array([None]*Xt.shape[0])

    # choose sample
    # --- normalize test_prob to a 2D (N,2) array for binary logic ---
    if test_prob.ndim == 1:
        prob_pos = np.asarray(test_prob, dtype=float)
        prob_neg = 1.0 - prob_pos
        probs2 = np.stack([prob_neg, prob_pos], axis=1)  # (N,2), [:,1] is class-1
    else:
        probs2 = test_prob

    # test_prob is 1-D: P(class=1)
    prob1 = np.asarray(test_prob, float).reshape(-1)
    pred = (prob1 >= 0.5).astype(int)
    if pick_class == 1:
        cand = np.where(pred == 1)[0]
        best_idx = int(cand[np.argmax(prob1[cand])]) if cand.size > 0 else int(np.argmax(prob1))
    else:  # pick_class == 0
        cand = np.where(pred == 0)[0]
        # "confidence" for class 0 = 1 - prob1
        conf0 = 1.0 - prob1
        best_idx = int(cand[np.argmax(conf0[cand])]) if cand.size > 0 else int(np.argmax(conf0))

    sid = str(ids_test[best_idx]) if best_idx < len(ids_test) else f"sample_{best_idx}"
    src_path = str(paths_test[best_idx]) if best_idx < len(paths_test) else None

    print(f"[HEATMAP] Selected test sample: idx={best_idx}, sid={sid}")

    # run IG for chosen sample
    vol = np.array(Xt[best_idx], dtype=np.float32)
    tab = np.array(Tt[best_idx], dtype=np.float32)
    target_class = int(np.argmax(probs2[best_idx]))  # will be 0 or 1
    ig_img, ig_tab = _ig_single_mixed(model, vol, tab,
                                      target_class, m_steps=128, n_smooth=16, noise_std=0.02)

    ig_norm, _ = _postprocess_ig(ig_img, vol, sigma=sigma, bg_factor=bg_factor)

    out_heat = os.path.join(OUT, "heatmaps")
    os.makedirs(out_heat, exist_ok=True)

    if not src_path or not os.path.exists(src_path):
        print("[HEATMAP] Missing source path; saving fallback identity NIfTI.")
        nib.save(nib.Nifti1Image(ig_norm.astype(np.float32), np.eye(4)),
                 os.path.join(out_heat, f"{sid}__class{target_class}_IGheatmap.nii.gz"))
    else:
        # rebuild box + padding to align affine
        SIDE_MM = P.get("preproc", {}).get("side_mm", 70.0)
        TOP_PCT = P.get("preproc", {}).get("top_pct", 0.01)
        box = compute_box_for_path(src_path, side_mm=SIDE_MM, top_pct=TOP_PCT)

        x0,x1,y0,y1,z0,z1 = map(int, box)
        xs, ys, zs = (x1-x0, y1-y0, z1-z0)
        tgtX, tgtY, tgtZ, _ = Xt.shape[1:]
        sox = max((tgtX - xs) // 2, 0)
        soy = max((tgtY - ys) // 2, 0)
        soz = max((tgtZ - zs) // 2, 0)

        out_full  = os.path.join(out_heat, f"{sid}__class{target_class}_IGheatmap.nii.gz")
        _save_ig_with_affine(
            ig_xyz=ig_norm, src_path=src_path,
            box_xyz=box, pad_offsets=(sox,soy,soz),
            out_path=out_full, tight=False, t1_path=None
        )

        out_tight = os.path.join(out_heat, f"{sid}__class{target_class}_IGheatmap__tight.nii.gz")
        _save_ig_with_affine(
            ig_xyz=ig_norm, src_path=src_path,
            box_xyz=box, pad_offsets=(sox,soy,soz),
            out_path=out_tight, tight=True, t1_path=None
        )

        print(f"[HEATMAP] Saved: {out_full}")
        print(f"[HEATMAP] Saved: {out_tight}")

    # save tabular attributions for this subject
    tab_w = _tabular_weights_from_ig(ig_tab)
    with open(os.path.join(out_heat, f"{sid}__tabular_IG.json"), "w") as f:
        json.dump({
            "sid": sid,
            "target_class": int(target_class),
            "tab_names": list(tab_names),
            "weights_sum1": tab_w.tolist(),
            "raw_ig": np.asarray(ig_tab, dtype=float).tolist(),
        }, f, indent=2)

def make_memmaps_bin(root, x_name, y_name):
    x = np.load(os.path.join(root, x_name), mmap_mode="r")
    y = np.load(os.path.join(root, y_name), mmap_mode="r")
    return x, y

@register_keras_serializable(package="custom")
class AddScalar(layers.Layer):
    def __init__(self, value=0.0, **kwargs):
        super().__init__(**kwargs)
        self.value = float(value)
    def call(self, x):
        return x + tf.cast(self.value, x.dtype)
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"value": self.value})
        return cfg

@register_keras_serializable(package="custom")
class MatchDType(layers.Layer):
    """Cast first tensor to the dtype of the second tensor."""
    def call(self, inputs):
        t, ref = inputs
        return tf.cast(t, ref.dtype)

@register_keras_serializable(package="custom")
class ScalarGate(layers.Layer):
    def __init__(self, init=0.1, minv=0.0, maxv=0.5, **kwargs):
        super().__init__(**kwargs)
        self.init = float(init); self.minv=float(minv); self.maxv=float(maxv)

    def build(self, _):
        init_clip = np.clip(self.init, 1e-4, 1-1e-4)
        logit = float(np.log(init_clip/(1-init_clip)))
        self.logit = self.add_weight(
            name="logit",                 # <- keyword, not positional
            shape=(),                     # scalar param
            initializer=tf.keras.initializers.Constant(logit),
            trainable=True,
            dtype=tf.float32,             # explicit dtype is safest
        )

    def call(self, x):
        s = tf.cast(tf.sigmoid(self.logit) * (self.maxv - self.minv) + self.minv, x.dtype)
        return x * s

@register_keras_serializable(package="custom")
class GroupNorm(layers.Layer):
    def __init__(self, groups=8, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.groups = int(groups)
        self.epsilon = float(epsilon)
        # will be set in build()
        self._c = None
        self._g = None
        self._cg = None

    def build(self, input_shape):
        c = int(input_shape[-1])
        self._c = c
        # choose g <= groups that divides c (static)
        g = min(self.groups, c)
        g = max(gcd(g, c), 1)
        self._g = g
        self._cg = c // g

        self.gamma = self.add_weight(name="gamma", shape=(c,), initializer="ones", trainable=True)
        self.beta  = self.add_weight(name="beta",  shape=(c,), initializer="zeros", trainable=True)

    def call(self, x):
        # x: [N, D, H, W, C]
        x_dtype = x.dtype
        n, d, h, w = tf.unstack(tf.shape(x))[:4]
        g  = self._g
        cg = self._cg
    
        # [N*D*H*W, G, C//G]
        xg = tf.reshape(x, [n * d * h * w, g, cg])
    
        # compute in the same dtype to save memory
        mean = tf.reduce_mean(xg, axis=[0, 2], keepdims=True)                 # [1, G, 1]
        var  = tf.reduce_mean(tf.square(xg - mean), axis=[0, 2], keepdims=True)
        xg_norm = (xg - mean) / tf.sqrt(var + tf.cast(self.epsilon, xg.dtype))  # [N*D*H*W, G, C//G]
    
        # back to [N, D, H, W, C]
        x_norm = tf.reshape(xg_norm, [n, d, h, w, g * cg])
    
        gamma = tf.cast(self.gamma, x_dtype)
        beta  = tf.cast(self.beta,  x_dtype)
        return x_norm * gamma + beta

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"groups": self.groups, "epsilon": self.epsilon})
        return cfg

def load_params(p="param.json"):
    with open(p, "r") as f:
        return json.load(f)

def ensure_dir(p):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

# -----------------------
# Data pipeline (memmap + tf.data)
# -----------------------
def dataset_from_indices_bin(x_mem, y_mem, tab_mem, idx, batch_size, shuffle=False, repeat=False):
    """tf.data pipeline yielding ((X, tab), y) for binary classification."""
    x_shape   = tuple(x_mem.shape[1:])
    tab_shape = (int(tab_mem.shape[1]),)
    idx = np.asarray(idx, dtype=np.int64)

    def gen():
        for i in idx:
            X   = x_mem[i].astype(np.float32)
            tab = tab_mem[i].astype(np.float32)
            y   = np.array([float(y_mem[i])], dtype=np.float32)  # shape (1,) for Keras
            yield ((X, tab), y)

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            (tf.TensorSpec(shape=x_shape, dtype=tf.float32),
             tf.TensorSpec(shape=tab_shape, dtype=tf.float32)),
            tf.TensorSpec(shape=(1,), dtype=tf.float32),
        ),
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(idx), 2048), reshuffle_each_iteration=True)
    if repeat:
        ds = ds.repeat()
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# -----------------------
# Model: 3D ResNet
# -----------------------
def GN(groups_gn):
    return lambda: GroupNorm(groups=groups_gn)

def conv_bn_act(x, filters, k=3, s=1, groups_gn=8):
    x = layers.Conv3D(filters, k, strides=s, padding="same", use_bias=False,
                      kernel_initializer="he_normal")(x)
    x = GN(groups_gn)()(x)
    return layers.Activation("relu")(x)

def residual_block(x, filters, s=1, bottleneck=False, groups_gn=8, drop=0.0, cond=None, name=None):
    """If cond is provided, apply FiLM on the main path after the last GN."""
    name = name or "res"
    shortcut = x
    if bottleneck:
        out = conv_bn_act(x, filters, k=1, s=s, groups_gn=groups_gn)
        out = conv_bn_act(out, filters, k=3, s=1, groups_gn=groups_gn)
        out = layers.Conv3D(filters*4, 1, use_bias=False, padding="same",
                            kernel_initializer="he_normal", name=f"{name}_conv3")(out)
        out = GN(groups_gn)( )(out)
        # --- FiLM on main path ---
        if cond is not None:
            gamma, beta = film_params(cond, filters*4, name=f"{name}_film")
            out = apply_film(out, gamma, beta, name=f"{name}_film_apply")
        if s != 1 or shortcut.shape[-1] != filters*4:
            shortcut = layers.Conv3D(filters*4, 1, strides=s, use_bias=False, padding="same",
                                     kernel_initializer="he_normal", name=f"{name}_proj")(shortcut)
            shortcut = GN(groups_gn)( )(shortcut)
        out = layers.Add(name=f"{name}_add")([out, shortcut])
        out = layers.Activation("relu", name=f"{name}_relu")(out)
        if drop and drop > 0:
            out = layers.SpatialDropout3D(drop, name=f"{name}_sd")(out)
        return out
    else:
        out = conv_bn_act(x, filters, k=3, s=s, groups_gn=groups_gn)
        out = layers.Conv3D(filters, 3, padding="same", use_bias=False,
                            kernel_initializer="he_normal", name=f"{name}_conv2")(out)
        out = GN(groups_gn)( )(out)
        # --- FiLM on main path ---
        if cond is not None:
            gamma, beta = film_params(cond, filters, name=f"{name}_film")
            out = apply_film(out, gamma, beta, name=f"{name}_film_apply")
        if s != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv3D(filters, 1, strides=s, use_bias=False, padding="same",
                                     kernel_initializer="he_normal", name=f"{name}_proj")(shortcut)
            shortcut = GN(groups_gn)( )(shortcut)
        out = layers.Add(name=f"{name}_add")([out, shortcut])
        out = layers.Activation("relu", name=f"{name}_relu")(out)
        if drop and drop > 0:
            out = layers.SpatialDropout3D(drop, name=f"{name}_sd")(out)
        return out

def build_resnet3d_film(
    input_shape, tab_dim, initial_filters=32, block_layers=(2,2,2,2),
    bottleneck=False, dropout_rate=0.1, groups_gn=8,
    tab_norm_mean=None, tab_norm_var=None,
    film_hidden=64, film_layers=2, film_dropout=0.10, film_activation="relu",
    film_apply_stages="all"  # "all" or "late"
):
    img_in = keras.Input(shape=input_shape, name="img")
    tab_in = keras.Input(shape=(tab_dim,), name="tab")

    norm = layers.Normalization(mean=tab_norm_mean, variance=tab_norm_var, dtype="float32", name="tab_norm")
    t = norm(tab_in)
    for li in range(int(film_layers)):
        t = layers.Dense(int(film_hidden), activation=film_activation, name=f"tab_fc{li+1}")(t)
        if film_dropout and film_dropout > 0:
            t = layers.Dropout(float(film_dropout), name=f"tab_drop{li+1}")(t)

    x = conv_bn_act(img_in, initial_filters, k=7, s=2, groups_gn=groups_gn)
    x = layers.MaxPool3D(pool_size=3, strides=2, padding="same")(x)

    # choose where to apply FiLM
    apply_on_stage = lambda bi: True if film_apply_stages=="all" else (bi >= len(block_layers)-2)

    filters = initial_filters
    for bi, n_blocks in enumerate(block_layers):
        s = 1 if bi == 0 else 2
        cond = t if apply_on_stage(bi) else None
        x = residual_block(x, filters, s=s, bottleneck=bottleneck, groups_gn=groups_gn,
                           drop=dropout_rate, cond=cond, name=f"b{bi}_0")
        for bj in range(1, n_blocks):
            x = residual_block(x, filters, s=1, bottleneck=bottleneck, groups_gn=groups_gn,
                               drop=dropout_rate, cond=cond, name=f"b{bi}_{bj}")
        filters *= (4 if bottleneck else 2)

    x = layers.GlobalAveragePooling3D(name="gap")(x)
    x = layers.Dropout(dropout_rate, name="head_drop")(x)
    out = layers.Dense(1, activation="sigmoid", dtype="float32", name="prob")(x)
    return keras.Model([img_in, tab_in], out, name="ResNet3D_FiLM")

def film_params(cond, channels, name):
    h = layers.Dense(64, activation="relu", name=f"{name}_mlp1")(cond)
    h = layers.Dropout(0.1, name=f"{name}_drop")(h)

    dgamma = layers.Dense(channels, kernel_initializer="zeros", bias_initializer="zeros",
                          name=f"{name}_dgamma")(h)
    dbeta  = layers.Dense(channels, kernel_initializer="zeros", bias_initializer="zeros",
                          name=f"{name}_dbeta")(h)

    dgamma = layers.Activation("tanh", name=f"{name}_gamma_tanh")(dgamma)
    dbeta  = layers.Activation("tanh", name=f"{name}_beta_tanh")(dbeta)

    dgamma = ScalarGate(init=0.1, minv=0.0, maxv=0.5, name=f"{name}_gamma_gate")(dgamma)
    dbeta  = ScalarGate(init=0.1, minv=0.0, maxv=0.5, name=f"{name}_beta_gate")(dbeta)

    # Make gamma = 1 + dgamma without tf.*:
    gamma = AddScalar(1.0, name=f"{name}_gamma_bias")(dgamma)
    beta  = dbeta
    return gamma, beta

def apply_film(x, gamma, beta, name):
    """Broadcast FiLM params across (D,H,W) and apply: y = gamma * x + beta."""
    c = int(x.shape[-1])
    g = layers.Reshape((1,1,1,c), name=f"{name}_g_reshape")(gamma)
    b = layers.Reshape((1,1,1,c), name=f"{name}_b_reshape")(beta)
    # keep dtype stable under mixed precision
    g = MatchDType(name=f"{name}_g_cast")([g, x])
    b = MatchDType(name=f"{name}_b_cast")([b, x])
    xg = layers.Multiply(name=f"{name}_mul")([x, g])
    return layers.Add(name=f"{name}_add")([xg, b])

def compute_class_weights(y):
    y = np.asarray(y).astype(int)
    counts = np.bincount(y, minlength=2).astype(float)
    total = counts.sum()
    w = total / (2.0 * np.maximum(counts, 1.0))
    return {0: float(w[0]), 1: float(w[1])}

# -----------------------
# Main training/testing
# -----------------------
def main():
    P = load_params()
    DO_TRAIN = bool(P.get("TRAIN", True))
    DO_TEST  = bool(P.get("TEST", True))
    ROOT = P["data"]["root"]
    OUT  = P["output"]["workdir"]
    ensure_dir(OUT)

    if P["training"]["mixed_precision"]:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    # Memmaps
    Xm, Ym = make_memmaps_bin(
        ROOT, P["data"]["x_train"], P["data"]["y_train"]
    )

    # Tabular memmaps
    tab_train_path = os.path.join(ROOT, P["data"].get("tab_train", "tab_train.npy"))
    TABm = np.load(tab_train_path, mmap_mode="r")  # shape: (n_total, 3)




    print("\n=== DATA SANITY CHECK ===")

    def check_array(name, arr):
        arr_finite = np.isfinite(arr)
        if not arr_finite.all():
            n_bad = np.size(arr) - np.count_nonzero(arr_finite)
            print(f"[WARN] {name}: {n_bad} / {arr.size} NaN or Inf values")
            bad_idx = np.where(~arr_finite.any(axis=tuple(range(1, arr.ndim))))[0] if arr.ndim > 1 else np.where(~arr_finite)[0]
            print(f"        Example bad indices: {bad_idx[:10].tolist()}")
        else:
            print(f"[OK] {name}: all finite")

    check_array("X_train", Xm[:100])   # sample a few to avoid loading everything
    check_array("tab_train", TABm)
    check_array("y_train", Ym)

    # Also check label range
    yvals = np.array(Ym, dtype=float)
    uniq = np.unique(yvals[np.isfinite(yvals)])
    print(f"[INFO] Unique finite y_train values: {uniq[:20]}")
    if not np.all(np.isin(uniq, [0.0, 1.0])):
        print("[WARN] y_train has values outside {0,1} – may need binarization.")
    print("==========================\n")






    tab_dim = int(TABm.shape[1])
    print("Tabular dim:", tab_dim)

    y_full = np.array(Ym, dtype=float)  # may include NaNs for unlabeled
    labeled_mask = np.isfinite(y_full)
    
    idx_all = np.arange(y_full.shape[0])
    idx_lab = idx_all[labeled_mask]
    y_lab   = y_full[labeled_mask].astype(int)
    
    # simple stratification by binary label
    strat_y = y_lab

    # --- group ids for leakage-free CV (subject-level) ---
    IDs_PATH = os.path.join(ROOT, "ids_train.npy")
    if not os.path.exists(IDs_PATH):
        raise FileNotFoundError("ids_train.npy not found; needed for group-aware CV.")
    ids_all = np.load(IDs_PATH, allow_pickle=True)
    _, groups_all = np.unique(ids_all, return_inverse=True)
    groups_lab = groups_all[labeled_mask]

    input_shape = tuple(Xm.shape[1:])  # (70,70,70,4)

    # =================== TRAIN ===================
    if DO_TRAIN:
        cv = StratifiedGroupKFold(
            n_splits=P["cv"]["n_splits"],
            shuffle=P["cv"]["shuffle"],
            random_state=P["cv"]["random_state"]
        )

        oof_scores = np.zeros(len(idx_lab), dtype=np.float32)
        for fold, (tr, va) in enumerate(cv.split(np.arange(len(idx_lab)), strat_y, groups=groups_lab), start=1):
            print(f"\n========== Fold {fold}/{P['cv']['n_splits']} ==========")
            idx_tr = idx_lab[tr]
            idx_va = idx_lab[va]

            # leakage check
            gids_tr = set(groups_lab[tr].tolist())
            gids_va = set(groups_lab[va].tolist())
            overlap = gids_tr & gids_va
            assert len(overlap) == 0, f"Group leakage detected! {sorted(list(overlap))[:10]}"

            bs = P["training"].get("batch_size", 1)

            ds_tr = dataset_from_indices_bin(Xm, y_full, TABm, idx_tr,
                                             batch_size=bs, shuffle=True, repeat=True)
            ds_va = dataset_from_indices_bin(Xm, y_full, TABm, idx_va,
                                             batch_size=bs, shuffle=False, repeat=False)

            # 1) pick one row per subject among training indices
            grp_tr = groups_all[idx_tr]                       # group id per row in this fold's train
            _, first_pos = np.unique(grp_tr, return_index=True)
            idx_tr_base = idx_tr[np.sort(first_pos)]          # de-duplicated by subject

            mu  = TABm[idx_tr_base].mean(axis=0).astype("float32")
            var = TABm[idx_tr_base].var(axis=0).astype("float32")
            var[var < 1e-12] = 1.0  # keep divide-by-std stable

            FILM = P.get("film", {})
            M = build_resnet3d_film(
                input_shape=input_shape,
                tab_dim=tab_dim,
                initial_filters=P["model"]["initial_filters"],
                block_layers=tuple(P["model"]["block_layers"]),
                bottleneck=P["model"]["bottleneck"],
                dropout_rate=P["model"]["dropout_rate"],
                groups_gn=P["model"]["groups_gn"],
                tab_norm_mean=mu, tab_norm_var=var,
                film_hidden= int(FILM.get("mlp_hidden", 64)),
                film_layers= int(FILM.get("mlp_layers", 2)),
                film_dropout=float(FILM.get("mlp_dropout", 0.10)),
                film_activation=     FILM.get("mlp_activation", "relu"),
                film_apply_stages=   FILM.get("apply_stages", "all"),
            )
            
            fold_dir = os.path.join(OUT, f"fold_{fold}")
            ensure_dir(fold_dir)
            with open(os.path.join(fold_dir, "tab_norm.json"), "w") as f:
                json.dump({"mean": mu.tolist(), "var": var.tolist()}, f, indent=2)

            opt_base = tf.keras.optimizers.AdamW(
                learning_rate=float(P["training"]["base_lr"]),
                weight_decay=float(P["training"]["weight_decay"]),
                clipnorm=1.0
            )
            opt = (tf.keras.mixed_precision.LossScaleOptimizer(opt_base)
                   if P["training"]["mixed_precision"] else opt_base)
           
            loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            metrics = [
                tf.keras.metrics.AUC(curve="ROC", name="auc"),
                tf.keras.metrics.AUC(curve="PR",  name="auprc"),
                tf.keras.metrics.BinaryAccuracy(name="acc", threshold=0.5),
            ]

            M.compile(optimizer=opt, loss=loss_fn, metrics=metrics)

            # ---- Callbacks ----
            fold_dir = os.path.join(OUT, f"fold_{fold}")
            ensure_dir(fold_dir)
            
            ckpt = tf.keras.callbacks.ModelCheckpoint(
                os.path.join(fold_dir, "best_model.keras"),
                monitor="val_auc", mode="max",
                save_best_only=True, save_weights_only=False
            )
            es = tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max",
                patience=int(P["training"]["early_stopping_patience"]),
                restore_best_weights=True
            )
            rlrop = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_auc", mode="max",
                factor=0.5,
                patience=int(P["training"]["lr_plateau_patience"]),
                min_lr=1e-6
            )
            
            # ---- Optional class weights for imbalance ----
            cw = compute_class_weights(y_lab[tr]) if P["training"].get("use_class_weights", True) else None
            
            # ---- Train ----
            history = M.fit(
                ds_tr,
                validation_data=ds_va,
                epochs=int(P["training"]["epochs"]),
                steps_per_epoch=int(math.ceil(len(idx_tr) / bs)),
                class_weight=cw,
                callbacks=[ckpt, es, rlrop],
                verbose=2,
            )
            
            # Save history
            with open(os.path.join(fold_dir, "history.json"), "w") as f:
                json.dump({k: [float(v) for v in history.history.get(k, [])]
                           for k in history.history}, f, indent=2)
           
            best_p = os.path.join(fold_dir, "best_model.keras")
            if os.path.exists(best_p):
                keras.config.enable_unsafe_deserialization()
                M = keras.models.load_model(
                    best_p,
                    compile=False,
                    custom_objects={
                        "GroupNorm": GroupNorm,
                        "ScalarGate": ScalarGate,
                    },
                )
            else:
                print(f"[fold {fold}] WARNING: best_model.keras not found; using last weights.")

            # Collect fold-val probabilities
            probs_va = M.predict(ds_va, verbose=0).reshape(-1)
            oof_scores[va] = probs_va

            # Save fold artifacts
            fold_dir = os.path.join(OUT, f"fold_{fold}")
            ensure_dir(fold_dir)
            
            if P["output"]["save_fold_models"]:
                M.save(os.path.join(fold_dir, "model.keras"))

            del ds_va, M

            tf.keras.backend.clear_session()
            gc.collect()

        y_lab_all = y_full[labeled_mask].astype(int)
        oof_auc   = roc_auc_score(y_lab_all, oof_scores)
        oof_auprc = average_precision_score(y_lab_all, oof_scores)
        # --- pick optimal threshold on OOF (change mode to "f1" if preferred) ---
        best_thr, best_score = _best_threshold(y_lab_all, oof_scores, mode="youden")
        print(f"\n[oof] roc-auc: {oof_auc:.5f} | pr-auc: {oof_auprc:.5f} | best_thr(youden): {best_thr:.4f} (score={best_score:.4f})")

        np.save(os.path.join(OUT, "oof_probs.npy"),  oof_scores)
        np.save(os.path.join(OUT, "oof_labels.npy"), y_lab_all)
        with open(os.path.join(OUT, "run_summary.json"), "w") as f:
            json.dump({
                "oof_roc_auc": float(oof_auc),
                "oof_pr_auc":  float(oof_auprc),
                "n_samples_labeled": int(len(idx_lab))
            }, f, indent=2)
        with open(os.path.join(OUT, "optimal_threshold.json"), "w") as f:
            json.dump({
                "mode": "youden",
                "threshold": float(best_thr),
                "score": float(best_score)
            }, f, indent=2)

    x_test_path = os.path.join(ROOT, P["data"]["x_test"])
    if DO_TEST and os.path.exists(x_test_path):
        Xt = np.load(x_test_path, mmap_mode="r")
        tab_te_path = os.path.join(ROOT, P["data"].get("tab_test", "tab_test.npy"))
        TABt = np.load(tab_te_path, mmap_mode="r") if os.path.exists(tab_te_path) else None
        if TABt is None:
            raise FileNotFoundError("tab_test.npy not found; FiLM model needs tabular test features.")

        y_test_path = os.path.join(ROOT, P["data"].get("y_test", "y_test.npy"))
        Yt = np.load(y_test_path, mmap_mode="r") if os.path.exists(y_test_path) else None

        probs_accum = np.zeros((Xt.shape[0],), dtype=np.float32)
        n_models = 0
        used_models = []

        for fold in range(1, P["cv"]["n_splits"] + 1):
            fold_dir = os.path.join(OUT, f"fold_{fold}")
            best_p = os.path.join(fold_dir, "best_model.keras")
            last_p = os.path.join(fold_dir, "model.keras")
            load_p = best_p if os.path.exists(best_p) else last_p
            if not os.path.exists(load_p):
                continue

            used_models.append(os.path.relpath(load_p, OUT))

            keras.config.enable_unsafe_deserialization()
            Mf = keras.models.load_model(
                load_p,
                compile=False,
                custom_objects={
                    "GroupNorm": GroupNorm,
                    "ScalarGate": ScalarGate,
                    "AddScalar": AddScalar,
                    "MatchDType": MatchDType,
                },
            )
            # dataset for inference
            dummy_y = np.zeros(Xt.shape[0], dtype=np.float32)
            ds_te = dataset_from_indices_bin(
                Xt, dummy_y, TABt, np.arange(Xt.shape[0]),
                batch_size=P["training"].get("batch_size", 1)
            )

            probs = Mf.predict(ds_te, verbose=0).reshape(-1)
            probs_accum += probs
            n_models += 1

            del ds_te, Mf
            tf.keras.backend.clear_session()
            gc.collect()

        # ---- After ensembling ----
        if n_models == 0:
            raise RuntimeError("No fold models found to run TEST.")

        test_probs = probs_accum / n_models
        np.save(os.path.join(OUT, "test_probs.npy"), test_probs)
        tab_test  = TABt
        test_prob = test_probs

        with open(os.path.join(OUT, "test_used_models.json"), "w") as f:
            json.dump(used_models, f, indent=2)

        # Optional test metrics if y_test exists
        if Yt is not None:
            y_true = np.array(Yt, float)
            mask = np.isfinite(y_true)
            if np.any(mask):
                auc   = roc_auc_score(y_true[mask].astype(int), test_probs[mask])
                auprc = average_precision_score(y_true[mask].astype(int), test_probs[mask])
                acc05 = ((test_probs[mask] >= 0.5).astype(int) == y_true[mask].astype(int)).mean()
                # load tuned threshold if available
                thr_path = os.path.join(OUT, "optimal_threshold.json")
                if os.path.exists(thr_path):
                    with open(thr_path, "r") as f:
                        thr = float(json.load(f).get("threshold", 0.5))
                else:
                    thr = 0.5
                acc_tuned = ((test_probs[mask] >= thr).astype(int) == y_true[mask].astype(int)).mean()
                f1_tuned  = f1_score(y_true[mask].astype(int), (test_probs[mask] >= thr).astype(int))
                with open(os.path.join(OUT, "test_summary.json"), "w") as f:
                    json.dump({
                        "roc_auc": float(auc),
                        "pr_auc": float(auprc),
                        "acc@0.5": float(acc05),
                        "acc@tuned": float(acc_tuned),
                        "f1@tuned": float(f1_tuned),
                        "thr_tuned": float(thr)
                    }, f, indent=2)
                print(f"[TEST] ROC-AUC: {auc:.5f} | PR-AUC: {auprc:.5f} | Acc@0.5: {acc05:.5f} | Acc@tuned({thr:.3f}): {acc_tuned:.5f} | F1@tuned: {f1_tuned:.5f}")
            else:
                print("\n[TEST] Saved probabilities; y_test has no finite labels.")
        else:
            print("\n[TEST] Saved probabilities; no y_test found.")


        # ========= MIXED IG: aggregate on TEST =========
        if 1:
            try:
                model_for_ig = None
                for fold in range(1, P["cv"]["n_splits"] + 1):
                    for name in ("best_model.keras", "model.keras"):
                        cand = os.path.join(OUT, f"fold_{fold}", name)
                        if os.path.exists(cand):
                            model_for_ig = keras.models.load_model(
                                cand,
                                compile=False,
                                custom_objects={
                                    "GroupNorm": GroupNorm,
                                    "ScalarGate": ScalarGate,
                                    "AddScalar": AddScalar,
                                    "MatchDType": MatchDType,
                                },
                            )
                            print(f"[MIXED-IG] Using model: {cand}")
                            break
                    if model_for_ig is not None:
                        break
            
                if model_for_ig is None:
                    print("[MIXED-IG] No model to run IG; skipping.")
                else:
                    # Load test arrays (assumes already in memory as Xt, yt, and tab_test)
                    idx_te = np.arange(Xt.shape[0], dtype=int)
                    TAB_NAMES = ("Age", "Sex", "EOR")
                    TAB_BASELINE = TABt.mean(axis=0).astype(np.float32)  # dataset-mean baseline

                    usage = aggregate_usage_mixed(
                        model_for_ig, Xt, tab_test, idx_te, probs=test_prob,
                        steps=32, smooth_repeats=4, noise_std=0.02,
                        post_sigma=1.0, verbose_every=50,
                        tab_names=TAB_NAMES, tab_baseline=TAB_BASELINE
                    )
            
                    # Print summary
                    ch_mean = np.round(usage["image_channels"]["mean"], 4).tolist()
                    ch_std  = np.round(usage["image_channels"]["std"], 4).tolist()
                    tb_mean = np.round(usage["tabular"]["mean"], 4).tolist()
                    tb_std  = np.round(usage["tabular"]["std"], 4).tolist()
                    md_mean = np.round(usage["modality_share"]["mean"], 4).tolist()
                    md_std  = np.round(usage["modality_share"]["std"], 4).tolist()
                    print(f"[MIXED-IG][TEST] image ch mean={ch_mean} std={ch_std} (C={Xt.shape[-1]})")
                    print(f"[MIXED-IG][TEST] tabular  mean={tb_mean} std={tb_std} names={usage['tabular']['names']}")
                    print(f"[MIXED-IG][TEST] modality share mean(img,tab)={md_mean} std={md_std}")
            
                    # Save JSON summaries
                    with open(os.path.join(OUT, "mixed_usage_summary.json"), "w") as f:
                        json.dump(usage, f, indent=2)
            
                    # Save per-sample CSVs (needs ids_test for alignment)
                    IDS_TEST_PATH = os.path.join(ROOT, "ids_test.npy")
                    ids_test = np.load(IDS_TEST_PATH, allow_pickle=True) if os.path.exists(IDS_TEST_PATH) \
                               else np.array([f"sample_{i}" for i in range(Xt.shape[0])])
            
                    # 1) image channel weights
                    ch_csv = os.path.join(OUT, "variant_channel_usage_test.csv")
                    with open(ch_csv, "w", newline="") as f:
                        w = csv.writer(f)
                        header = ["sid"] + [f"ch{i}" for i in range(Xt.shape[-1])]
                        w.writerow(header)
                        for sid, row in zip(ids_test, usage["image_channels"]["per_sample"]):
                            w.writerow([sid] + [f"{v:.6f}" for v in row])
            
                    # 2) tabular feature weights
                    tb_csv = os.path.join(OUT, "tabular_importance_test.csv")
                    with open(tb_csv, "w", newline="") as f:
                        w = csv.writer(f)
                        header = ["sid"] + list(usage["tabular"]["names"])
                        w.writerow(header)
                        for sid, row in zip(ids_test, usage["tabular"]["per_sample"]):
                            w.writerow([sid] + [f"{v:.6f}" for v in row])
            
                    # 3) modality share (image vs tabular)
                    md_csv = os.path.join(OUT, "modality_share_test.csv")
                    with open(md_csv, "w", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["sid", "img_share", "tab_share"])
                        for sid, row in zip(ids_test, usage["modality_share"]["per_sample"]):
                            w.writerow([sid, f"{row[0]:.6f}", f"{row[1]:.6f}"])
            
                    print(f"[MIXED-IG] Saved CSVs:\n  {ch_csv}\n  {tb_csv}\n  {md_csv}")
                    # 4) combined 7-way attribution
                    comb_csv = os.path.join(OUT, "combined_attribution_7_test.csv")
                    with open(comb_csv, "w", newline="") as f:
                        w = csv.writer(f)
                        header = ["sid"] + usage["combined_7"]["names"]
                        w.writerow(header)
                        for sid, row in zip(ids_test, usage["combined_7"]["per_sample"]):
                            w.writerow([sid] + [f"{v:.6f}" for v in row])
                    print(f"[MIXED-IG] Saved 7-way attribution CSV:\n  {comb_csv}")
 
                    # Save also one pretty heatmap + that subject's tabular IG
                    if 1:
                        save_one_test_heatmap_mixed(
                            model_for_ig, Xt, tab_test, test_prob, OUT, ROOT, P,
                            pick_class=1, sigma=1.2, bg_factor=0.4,
                            tab_names=TAB_NAMES
                        )
            
                    # cleanup
                    del model_for_ig
                    tf.keras.backend.clear_session()
                    gc.collect()
            
            except Exception as e:
                print(f"[MIXED-IG] Error during mixed IG analysis: {e}")

    elif not os.path.exists(x_test_path):
        print("\n[TEST] Skipped: X_test file not found.")

    else:
        print("\n[TEST] Skipped.")

if __name__ == "__main__":
    main()


