#!/usr/bin/env python3
import os, json, math, pathlib
import numpy as np
import gc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    accuracy_score, balanced_accuracy_score, f1_score
)
from collections import Counter
from tensorflow.keras.utils import register_keras_serializable
from math import gcd
import nibabel as nib
from nibabel.processing import resample_from_to
from nibabel.affines import voxel_sizes as _voxel_sizes
from scipy.ndimage import gaussian_filter


# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# -----------------------
# Utilities / Params
# -----------------------
def _ensure_5d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3:
        x = x[..., None]
    assert x.ndim == 4, f"Expected (D,H,W,C), got {x.shape}"
    return x[None, ...]  # (1,D,H,W,C)

def _ig_single(model, vol_4d, target_class, m_steps=64, n_smooth=4, noise_std=0.05):
    """
    Integrated Gradients on a single (D,H,W,C) volume for target_class.
    Returns (D,H,W,C) IG attribution (same shape as input).
    """
    x_np = _ensure_5d(vol_4d)  # (1,D,H,W,C)
    x_tf = tf.convert_to_tensor(x_np)
    baseline = tf.zeros_like(x_tf)
    alphas = tf.linspace(0.0, 1.0, m_steps)

    repeats = max(1, int(n_smooth))
    ig_total = np.zeros_like(x_np, dtype=np.float32)

    for _ in range(repeats):
        if n_smooth > 0:
            x_use_np = x_np.copy()
            std = float(x_use_np.std())
            if std > 0:
                x_use_np += np.random.normal(0.0, noise_std * std, size=x_use_np.shape).astype(np.float32)
            x_use = tf.convert_to_tensor(x_use_np)
        else:
            x_use = x_tf

        delta = x_use - baseline
        grads_accum = tf.zeros_like(x_use)

        for a in tf.unstack(alphas):
            x_step = baseline + a * delta
            with tf.GradientTape() as tape:
                tape.watch(x_step)
                y = model(x_step, training=False)              # (1,K)
                y_tar = y[..., target_class:target_class+1]    # (1,1)
            grads = tape.gradient(y_tar, x_step)
            grads_accum += grads

        avg_grads = grads_accum / float(m_steps)
        ig = (x_use - baseline) * avg_grads                   # (1,D,H,W,C)
        ig_total += ig.numpy()

    ig_total = ig_total / float(repeats)
    return ig_total[0]  # (D,H,W,C)

def _channel_weights_from_ig(ig_vol: np.ndarray, eps: float = 1e-8):
    """
    ig_vol: (D,H,W,C) attributions. Returns vector (C,) with abs-sum per channel,
    normalized to sum=1 (coeffs that add to 1).
    """
    ch = np.sum(np.abs(ig_vol), axis=(0, 1, 2)).astype(np.float32)  # (C,)
    s = float(np.sum(ch))
    if s < eps:
        # fallback to uniform if all zeros
        return np.ones_like(ch, dtype=np.float32) / max(1, ch.size)
    return ch / s

def _aggregate_channel_usage(model, X_mem, indices, probs=None,
                             steps=32, smooth_repeats=4, noise_std=0.02,
                             post_sigma=1.0, verbose_every=50):
    """
    Compute per-sample channel weights via IG and aggregate mean/std.
    - model: keras model to attribute
    - X_mem: memmap array (N,D,H,W,C)
    - indices: iterable of sample indices
    - probs: optional (N,K) to pick target_class = argmax(probs[i]); if None, we use model prediction
    Returns dict: {"mean": list, "std": list, "n": int, "per_sample": [[...], ...]}
    """
    coeffs = []
    for c, i in enumerate(indices):
        vol = np.array(X_mem[i], dtype=np.float32)
        if probs is not None:
            target_class = int(np.argmax(probs[i]))
        else:
            p = model.predict(vol[None, ...], verbose=0)[0]
            target_class = int(np.argmax(p))

        ig = _ig_single(model, vol, target_class,
                        m_steps=steps, n_smooth=smooth_repeats, noise_std=noise_std)
        # light denoise before summing (keeps shape)
        if post_sigma and post_sigma > 0:
            ig = gaussian_filter(ig, sigma=(post_sigma, post_sigma, post_sigma, 0), mode="nearest")

        w = _channel_weights_from_ig(ig)  # (C,), sums to 1
        coeffs.append(w.tolist())
        if verbose_every and ((c+1) % verbose_every == 0):
            print(f"[VARIANTS] processed {c+1} samples...")

    coeffs = np.asarray(coeffs, dtype=np.float32)
    mean = coeffs.mean(axis=0).tolist() if coeffs.size else []
    std  = coeffs.std(axis=0).tolist() if coeffs.size else []
    return {"mean": mean, "std": std, "n": int(coeffs.shape[0]), "per_sample": coeffs.tolist()}

def _postprocess_ig(ig_vol, vol_4d, sigma=1.0, air_pct=10.0,
                    p_low=1.0, p_high=99.5, bg_factor=0.3):
    """
    Clean and normalize an Integrated-Gradients volume.
    - Removes padding/air.
    - Applies light smoothing.
    - Performs robust normalization only inside the foreground.
    - Zeroes out low-magnitude background after normalization.

    Returns (ig_norm, brainmask)
    """
    vol_4d = np.asarray(vol_4d, dtype=np.float32)
    ig_abs = np.sum(np.abs(ig_vol), axis=-1).astype(np.float32)  # (D,H,W)

    # --- foreground mask (based on input intensities) ---
    padmask = (np.max(vol_4d, axis=-1) > 0)
    mean_img = np.mean(vol_4d, axis=-1)
    if padmask.any():
        t_air = np.percentile(mean_img[padmask], air_pct)
        brainmask = padmask & (mean_img > t_air)
    else:
        brainmask = padmask

    # --- kill outside of mask immediately ---
    ig_abs[~brainmask] = 0.0

    # --- denoise a bit ---
    if sigma > 0:
        ig_abs = gaussian_filter(ig_abs, sigma=sigma, mode="nearest")

    # --- robust normalization ---
    vals = ig_abs[brainmask]
    if vals.size > 0:
        lo = np.percentile(vals, p_low)
        hi = np.percentile(vals, p_high)
        hi = max(hi, lo + 1e-6)
        ig_norm = np.clip((ig_abs - lo) / (hi - lo), 0, 1)
    else:
        ig_norm = np.zeros_like(ig_abs, dtype=np.float32)

    # --- suppress weak noise in background ---
    # compute global mean inside mask, threshold by factor
    m = ig_norm[brainmask].mean() if brainmask.any() else 0.0
    ig_norm[ig_norm < bg_factor * m] = 0.0

    return ig_norm, brainmask

def categorical_focal_loss(gamma=2.0, alpha=None, label_smoothing=0.0, eps=1e-7):
    """
    Focal loss for single-label multi-class with SOFTMAX probs.
    y_true: one-hot; y_pred: softmax probs.
    alpha: None (no class weighting), scalar, or list/1D tensor of per-class weights.
    """
    def loss_fn(y_true, y_pred):
        if label_smoothing and label_smoothing > 0:
            K = tf.cast(tf.shape(y_true)[-1], y_true.dtype)
            y_true = (1.0 - label_smoothing) * y_true + label_smoothing / K

        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        ce = -tf.math.log(y_pred)                # CE per class
        mod = tf.pow(1.0 - y_pred, gamma)        # (1-p)^gamma
        fl = mod * ce                             # focal factor
        if alpha is not None:
            a = tf.convert_to_tensor(alpha, dtype=fl.dtype)
            fl = fl * a                           # per-class weighting
        fl = fl * y_true                          # pick true class term
        return tf.reduce_sum(fl, axis=-1)         # mean later
    return loss_fn

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
        # dynamic batch/spatial dims are fine; group dims are static ints
        n, d, h, w = tf.unstack(tf.shape(x))[:4]
        x = tf.reshape(x, [n, d, h, w, self._g, self._cg])
        mean, var = tf.nn.moments(x, axes=[1, 2, 3, 5], keepdims=True)
        x = (x - mean) / tf.sqrt(var + self.epsilon)
        x = tf.reshape(x, [n, d, h, w, self._c])
        return x * self.gamma + self.beta

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"groups": self.groups, "epsilon": self.epsilon})
        return cfg

def load_params(p="param.json"):
    with open(p, "r") as f:
        return json.load(f)

def ensure_dir(p):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def remap_labels(y, old_vals):
    """Map e.g. {2,3,4} -> {0,1,2}. Returns (y_mapped, map_dict). NaNs are kept."""
    mapping = {int(v): i for i, v in enumerate(old_vals)}
    y_out = y.copy()
    mask = ~np.isnan(y_out)
    y_out[mask] = np.vectorize(lambda z: mapping[int(z)])(y_out[mask])
    return y_out.astype(float), mapping

def compute_box_for_path(src_path: str, side_mm: float = 70.0, top_pct: float = 0.01):
    """
    Reconstruct a cube crop box in *native voxel coords* for the given ASL volume.
    Strategy (matches your pad-center pipeline):
      1) Centered cube of ~side_mm in each axis (converted to voxels via affine).
      2) Enforce 'top_pct' by limiting the superior bound to z_limit = Z*(1-top_pct).
      3) Clip to image bounds while preserving the requested size as much as possible.

    Returns: (x0, x1, y0, y1, z0, z1) with 0 <= x0 < x1 <= X, etc.
    """
    img = nib.load(src_path)
    X, Y, Z = img.shape[:3]

    # Voxel sizes (mm/voxel) along each native axis
    vx, vy, vz = _voxel_sizes(img.affine)

    # Desired side in voxels per axis
    sx = int(max(1, round(side_mm / float(vx))))
    sy = int(max(1, round(side_mm / float(vy))))
    sz = int(max(1, round(side_mm / float(vz))))

    sx = min(sx, X)
    sy = min(sy, Y)
    sz = min(sz, Z)

    # Initial center = volume center
    cx, cy, cz = X // 2, Y // 2, Z // 2

    # Limit superior slices by top_pct (e.g., remove top 1% of axial slices)
    z_limit = int(round(Z * (1.0 - float(top_pct))))
    z_limit = max(sz, min(z_limit, Z))  # keep room for sz and stay within [0, Z]

    def _bounds(c, s, N, upper=None):
        """Place a window of size s centered at c inside [0, upper] (default upper=N)."""
        if upper is None:
            upper = N
        half = s // 2
        a = c - half
        b = a + s
        # Shift to fit into [0, upper]
        if a < 0:
            a = 0
            b = s
        if b > upper:
            b = upper
            a = upper - s
        # Final clamp to [0, N]
        a = max(0, min(a, N))
        b = max(a, min(b, N))
        return int(a), int(b)

    x0, x1 = _bounds(cx, sx, X)
    y0, y1 = _bounds(cy, sy, Y)
    z0, z1 = _bounds(cz, sz, Z, upper=z_limit)

    return (x0, x1, y0, y1, z0, z1)



def compute_class_weights(y_mapped, n_classes):
    mask = ~np.isnan(y_mapped)
    yv = y_mapped[mask].astype(int)
    cnt = Counter(yv.tolist())
    total = sum(cnt.values())
    # inverse frequency
    w = {c: total / (n_classes * cnt.get(c, 1)) for c in range(n_classes)}
    return w

def summarize_metrics(prefix, y_true, y_prob):
    n_classes = y_prob.shape[1]
    # AUC (threshold-free)
    y_true_oh = keras.utils.to_categorical(y_true, n_classes)
    auc = roc_auc_score(y_true_oh, y_prob, average="macro", multi_class="ovr")
    # Argmax baseline
    y_pred = np.argmax(y_prob, axis=1)
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    print(f"\n[{prefix}] Macro OvR AUC: {auc:.5f} | Acc: {acc:.5f} | BalAcc: {bacc:.5f} | F1-macro: {f1m:.5f}")
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    return {"auc_macro_ovr": float(auc), "acc": float(acc), "bal_acc": float(bacc), "f1_macro": float(f1m)}

# -----------------------
# Data pipeline (memmap + tf.data)
# -----------------------
def make_memmaps(root, x_name, y_name=None):
    x = np.load(os.path.join(root, x_name), mmap_mode="r")
    y = None
    if y_name and os.path.exists(os.path.join(root, y_name)):
        y = np.load(os.path.join(root, y_name), mmap_mode="r")
    return x, y

def dataset_from_indices(x_mem, y_mem, idx, batch_size, shuffle=False, repeat=False):
    """Create a tf.data pipeline that fetches slices from memmap by index."""
    x_shape = tuple(x_mem.shape[1:])
    has_y = y_mem is not None
    idx = np.asarray(idx, dtype=np.int64)

    def gen():
        for i in idx:
            X = x_mem[i]
            if has_y:
                y = y_mem[i]
                yield (X.astype(np.float32), int(y))
            else:
                yield (X.astype(np.float32), 0)

    output_types = (tf.float32, tf.int32)
    output_shapes = (tf.TensorShape(x_shape), tf.TensorShape([]))
    ds = tf.data.Dataset.from_generator(gen, output_types=output_types, output_shapes=output_shapes)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(idx), 2048), reshuffle_each_iteration=True)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

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

def residual_block(x, filters, s=1, bottleneck=False, groups_gn=8, drop=0.0):
    shortcut = x
    if bottleneck:
        out = conv_bn_act(x, filters, k=1, s=s, groups_gn=groups_gn)
        out = conv_bn_act(out, filters, k=3, s=1, groups_gn=groups_gn)
        out = layers.Conv3D(filters*4, 1, use_bias=False, padding="same")(out)
        out = GN(groups_gn)( )(out)
        if s != 1 or shortcut.shape[-1] != filters*4:
            shortcut = layers.Conv3D(filters*4, 1, strides=s, use_bias=False, padding="same")(shortcut)
            shortcut = GN(groups_gn)( )(shortcut)
    else:
        out = conv_bn_act(x, filters, k=3, s=s, groups_gn=groups_gn)
        out = layers.Conv3D(filters, 3, padding="same", use_bias=False)(out)
        out = GN(groups_gn)( )(out)
        if s != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv3D(filters, 1, strides=s, use_bias=False, padding="same")(shortcut)
            shortcut = GN(groups_gn)( )(shortcut)
    out = layers.Add()([out, shortcut])
    out = layers.Activation("relu")(out)
    if drop and drop > 0:
        out = layers.SpatialDropout3D(drop)(out)
    return out

def build_resnet3d(input_shape, num_classes, initial_filters=32, block_layers=(2,2,2,2),
                   bottleneck=False, dropout_rate=0.1, groups_gn=8):
    inputs = keras.Input(shape=input_shape)
    x = conv_bn_act(inputs, initial_filters, k=7, s=2, groups_gn=groups_gn)
    x = layers.MaxPool3D(pool_size=3, strides=2, padding="same")(x)
    filters = initial_filters
    for bi, n_blocks in enumerate(block_layers):
        s = 1 if bi == 0 else 2
        x = residual_block(x, filters, s=s, bottleneck=bottleneck, groups_gn=groups_gn, drop=dropout_rate)
        for _ in range(n_blocks - 1):
            x = residual_block(x, filters, s=1, bottleneck=bottleneck, groups_gn=groups_gn, drop=dropout_rate)
        filters *= (4 if bottleneck else 2)
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

# -----------------------
# Metrics & Callbacks
# -----------------------
class ValAUC_OvR(keras.callbacks.Callback):
    """Compute macro OvR AUC on the validation dataset each epoch; track best; save best."""
    def __init__(self, val_ds, patience=20, workdir=".", fold=0):
        super().__init__()
        self.val_ds = val_ds
        self.best = -np.inf
        self.wait = 0
        self.patience = patience
        self.workdir = workdir
        self.fold = fold
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        y_true, y_prob = [], []
        for xb, yb in self.val_ds:
            yp = self.model.predict(xb, verbose=0)
            y_true.append(yb.numpy())
            y_prob.append(yp)
        y_true = np.concatenate(y_true)
        y_prob = np.concatenate(y_prob)
        y_true_oh = keras.utils.to_categorical(y_true, y_prob.shape[1])
        try:
            auc = roc_auc_score(y_true_oh, y_prob, average="macro", multi_class="ovr")
        except Exception:
            auc = np.nan
        logs = logs or {}
        logs["val_auc_ovr_macro"] = auc
        print(f"\n[fold {self.fold}] epoch {epoch+1}: val_auc_ovr_macro = {auc:.5f}")

        if np.isnan(auc):
            return
        if auc > self.best:
            self.best = auc
            self.best_weights = self.model.get_weights()
            self.wait = 0
            # save best immediately
            fold_dir = os.path.join(self.workdir, f"fold_{self.fold}")
            pathlib.Path(fold_dir).mkdir(parents=True, exist_ok=True)
            self.model.save(os.path.join(fold_dir, "best_model.keras"))
            print(f"[fold {self.fold}] New best AUC={auc:.5f} -> saved best_model.keras")
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                print(f"[fold {self.fold}] Early stopping on AUC. Best={self.best:.5f}")

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            self.best_weights = None

# -----------------------
# OOF-based decision scaling
# -----------------------
def optimize_scaling_by_oof(oof_prob, y_true, rounds=2):
    """
    AUC is threshold-free, so we optimize the *decision rule* used for argmax by
    learning per-class positive scaling s_c, maximizing macro F1 on OOF.
    We predict argmax( s_c * p_c ), which is equivalent to class-specific thresholds.
    """
    K = oof_prob.shape[1]
    s = np.ones(K, dtype=np.float32)
    grid = np.linspace(0.6, 1.4, 17)  # coarse but fast

    def score(s_vec):
        y_pred = np.argmax(oof_prob * s_vec[None, :], axis=1)
        return f1_score(y_true, y_pred, average="macro")

    best = score(s)
    for _ in range(rounds):
        for c in range(K):
            best_sc = s[c]
            for g in grid:
                s_try = s.copy()
                s_try[c] = g
                cur = score(s_try)
                if cur > best:
                    best, s[c] = cur, g
    return s, float(best)

# -----------------------
# Main training/testing
# -----------------------
def main():
    P = load_params()
    loss_type = P["training"].get("loss_type", "cce")  # "cce" or "focal"
    focal_gamma = float(P["training"].get("focal_gamma", 2.0))
    focal_alpha = P["training"].get("focal_alpha", None)  # None or scalar or list
    DO_TRAIN = bool(P.get("TRAIN", True))
    DO_TEST  = bool(P.get("TEST", True))
    ROOT = P["data"]["root"]
    OUT  = P["output"]["workdir"]
    ensure_dir(OUT)

    if P["training"]["mixed_precision"]:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    # Memmaps
    Xm, ym = make_memmaps(ROOT, P["data"]["x_train"], P["data"]["y_train"])
    y_full = np.array(ym, dtype=float)
    label_vals = P["data"]["label_values"]
    y_mapped, label_map = remap_labels(y_full, label_vals)
    labeled_mask = ~np.isnan(y_mapped)
    idx_all = np.arange(y_mapped.shape[0])
    idx_lab = idx_all[labeled_mask]
    y_lab   = y_mapped[labeled_mask].astype(int)

    # --- group ids for leakage-free CV (subject-level) ---
    IDs_PATH = os.path.join(ROOT, "ids_train.npy")
    if not os.path.exists(IDs_PATH):
        raise FileNotFoundError("ids_train.npy not found; needed for group-aware CV to prevent leakage.")
    ids_all = np.load(IDs_PATH, allow_pickle=True)  # saved as object dtype
    
    # factorize subject IDs to integer groups (aligned to full array, then take labeled slice)
    _, groups_all = np.unique(ids_all, return_inverse=True)
    groups_lab = groups_all[labeled_mask]


    input_shape = tuple(Xm.shape[1:])  # (70,70,70,4)
    n_classes = P["data"]["num_classes"]

    # OOF holders (used for scaling optimization)
    oof_prob = np.zeros((len(idx_lab), n_classes), dtype=np.float32)
    oof_true = y_lab.copy()

    class_weights = None
    if P["training"]["class_weights"]:
        cw = compute_class_weights(y_mapped, n_classes)
        class_weights = {int(k): float(v) for k, v in cw.items()}
        print("Class weights:", class_weights)

    # =================== TRAIN ===================
    if DO_TRAIN:
        cv = StratifiedGroupKFold(
            n_splits=P["cv"]["n_splits"],
            shuffle=P["cv"]["shuffle"],
            random_state=P["cv"]["random_state"]
        )

        for fold, (tr, va) in enumerate(cv.split(idx_lab, y_lab, groups=groups_lab), start=1):
            print(f"\n========== Fold {fold}/{P['cv']['n_splits']} ==========")
            idx_tr = idx_lab[tr]
            idx_va = idx_lab[va]

            # ---- sanity checks for each fold ----
            y_tr_fold = y_lab[np.isin(idx_lab, idx_tr)]
            y_va_fold = y_lab[np.isin(idx_lab, idx_va)]
            
            # (1) no group leakage
            ids_tr = set(groups_lab[np.isin(idx_lab, idx_tr)].tolist())
            ids_va = set(groups_lab[np.isin(idx_lab, idx_va)].tolist())
            overlap = ids_tr & ids_va
            assert len(overlap) == 0, f"Group leakage detected! Overlapping group ids: {sorted(list(overlap))[:10]}"
            
            # (2) class coverage + distribution similarity
            def counts_props(y, n_classes):
                cnt = np.bincount(y, minlength=n_classes)
                prop = cnt / max(1, cnt.sum())
                return cnt, prop
            
            global_cnt, global_prop = counts_props(y_lab, n_classes)
            tr_cnt, tr_prop         = counts_props(y_tr_fold, n_classes)
            va_cnt, va_prop         = counts_props(y_va_fold, n_classes)
            
            missing_val = np.where(va_cnt == 0)[0].tolist()
            if missing_val:
                print(f"[WARN][fold {fold}] Missing classes in VAL: {missing_val}")
            
            # simple divergence score (max abs diff in proportions vs global)
            max_diff_tr = float(np.max(np.abs(tr_prop - global_prop)))
            max_diff_va = float(np.max(np.abs(va_prop - global_prop)))
            
            print(f"[fold {fold}] n_tr={len(y_tr_fold)} n_va={len(y_va_fold)} "
                  f"| groups_tr={len(ids_tr)} groups_va={len(ids_va)}")
            print(f"[fold {fold}] global_cnt={global_cnt.tolist()} tr_cnt={tr_cnt.tolist()} va_cnt={va_cnt.tolist()}")
            print(f"[fold {fold}] max|tr-prop - global|={max_diff_tr:.3f} "
                  f"max|va-prop - global|={max_diff_va:.3f}")

            bs = 1
            ds_tr = dataset_from_indices(Xm, y_mapped, idx_tr, batch_size=bs, shuffle=True, repeat=True)
            ds_va = dataset_from_indices(Xm, y_mapped, idx_va, batch_size=bs, shuffle=False, repeat=False)
            steps_per_epoch = math.ceil(len(idx_tr) / bs)
            val_steps_full  = math.ceil(len(idx_va) / bs)

            M = build_resnet3d(
                input_shape=input_shape,
                num_classes=n_classes,
                initial_filters=P["model"]["initial_filters"],
                block_layers=tuple(P["model"]["block_layers"]),
                bottleneck=P["model"]["bottleneck"],
                dropout_rate=P["model"]["dropout_rate"],
                groups_gn=P["model"]["groups_gn"]
            )

            base_lr = P["training"]["base_lr"]
            wd      = P["training"]["weight_decay"]
            opt = tf.keras.optimizers.AdamW(learning_rate=base_lr, weight_decay=wd)

            if loss_type == "focal":
                loss_obj = categorical_focal_loss(gamma=focal_gamma, alpha=focal_alpha, label_smoothing=0.0)
            else:
                loss_obj = keras.losses.CategoricalCrossentropy(
                    label_smoothing=P["training"]["label_smoothing"]
                )

            M.compile(optimizer=opt, loss=loss_obj, metrics=["accuracy"])

            cb_auc = ValAUC_OvR(ds_va, patience=P["training"]["early_stopping_patience"],
                                workdir=OUT, fold=fold)
            cb_rlr = keras.callbacks.ReduceLROnPlateau(
                monitor="val_auc_ovr_macro",
                mode="max",
                factor=0.5,
                patience=P["training"]["lr_plateau_patience"],
                min_lr=1e-6,
                verbose=1
            )

            def map_one_hot(x, y):
                y = tf.one_hot(tf.cast(y, tf.int32), depth=n_classes)
                return x, y

            ds_tr_oh = ds_tr.map(map_one_hot, num_parallel_calls=tf.data.AUTOTUNE)
            ds_va_oh = ds_va.map(map_one_hot, num_parallel_calls=tf.data.AUTOTUNE)

            use_class_weight = (loss_type == "cce")
            hist = M.fit(
                ds_tr_oh,
                epochs=P["training"]["epochs"],
                steps_per_epoch=steps_per_epoch,
                validation_data=ds_va_oh,
                validation_steps=val_steps_full,
                callbacks=[cb_auc, cb_rlr],
                class_weight=(class_weights if use_class_weight and class_weights is not None else None),
                verbose=2,
            )

            # Collect fold-val predictions for OOF and print fold metrics
            y_prob_va = M.predict(ds_va, verbose=0)
            oof_prob[va] = y_prob_va
            _ = summarize_metrics(prefix=f"Fold {fold} VAL", y_true=y_lab[va], y_prob=y_prob_va)

            # Save fold artifacts
            fold_dir = os.path.join(OUT, f"fold_{fold}")
            ensure_dir(fold_dir)
            with open(os.path.join(fold_dir, "history.json"), "w") as f:
                json.dump({k:[float(x) for x in v] for k,v in hist.history.items()}, f, indent=2)
            if P["output"]["save_fold_models"]:
                M.save(os.path.join(fold_dir, "model.keras"))
            # --- CLEANUP GPU & RAM between folds ---
            del ds_tr, ds_va, ds_tr_oh, ds_va_oh, hist, M
            tf.keras.backend.clear_session()
            gc.collect()

        # OOF metrics
        print("\n========== OOF METRICS ==========")
        oof_stats = summarize_metrics(prefix="OOF", y_true=oof_true, y_prob=oof_prob)

        # Optimize decision scaling via OOF (macro-F1)
        s_vec, best_f1 = optimize_scaling_by_oof(oof_prob, oof_true, rounds=2)
        print(f"\n[Decision scaling] learned s = {s_vec.tolist()} | best OOF F1-macro = {best_f1:.5f}")
        with open(os.path.join(OUT, "decision_scaling.json"), "w") as f:
            json.dump({"scaling": s_vec.tolist(), "oof_best_f1_macro": best_f1}, f, indent=2)

        # Save OOF artifacts
        np.save(os.path.join(OUT, "oof_prob.npy"), oof_prob)
        np.save(os.path.join(OUT, "oof_true.npy"), oof_true)
        with open(os.path.join(OUT, "label_mapping.json"), "w") as f:
            json.dump({str(k): int(v) for k, v in label_map.items()}, f, indent=2)

        with open(os.path.join(OUT, "run_summary.json"), "w") as f:
            json.dump({
                "oof_macro_auc_ovr": float(oof_stats["auc_macro_ovr"]),
                "oof_acc": float(oof_stats["acc"]),
                "oof_bal_acc": float(oof_stats["bal_acc"]),
                "oof_f1_macro": float(oof_stats["f1_macro"]),
                "n_samples_labeled": int(len(idx_lab)),
                "class_weights": class_weights
            }, f, indent=2)

    # =================== TEST ===================
    x_test_path = os.path.join(ROOT, P["data"]["x_test"])
    if os.path.exists(x_test_path):
        Xt, yt = make_memmaps(ROOT, P["data"]["x_test"], P["data"]["y_test"])

        # Load decision scaling (if available) else identity
        s_vec = np.ones((n_classes,), dtype=np.float32)
        scale_path = os.path.join(OUT, "decision_scaling.json")
        if os.path.exists(scale_path):
            try:
                payload = json.load(open(scale_path, "r"))
                s_arr = np.array(payload.get("scaling", []), dtype=np.float32)
                if s_arr.shape == (n_classes,):
                    s_vec = s_arr
                else:
                    print("[TEST] decision_scaling.json shape mismatch; using identity scaling.")
            except Exception as e:
                print(f"[TEST] failed to load decision_scaling.json: {e}; using identity.")

        # Ensemble predictions from all available fold models (prefer best_model.keras if present)
        preds_accum = np.zeros((Xt.shape[0], n_classes), dtype=np.float32)
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
            Mf = keras.models.load_model(load_p, compile=False, custom_objects={"GroupNorm": GroupNorm})
        
            ds_te = dataset_from_indices(Xt, None, np.arange(Xt.shape[0]), batch_size=1)
            try:
                preds_accum += Mf.predict(ds_te, verbose=0)
            except tf.errors.ResourceExhaustedError:
                b = max(1, P["training"]["batch_size"] // 2)
                while True:
                    try:
                        ds_te_small = dataset_from_indices(Xt, None, np.arange(Xt.shape[0]), batch_size=b)
                        preds_accum += Mf.predict(ds_te_small, verbose=0)
                        break
                    except tf.errors.ResourceExhaustedError:
                        if b == 1:
                            raise
                        b = max(1, b // 2)
        
            n_models += 1
       
            # free GPU/graph before next fold model
            del ds_te, Mf
            tf.keras.backend.clear_session()
            gc.collect()
       
        if n_models == 0:
            raise RuntimeError("No fold models found to run TEST.")

        # Save which models were used
        with open(os.path.join(OUT, "test_used_models.json"), "w") as f:
            json.dump(used_models, f, indent=2)

        # Average ensemble
        test_prob = preds_accum / n_models

        # Apply decision scaling (for label assignment/reporting)
        scaled_for_decision = test_prob * s_vec[None, :]
        y_pred_test = np.argmax(scaled_for_decision, axis=1)

        # Save raw probs and scaled-argmax predictions
        np.save(os.path.join(OUT, "test_prob.npy"), test_prob)
        np.save(os.path.join(OUT, "test_pred.npy"), y_pred_test)

        # Report test metrics if labels exist
        if yt is not None:
            y_test_raw = np.array(yt, dtype=float)
            mask = ~np.isnan(y_test_raw)
            y_test = remap_labels(y_test_raw, label_vals)[0][mask].astype(int)
            test_prob_eval = test_prob[mask]
            scaled_eval = scaled_for_decision[mask]

            # Threshold-free AUC on raw probs; acc/F1/etc on scaled decisions
            stats = summarize_metrics(prefix="TEST", y_true=y_test, y_prob=test_prob_eval)
            with open(os.path.join(OUT, "test_summary.json"), "w") as f:
                json.dump(stats, f, indent=2)

            # Detailed report/confusion using the SAME (scaled) decision rule
            y_pred_argmax = np.argmax(scaled_eval, axis=1)
            report_txt = classification_report(y_test, y_pred_argmax, digits=4)
            with open(os.path.join(OUT, "test_classification_report.txt"), "w") as f:
                f.write(report_txt + "\n")

            cm = confusion_matrix(y_test, y_pred_argmax)
            np.save(os.path.join(OUT, "test_confusion_matrix.npy"), cm)
        else:
            print("\n[TEST] No y_test available; saved probabilities and argmax predictions only.")

        if 1:
            # =================== VARIANT (CHANNEL) USAGE VIA IG ===================
            try:
                model_for_ig = None
                if 'M_ig' in locals() and isinstance(M_ig, keras.Model):
                    model_for_ig = M_ig
                else:
                    first_model_path = None
                    used_models_path = os.path.join(OUT, "test_used_models.json")
                    if os.path.exists(used_models_path):
                        um = json.load(open(used_models_path, "r"))
                        if len(um) > 0:
                            cand = os.path.join(OUT, um[0])
                            if os.path.exists(cand):
                                first_model_path = cand
                    if first_model_path is None:
                        for fold in range(1, P["cv"]["n_splits"] + 1):
                            for name in ("best_model.keras", "model.keras"):
                                cand = os.path.join(OUT, f"fold_{fold}", name)
                                if os.path.exists(cand):
                                    first_model_path = cand
                                    break
                            if first_model_path:
                                break
                    if first_model_path:
                        print(f"[VARIANTS] Using model for channel-usage IG: {first_model_path}")
                        model_for_ig = keras.models.load_model(first_model_path, compile=False,
                                                               custom_objects={"GroupNorm": GroupNorm})
            
                if model_for_ig is None:
                    print("[VARIANTS] No model available for IG; skipping variant usage aggregation.")
                else:
                    # --- TEST set aggregation ---
                    idx_te = np.arange(Xt.shape[0], dtype=int)
                    agg_te = _aggregate_channel_usage(
                        model_for_ig, Xt, idx_te,
                        probs=test_prob,
                        steps=32, smooth_repeats=4, noise_std=0.02,
                        post_sigma=1.0, verbose_every=50
                    )
                    print(f"[VARIANTS][TEST] mean={np.round(agg_te['mean'],4).tolist()} "
                          f"std={np.round(agg_te['std'],4).tolist()} n={agg_te['n']}")
            
                    if 'M_ig' not in locals():
                        del model_for_ig
                        tf.keras.backend.clear_session()
                        gc.collect()
            except Exception as e:
                print(f"[VARIANTS] Error while aggregating channel usage: {e}")


    def _ensure_dir(p):
        os.makedirs(p, exist_ok=True)
        return p

    def _save_ig_with_affine(
        ig_xyz: np.ndarray,             # (X,Y,Z) IG volume in the *padded crop grid* (tgt_shape[:3])
        src_path: str,                  # path to the original ASL NIfTI
        box_xyz: tuple,                 # (x0,x1,y0,y1,z0,z1) crop box in native ASL voxel coords
        pad_offsets: tuple,             # (sox, soy, soz) padding offsets applied when centering into tgt_shape
        out_path: str,                  # where to write the NIfTI
        tight: bool = False,            # True -> save tight crop (no padding); False -> save full padded cube
        t1_path: str | None = None,
    ):
        """
        Writes the IG heatmap with a *correct affine* so it overlays on the ASL (and, optionally, on T1).
        If tight=False, we save the whole padded cube at tgt_shape with an affine that accounts for padding.
        If tight=True, we save the tight crop (x1-x0, y1-y0, z1-z0) with affine starting at (x0,y0,z0).
        """
        # Load source (ASL) header/affine
        src_img   = nib.load(src_path)
        src_aff   = src_img.affine.copy()
        x0,x1,y0,y1,z0,z1 = map(int, box_xyz)
        sox, soy, soz     = map(int, pad_offsets)
    
        if tight:
            # extract the non-padded block from ig_xyz and set affine at (x0,y0,z0)
            xs, ys, zs = x1-x0, y1-y0, z1-z0
            # where the real data lives inside the padded IG volume:
            # it begins at (sox,soy,soz) and has size (xs,ys,zs)
            core = ig_xyz[sox:sox+xs, soy:soy+ys, soz:soz+zs]
            arr_xyz = core.astype(np.float32, copy=False)
    
            # affine translation = src_aff * [x0,y0,z0,1]
            new_aff = src_aff.copy()
            new_aff[:3,3] = (src_aff @ np.array([x0, y0, z0, 1.0]))[:3]
    
        else:
            # Save the full padded cube but shift affine so voxel (0,0,0) corresponds to (x0-sox, y0-soy, z0-soz)
            arr_xyz = ig_xyz.astype(np.float32, copy=False)
            ox, oy, oz = (x0 - sox, y0 - soy, z0 - soz)
            new_aff = src_aff.copy()
            new_aff[:3,3] = (src_aff @ np.array([ox, oy, oz, 1.0]))[:3]
    
        img = nib.Nifti1Image(arr_xyz, new_aff, header=src_img.header)
        img.set_sform(new_aff, code=1)
        img.set_qform(new_aff, code=1)
        nib.save(img, out_path)
    
        # also resample to T1 space for perfect T1 overlay
        if t1_path is not None:
            try:
                t1_img = nib.load(t1_path)
                # resample IG to T1 grid (nearest or linear; here linear)
                ig_on_t1 = resample_from_to(img, t1_img, order=1)
                p = out_path.replace(".nii.gz", "_onT1.nii.gz")
                nib.save(ig_on_t1, p)
            except Exception as e:
                print(f"[HEATMAP] T1 resample failed: {e}")

    if 0:
        # =================== EXPLANATION HEATMAPS (CLASS 2 ONLY) ===================
        try:
            # Load IDs to print alongside outputs
            IDS_TEST_PATH = os.path.join(ROOT, "ids_test.npy")
            ids_test = np.load(IDS_TEST_PATH, allow_pickle=True) if os.path.exists(IDS_TEST_PATH) \
                       else np.array([f"sample_{i}" for i in range(Xt.shape[0])])
    
            # Find a model to use for IG (same logic as before)
            first_model_path = None
            used_models_path = os.path.join(OUT, "test_used_models.json")
            if os.path.exists(used_models_path):
                um = json.load(open(used_models_path, "r"))
                if len(um) > 0:
                    cand = os.path.join(OUT, um[0])  # relative -> absolute
                    if os.path.exists(cand):
                        first_model_path = cand
    
            if first_model_path is None:
                for fold in range(1, P["cv"]["n_splits"] + 1):
                    for name in ("best_model.keras", "model.keras"):
                        cand = os.path.join(OUT, f"fold_{fold}", name)
                        if os.path.exists(cand):
                            first_model_path = cand
                            break
                    if first_model_path:
                        break
    
            if first_model_path is None:
                print("[HEATMAP] No model found for IG; skipping heatmap generation.")
            else:
                print(f"[HEATMAP] Using model for IG: {first_model_path}")
                M_ig = keras.models.load_model(first_model_path, compile=False, custom_objects={"GroupNorm": GroupNorm})
    
                # --- Pick exactly ONE test sample: most confidently classified class 2 ---
                K = test_prob.shape[1]
                assert K >= 3, f"Expected at least 3 classes, got {K}"
    
                pred = np.argmax(test_prob, axis=1)
                cand = np.where(pred == 2)[0]  # samples whose predicted class is 2
    
                if cand.size > 0:
                    best_idx = cand[np.argmax(test_prob[cand, 2])]
                else:
                    best_idx = int(np.argmax(test_prob[:, 2]))  # fallback: global max P(class=2)
    
                explain_indices = [int(best_idx)]
                print(f"[HEATMAP] Selected 1 sample (pred=2-most-confident): idx={best_idx}, "
                      f"sid={str(ids_test[best_idx]) if best_idx < len(ids_test) else 'N/A'}")
    
                out_heat = _ensure_dir(os.path.join(OUT, "heatmaps"))
                # IG hyperparams
                IG_STEPS = 128
                IG_SMOOTH = 16
                IG_NOISE_STD = 0.02
                TARGET_CLASS = 2
    
                for i in explain_indices:
                    sid = str(ids_test[i]) if i < len(ids_test) else f"sample_{i}"
                    target_class = int(np.argmax(test_prob[i]))  # predicted class
                
                    # load the (D,H,W,C) volume from memmap
                    vol = np.array(Xt[i], dtype=np.float32)
                    
                    ig_vol = _ig_single(M_ig, vol, target_class,
                                        m_steps=IG_STEPS, n_smooth=IG_SMOOTH,
                                        noise_std=IG_NOISE_STD)
                    
                    ig_norm, _ = _postprocess_ig(ig_vol, vol, sigma=1.2, bg_factor=0.4)
    
                    PATHS_TEST_PATH = os.path.join(ROOT, "paths_test.npy")
                    paths_test = np.load(PATHS_TEST_PATH, allow_pickle=True) if os.path.exists(PATHS_TEST_PATH) \
                                else np.array([None]*Xt.shape[0])
    
                    src_path = str(paths_test[i]) if i < len(paths_test) else None
                    if not src_path or not os.path.exists(src_path):
                        print("[HEATMAP] Missing source path for affine; saving fallback identity NIfTI.")
                        nib.save(nib.Nifti1Image(ig_norm.astype(np.float32), np.eye(4)), os.path.join(out_heat, f"{sid}__class{target_class}_IGheatmap.nii.gz"))
                    else:
                        # Recompute the crop box exactly as in preprocessing (same hyperparams)
                        SIDE_MM  = P["preproc"]["side_mm"] if "preproc" in P and "side_mm" in P["preproc"] else 70.0
                        TOP_PCT  = P["preproc"]["top_pct"] if "preproc" in P and "top_pct" in P["preproc"] else 0.01
                        box = compute_box_for_path(src_path, side_mm=SIDE_MM, top_pct=TOP_PCT)  # (x0,x1,y0,y1,z0,z1)
                       
                        # Figure out the padding offsets that were used to center the crop into tgt_shape
                        x0,x1,y0,y1,z0,z1 = map(int, box)
                        xs, ys, zs = (x1-x0, y1-y0, z1-z0)
                        tgtX, tgtY, tgtZ, _ = Xt.shape[1:]
                       
                        # pad_to_shape put the (xs,ys,zs) block centered inside (tgtX,tgtY,tgtZ):
                        sox = max((tgtX - xs) // 2, 0)
                        soy = max((tgtY - ys) // 2, 0)
                        soz = max((tgtZ - zs) // 2, 0)
                       
                        T1_PATH = None
                       
                        out_name_full  = os.path.join(out_heat, f"{sid}__class{target_class}_IGheatmap.nii.gz")
                        _save_ig_with_affine(
                            ig_xyz=ig_norm,
                            src_path=src_path,
                            box_xyz=box,
                            pad_offsets=(sox, soy, soz),
                            out_path=out_name_full,
                            tight=False,           # save the full padded cube in native coords
                            t1_path=T1_PATH,       # set to a real path to also get *_onT1.nii.gz
                        )
                       
                        # also save a tight-cropped version aligned to (x0,y0,z0) in native ASL:
                        out_name_tight = os.path.join(out_heat, f"{sid}__class{target_class}_IGheatmap__tight.nii.gz")
                        _save_ig_with_affine(
                            ig_xyz=ig_norm,
                            src_path=src_path,
                            box_xyz=box,
                            pad_offsets=(sox, soy, soz),
                            out_path=out_name_tight,
                            tight=True,
                            t1_path=None,
                        )
                       
                        print(f"[HEATMAP] Saved (native): {out_name_full}")
                        print(f"[HEATMAP] Saved (native tight): {out_name_tight}")
    
                # cleanup
                del M_ig
                tf.keras.backend.clear_session()
                gc.collect()
    
        except Exception as e:
            print(f"[HEATMAP] Error during heatmap generation: {e}")
        # =================== END EXPLANATION HEATMAPS ===================

if __name__ == "__main__":
    main()

