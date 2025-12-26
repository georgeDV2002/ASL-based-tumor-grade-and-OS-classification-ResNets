#!/usr/bin/env python3
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ASL_DIR = Path("~/programs/asl_os_wtab/raw").expanduser()
EXCEL   = Path("~/programs/asl_os_wtab/UCSF-PDGM-metadata_v2.xlsx").expanduser()
SCANS_XLSX = Path("available_scans_OS__0.xlsx")

OS_SOURCE_UNITS = "days"
DAYS_PER_MONTH = 30.44

def _load_os_values(prefer_scans_xlsx=True) -> pd.Series:
    """
    Returns a clean pd.Series of OS values (float), NaNs dropped.
    If prefer_scans_xlsx and file exists, read OS from there (only available scans).
    Else, read directly from the original Excel (all subjects).
    """
    if prefer_scans_xlsx and SCANS_XLSX.exists():
        df = pd.read_excel(SCANS_XLSX)
        if "OS" not in df.columns:
            raise ValueError("Column 'OS' not found in available_scans_OS__0.xlsx")
        os_vals = pd.to_numeric(df["OS"], errors="coerce")
        src = "available_scans_OS__0.xlsx"
    else:
        df = pd.read_excel(EXCEL)
        col = "OS"
        os_vals = pd.to_numeric(df[col], errors="coerce")
        src = str(EXCEL)

    os_vals = os_vals.replace([np.inf, -np.inf], np.nan).dropna()
    # basic sanity bounds: keep >0 and < 18250 days (~50 years)
    os_vals = os_vals[(os_vals > 0) & (os_vals < 18250)]
    if os_vals.empty:
        raise ValueError("No valid OS values after cleaning.")
    print(f"[INFO] Loaded {len(os_vals)} OS values from {src}")
    return os_vals

def _moving_average(y, k=7):
    if k <= 1: return y
    k = int(k) + (int(k) % 2 == 0)  # make odd
    pad = k // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(k) / k
    return np.convolve(ypad, kernel, mode="valid")

def _print_stats(x: pd.Series, unit_label: str):
    x = x.values
    q = np.percentile(x, [5, 25, 50, 75, 95])
    stats = {
        "count": int(x.size),
        "min": float(np.min(x)),
        "5%": float(q[0]),
        "Q1": float(q[1]),
        "median": float(q[2]),
        "Q3": float(q[3]),
        "95%": float(q[4]),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
    }
    print(f"\n=== OS summary (units: {unit_label}) ===")
    for k, v in stats.items():
        print(f"{k:>7}: {v:.3f}")

def plot_hist(os_vals: pd.Series, unit_label: str, bins=40, out="os_hist.png"):
    x = os_vals.values
    lo, hi = np.percentile(x, [0, 99.5])
    edges = np.linspace(lo, hi, bins + 1)
    hist, edges = np.histogram(x, bins=edges, density=True)
    ctrs = 0.5 * (edges[:-1] + edges[1:])
    sm   = _moving_average(hist, k=9)

    plt.figure(figsize=(8, 5))
    plt.hist(x, bins=edges, density=True, alpha=0.35)
    plt.plot(ctrs, hist, linewidth=1.0, label="density (raw)")
    plt.plot(ctrs, sm, linewidth=2.0, label="density (smoothed)")
    plt.xlabel(f"OS ({unit_label})")
    plt.ylabel("Density")
    plt.title(f"OS Distribution ({unit_label})")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"[OK] Saved {out}")

def plot_hist_log(os_vals: pd.Series, unit_label: str, bins=40, out="os_hist_log.png"):
    x = np.log1p(os_vals.values)
    lo, hi = np.percentile(x, [0, 99.5])
    edges = np.linspace(lo, hi, bins + 1)
    hist, edges = np.histogram(x, bins=edges, density=True)
    ctrs = 0.5 * (edges[:-1] + edges[1:])
    sm   = _moving_average(hist, k=9)

    plt.figure(figsize=(8, 5))
    plt.hist(x, bins=edges, density=True, alpha=0.35)
    plt.plot(ctrs, hist, linewidth=1.0, label="density (raw)")
    plt.plot(ctrs, sm, linewidth=2.0, label="density (smoothed)")
    plt.xlabel(f"log1p(OS {unit_label})")
    plt.ylabel("Density")
    plt.title(f"OS Distribution (log1p scale, {unit_label})")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"[OK] Saved {out}")

def plot_box(os_vals: pd.Series, unit_label: str, out="os_box.png"):
    x = os_vals.values
    plt.figure(figsize=(6, 4))
    plt.boxplot(x, vert=True, showfliers=True)
    plt.ylabel(f"OS ({unit_label})")
    plt.title("OS Boxplot")
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"[OK] Saved {out}")

def plot_ecdf(os_vals: pd.Series, unit_label: str, out="os_ecdf.png"):
    x = np.sort(os_vals.values)
    y = np.arange(1, len(x) + 1) / len(x)
    plt.figure(figsize=(7, 5))
    plt.plot(x, y, linewidth=2.0)
    plt.xlabel(f"OS ({unit_label})")
    plt.ylabel("Empirical CDF")
    plt.title(f"OS ECDF ({unit_label})")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"[OK] Saved {out}")

def main():
    unit_label = "Days"
    os_vals = _load_os_values(prefer_scans_xlsx=True)  # set False to use full Excel
    _print_stats(os_vals, unit_label)
    plot_hist(os_vals, unit_label, bins=40, out="os_hist.png")
    plot_hist_log(os_vals, unit_label, bins=40, out="os_hist_log.png")
    plot_box(os_vals, unit_label, out="os_box.png")
    plot_ecdf(os_vals, unit_label, out="os_ecdf.png")

if __name__ == "__main__":
    main()
 
