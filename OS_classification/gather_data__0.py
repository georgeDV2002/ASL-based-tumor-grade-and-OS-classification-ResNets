#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import re
import numpy as np

ASL_DIR = Path("~/programs/asl_os_wtab/raw").expanduser()
EXCEL   = Path("~/programs/asl_os_wtab/UCSF-PDGM-metadata_v2.xlsx").expanduser()

FNAME_RE = re.compile(r'^(UCSF-PDGM-(\d{4}))_ASL\.nii\.gz$', re.IGNORECASE)
_ID3_RE  = re.compile(r'(UCSF-PDGM-)(\d{1,4})', re.IGNORECASE)

# Exclusion due to severe artifacts
EXCLUDE_IDS = [
    "0005", "0013", "0035", "0036", "0050", "0067", "0073", "0075", "0092", "0096",
    "0159", "0196", "0229", "0259", "0264", "0288", "0332", "0379", "0415", "0442",
    "0474", "0489", "0522", "0526",
]

os_col = "OS"
ev_col = "1-dead 0-alive"

def _norm_id3_from_any(x: str) -> str | None:
    """Normalize any ID-ish value to UCSF-PDGM-XXX (3 digits) used in Excel."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    m = _ID3_RE.search(s)
    if m:
        d = m.group(2)[-3:]
        return f"UCSF-PDGM-{d.zfill(3)}"
    m2 = re.search(r'(\d{1,4})$', s)
    if m2:
        d = m2.group(1)[-3:]
        return f"UCSF-PDGM-{d.zfill(3)}"
    return None

def build_available_scans_os(asl_dir=ASL_DIR, excel_path=EXCEL,
                             out_xlsx="available_scans_BIN__0.xlsx"):
    # 1) Collect scans
    paths, ids4, ids3 = [], [], []
    for p in sorted(Path(asl_dir).glob("*_ASL.nii.gz")):
        m = FNAME_RE.match(p.name)
        if not m:
            continue
        full4 = m.group(1)
        digits4 = m.group(2)
        if digits4 in EXCLUDE_IDS:
            continue
        id3 = f"UCSF-PDGM-{digits4[-3:]}"
        paths.append(str(p))
        ids4.append(full4)
        ids3.append(id3)

    if not paths:
        print("[INFO] No matching *_ASL.nii.gz scans found.")
        return

    # 2) Read Excel
    df = pd.read_excel(excel_path)
    df.columns = df.columns.astype(str).str.strip()

    # ID column
    possible_id_cols = ["ID", "Subject ID", "SubjectID", "Subject", "Patient", "Case"]
    id_col = None
    for c in possible_id_cols:
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        for c in df.columns:
            if re.search(r"\bID\b", c, re.IGNORECASE):
                id_col = c
                break
    if id_col is None:
        raise ValueError("Could not locate an ID column in the Excel.")

    # Ensure survival/event columns exist
    if os_col not in df.columns:
        raise ValueError(f"Excel missing expected column '{os_col}'")
    if ev_col not in df.columns:
        raise ValueError(f"Excel missing expected column '{ev_col}'")

    # Tabular inputs
    age_series = pd.to_numeric(df.get("Age at MRI"), errors="coerce")  # float with NaNs

    def _norm_sex(x):
        if pd.isna(x): return np.nan
        s = str(x).strip().upper()
        if s == "M": return 0.0
        if s == "F": return 1.0
        return np.nan

    sex_series = df.get("Sex").map(_norm_sex)  # float with NaNs

    def _norm_eor(x):
        if pd.isna(x): return np.nan
        s = str(x).strip().upper()
        if s == "GTR":    return 0.0
        if s == "STR":    return 1.0
        if s == "BIOPSY": return 2.0
        return np.nan

    eor_series = df.get("EOR").map(_norm_eor)  # float with NaNs

    # Normalize IDs and build maps
    df["_norm_id3"] = df[id_col].map(_norm_id3_from_any)

    os_map  = dict(zip(df["_norm_id3"], pd.to_numeric(df[os_col], errors="coerce")))
    ev_map  = dict(zip(df["_norm_id3"], pd.to_numeric(df[ev_col], errors="coerce")))
    age_map = dict(zip(df["_norm_id3"], age_series))
    sex_map = dict(zip(df["_norm_id3"], sex_series))
    eor_map = dict(zip(df["_norm_id3"], eor_series))

    # 3) Assemble table
    out = pd.DataFrame({
        "subject_id_4d": ids4,
        "subject_id_3d": ids3,
        "path": paths,
        "OS": [os_map.get(i3, np.nan) for i3 in ids3],
        "event": [ev_map.get(i3, np.nan) for i3 in ids3],
        "Age": [age_map.get(i3, np.nan) for i3 in ids3],
        "Sex": [sex_map.get(i3, np.nan) for i3 in ids3],
        "EOR": [eor_map.get(i3, np.nan) for i3 in ids3],
    })

    # 4) Save
    out.to_excel(out_xlsx, index=False)
    print(f"[OK] Wrote {len(out)} rows to '{out_xlsx}'")

if __name__ == "__main__":
    build_available_scans_os()

