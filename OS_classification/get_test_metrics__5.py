#!/usr/bin/env python3
import os, sys, json, pathlib, subprocess, shutil, time, keras

OPTUNA_ROOT  = os.path.abspath("./optuna_outputs")
TRAIN_SCRIPT = os.path.abspath("./train__3.py")

def load_json(p):
    with open(p, "r") as f:
        return json.load(f)

def dump_json(obj, p):
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)

def absify_data_root(P, trial_dir):
    if "data" in P and "root" in P["data"]:
        P["data"]["root"] = os.path.abspath(P["data"]["root"])
    if "output" in P and "workdir" in P["output"]:
        P["output"]["workdir"] = trial_dir
    return P

def run_test_for_trial(trial_dir: str):
    trial_dir = os.path.abspath(trial_dir)
    if not os.path.exists(os.path.join(trial_dir, "run_summary.json")):
        print(f"[skip] {trial_dir}: no run_summary.json (training not finished).")
        return

    metrics_out = os.path.join(trial_dir, "test_metrics.json")
    if os.path.exists(metrics_out):
        print(f"[skip] {trial_dir}: test_metrics.json already exists.")
        return

    param_p = os.path.join(trial_dir, "param.json")
    if not os.path.exists(param_p):
        print(f"[skip] {trial_dir}: missing param.json.")
        return

    # Backup original params
    backup_p = os.path.join(trial_dir, "param.train.backup.json")
    if not os.path.exists(backup_p):
        shutil.copy2(param_p, backup_p)

    # Prepare TEST-only params
    P = load_json(param_p)
    P["TRAIN"] = False
    P["TEST"]  = True
    P = absify_data_root(P, trial_dir)
    dump_json(P, param_p)

    print(f"[run] Testing {os.path.basename(trial_dir)}…")
    try:
        proc = subprocess.run(
            [sys.executable, TRAIN_SCRIPT],
            cwd=trial_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )
        with open(os.path.join(trial_dir, "test_run.log"), "w") as f:
            f.write(proc.stdout)

        if proc.returncode != 0:
            print(f"[fail] {trial_dir}: test run failed (code {proc.returncode}).")
            return

        # Collect metrics from files produced by train__3.py
        summary_p = os.path.join(trial_dir, "test_summary.json")       # {"cindex": ...}
        used_p    = os.path.join(trial_dir, "test_used_models.json")   # list of model paths
        if not os.path.exists(summary_p):
            print(f"[warn] {trial_dir}: no test_summary.json produced.")
            return

        summary = load_json(summary_p)
        metrics = {
            "trial_dir": trial_dir,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cindex": float(summary.get("cindex")) if "cindex" in summary else None,
        }
        if os.path.exists(used_p):
            try:
                used = load_json(used_p)
                metrics["n_models"] = len(used)
                metrics["used_models"] = used
            except Exception:
                pass

        dump_json(metrics, metrics_out)
        print(f"[ok] {trial_dir}: wrote test_metrics.json (cindex={metrics.get('cindex')}).")

    finally:
        # Restore original param.json
        if os.path.exists(backup_p):
            shutil.move(backup_p, param_p)

def main():
    root = pathlib.Path(OPTUNA_ROOT)
    if not root.exists():
        print(f"[error] OPTUNA_ROOT not found: {OPTUNA_ROOT}")
        sys.exit(1)

    trials = sorted([p for p in root.iterdir() if p.is_dir()])
    if not trials:
        print("[info] No trial directories found.")
        return

    for td in trials:
        run_test_for_trial(str(td))

if __name__ == "__main__":
    main()

