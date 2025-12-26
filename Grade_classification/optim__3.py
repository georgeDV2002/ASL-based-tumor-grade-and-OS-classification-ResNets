#!/usr/bin/env python3
import os, json, pathlib, subprocess, sys, shutil
from datetime import datetime
import optuna

# ---- user knobs ----
N_TRIALS = 200
TRAIN_SCRIPT = os.path.abspath("./train__2.py")
BASE_PARAMS  = os.path.abspath("./param.json")
OUT_ROOT     = os.path.abspath("./optuna_outputs")

# --------------- helpers ---------------
def load_json(p):
    with open(p, "r") as f:
        return json.load(f)

def dump_json(obj, p):
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)

def make_trial_dir(trial_number: int):
    root = pathlib.Path(OUT_ROOT)
    d = root / f"output_{trial_number+1:04d}"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=False)
    return str(d)

def absify_paths(P, trial_dir):
    P["data"]["root"] = os.path.abspath(P["data"]["root"])
    P["output"]["workdir"] = os.path.abspath(trial_dir)
    return P

# --------------- objective ---------------
def objective(trial: optuna.Trial) -> float:
    P = load_json(BASE_PARAMS)
    trial_dir = make_trial_dir(trial.number)
    P = absify_paths(P, trial_dir)

    # --- SEARCH SPACE ---
   
    # Optimizer & regularization
    base_lr = trial.suggest_float("base_lr", 1e-4, 3e-3, log=True)
    # either categorical or log; categorical is reproducible across runs
    weight_decay = trial.suggest_categorical(
        "weight_decay", [0.0, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4]
    )
    # Normalization
    groups_gn = trial.suggest_categorical("groups_gn", [4, 8, 16, 32])

    # Capacity / depth family
    use_bottleneck = trial.suggest_categorical("bottleneck", [False, True])
    block_layers = [2,2,2,2] if not use_bottleneck else [3,4,6,3]
    
    # Width & regularization
    initial_filters = trial.suggest_categorical("initial_filters", [16, 24, 32, 48])
    dropout_rate = trial.suggest_categorical("dropout_rate", [0.0, 0.1, 0.2, 0.3, 0.4])
    
    # Callback knobs
    early_stopping_patience = trial.suggest_int("early_stopping_patience", 10, 22)
    lr_plateau_patience = trial.suggest_int("lr_plateau_patience", 2, 5)
   
    #loss_type = trial.suggest_categorical("loss_type", ["cce", "focal"])
    loss_type = trial.suggest_categorical("loss_type", ["cce"])
    P["training"]["loss_type"] = loss_type

    if loss_type == "cce":
        P["training"]["label_smoothing"] = float(trial.suggest_float("label_smoothing", 0.0, 0.12))
        P["training"]["focal_gamma"] = 2.0
        P["training"]["focal_alpha"] = None
    else:
        P["training"]["label_smoothing"] = 0.0
        P["training"]["focal_gamma"] = float(trial.suggest_float("focal_gamma", 1.0, 3.0))
        P["training"]["focal_alpha"] = None

    # ---- apply into params ----
    P["TRAIN"] = True
    P["TEST"]  = False
    P["training"]["base_lr"] = float(base_lr)
    P["training"]["weight_decay"] = float(weight_decay)
    P["model"]["initial_filters"] = int(initial_filters)
    P["model"]["bottleneck"] = bool(use_bottleneck)
    P["model"]["block_layers"] = block_layers
    P["model"]["dropout_rate"] = float(dropout_rate)
    
    P["training"]["early_stopping_patience"] = int(early_stopping_patience)
    P["training"]["lr_plateau_patience"] = int(lr_plateau_patience)
    P["training"]["batch_size"] = 1
    P["model"]["groups_gn"] = int(groups_gn)
    if "norm" in P["model"]:
        del P["model"]["norm"]

    P["output"]["workdir"] = os.path.abspath(trial_dir)

    # write params.json inside trial folder so the trainer picks it up
    dump_json(P, os.path.join(trial_dir, "param.json"))

    # --- run training (serial) ---
    env = os.environ.copy()
    cmd = [sys.executable, TRAIN_SCRIPT]  # use current python
    print(f"[trial {trial.number}] Running: {cmd} in {trial_dir}")
    proc = subprocess.run(cmd, cwd=trial_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # save raw log
    with open(os.path.join(trial_dir, "train.log"), "w") as f:
        f.write(proc.stdout)

    if proc.returncode != 0:
        print(f"[trial {trial.number}] Training failed (code {proc.returncode}). See train.log")
        # return very low score so Optuna moves on
        return 0.0

    # --- read metric from run_summary.json ---
    summary_path = os.path.join(trial_dir, "run_summary.json")
    if not os.path.exists(summary_path):
        print(f"[trial {trial.number}] Missing run_summary.json; returning 0.")
        return 0.0

    try:
        summary = load_json(summary_path)
        score = float(summary.get("oof_macro_auc_ovr", 0.0))
    except Exception as e:
        print(f"[trial {trial.number}] Failed to parse run_summary.json: {e}")
        score = 0.0

    trial.report(score, step=0)
    return score

# --------------- main ---------------
def main():
    pathlib.Path(OUT_ROOT).mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True, group=True)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=0)

    study = optuna.create_study(
        direction="maximize",
        study_name="resnet3d",
        storage=f"sqlite:///{os.path.join(OUT_ROOT, 'study.db')}",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True)

    # Save study artifacts
    best = study.best_trial
    print("\n=== BEST TRIAL ===")
    print(f"number={best.number}, value={best.value}")
    print("params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    # persist best params and summary
    dump_json({"best_value": best.value, "best_params": best.params}, os.path.join(OUT_ROOT, "best.json"))
    # also save a params.json for best (re-creating a runnable folder)
    best_dir = os.path.join(OUT_ROOT, "best_replay")
    pathlib.Path(best_dir).mkdir(parents=True, exist_ok=True)

    # recreate best params.json using template + best params
    P = load_json(BASE_PARAMS)
    P = absify_paths(P, best_dir)
    P["TRAIN"] = True
    P["TEST"]  = False
    P["training"]["base_lr"]       = float(best.params["base_lr"])
    P["training"]["weight_decay"]  = float(best.params["weight_decay"])
    P["model"]["initial_filters"]  = int(best.params["initial_filters"])
    P["model"]["bottleneck"]       = bool(best.params["bottleneck"])
    P["model"]["block_layers"]     = [2,2,2,2] if not P["model"]["bottleneck"] else [3,4,6,3]
    P["model"]["dropout_rate"]     = float(best.params["dropout_rate"])
    P["output"]["workdir"]         = os.path.abspath(best_dir)
    P["training"]["early_stopping_patience"] = int(best.params["early_stopping_patience"])
    P["training"]["lr_plateau_patience"]     = int(best.params["lr_plateau_patience"])

    P["training"]["batch_size"] = 1
    P["model"]["groups_gn"] = int(best.params["groups_gn"])
    if "norm" in P["model"]:
        del P["model"]["norm"]

    P["training"]["loss_type"] = best.params["loss_type"]
    if P["training"]["loss_type"] == "cce":
        P["training"]["label_smoothing"] = float(best.params["label_smoothing"])
        P["training"]["focal_gamma"] = 2.0
        P["training"]["focal_alpha"] = None
    else:
        P["training"]["label_smoothing"] = 0.0
        P["training"]["focal_gamma"] = float(best.params["focal_gamma"])
        P["training"]["focal_alpha"] = None

    dump_json(P, os.path.join(OUT_ROOT, "best_params.json"))

if __name__ == "__main__":
    main()

