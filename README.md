# ASL-based-tumor-grade-and-OS-classification-ResNets
Code for paper "ASLNet: An Explainable Deep Learning Framework for Glioma Grading and Survival Prediction". If you use this code or any part of it in academic work, please cite the corresponding publication.

## Repository Structure and Pipelines

This repository contains two closely related pipelines:

- Tumor grade classification  
- Overall survival (OS) prediction  

Both pipelines share the same data organization and preprocessing logic. Differences are limited to task-specific scripts and evaluation steps.

### Script naming convention

Scripts are prefixed with `__X`, where `X` denotes the recommended execution order (lower numbers should be run first).

---

## Data Organization

### raw/ directory

The `raw/` directory contains ASL MRI volumes (`.nii.gz`) that were manually copied from the original TCIA UCSF-PDGM dataset into this repository.

Example:

raw/  
├── UCSF-PDGM-0004_ASL.nii.gz  
├── UCSF-PDGM-0005_ASL.nii.gz  
├── UCSF-PDGM-0006_ASL.nii.gz  
└── ...

Each file follows the naming pattern:

UCSF-PDGM-XXXX_ASL.nii.gz

Only ASL volumes used in the experiments are included. Some subjects are excluded due to severe imaging artifacts, as handled in the preprocessing scripts.

### Clinical metadata (Excel file)

The official UCSF-PDGM Excel metadata file provided on the TCIA website is NOT included in this repository.

This file must be downloaded separately from TCIA and placed locally. The pipelines expect this file to contain, at minimum:

- Subject identifiers  
- Tumor grade (for grade classification)  
- Overall survival time  
- Survival event indicator  
- Tabular clinical covariates  

The expected file path and column names are specified directly in the scripts.

---

## Grade Prediction Pipeline

Recommended execution order:

1. preprocessing__1.py  
   - Loads ASL volumes and metadata  
   - Applies filtering and preprocessing  
   - Generates NumPy arrays for imaging and tabular data  

2. optim__3.py  
   - Hyperparameter optimization using Optuna  
   - Cross-validated training  
   - Stores fold models and out-of-fold predictions  

3. train__2.py  
   - Trains the final grade classification model  
   - Produces test-set predictions and metrics  

4. get_test_metrics__5.py  
   - Aggregates final test performance metrics  

5. dca__4.py  
   - Performs Decision Curve Analysis (DCA) using  
     - out-of-fold probabilities  
     - test-set probabilities  

---

## Overall Survival (OS) Pipeline

The OS pipeline follows the same data assumptions and organization as the grade pipeline.

Recommended execution order:

1. gather_data__0.py  
   - Matches ASL scans with survival metadata  
   - Filters subjects with missing or invalid OS information  

2. preprocessing__2a.py  
   - Prepares imaging and tabular inputs for OS prediction  

3. optim__4.py  
   - Hyperparameter optimization for survival modeling  

4. train__3.py  
   - Trains the final OS prediction model  
   - Produces test-set predictions  

5. get_test_metrics__5.py  
   - Computes OS-related performance metrics  

6. dca__6.py  
   - Performs Decision Curve Analysis (DCA) for OS prediction  

---

## Notes

- The raw/ directory and clinical metadata handling are identical for both pipelines.  
- Full reproducibility is achieved once the original TCIA metadata file is provided by the user.  
- No TCIA-distributed metadata files are redistributed in this repository.
- Requirement files for reproduction of python virtual environments are included.
