# Climate_digital-Twin
# CLIMATE-TWIN — Run the Baseline

PyTorch next-day rainfall forecasting for the Western Ghats. The actual training
configuration uses 3 previous days, Tmax/Tmin plus rainfall, a 53 x 29 grid, and
DualStream_AttnUConvLSTM(hidden_dim=32). It does NOT use the API's 14-day mock data.

## What each file does

- baseline_preprocess.ipynb: actual GRD preprocessing, model/loss definitions,
  training, evaluation, plotting, and optional Optuna search.
- state_manager.py: rolling observation history and cloned scenario perturbations.
- assimilation.py: alternative state-manager class with fixed-gain correction and
  forecast storage; it lacks the scenario method from state_manager.py.
- api_server.py: FastAPI demonstration using RANDOM observations/predictions;
  it does not load the trained model and is not needed for notebook training.

## 1. Set up

Copy the notebook and Python files into one project folder; open a terminal there.

    python -m venv .venv

Activate on Windows PowerShell:

    .venv\Scripts\Activate.ps1

Or Linux/macOS:

    source .venv/bin/activate

Install dependencies used by the notebook:

    python -m pip install --upgrade pip
    python -m pip install torch numpy pandas xarray scipy bottleneck netCDF4 matplotlib pytorch-msssim notebook ipykernel

Optional tuning/API dependencies:

    python -m pip install optuna fastapi uvicorn

No pinned environment was supplied. These commands install the required package
families, not a verified version-locked environment. CUDA is selected automatically
when available; otherwise training uses CPU. Exact trained weights require copying
the original checkpoint; retraining is not guaranteed to reproduce them exactly.

## 2. Prepare the data

Obtain these six IMD files and edit their hard-coded D:\roadmap\ISRO\... paths
in BOTH preprocessing cells to match your computer, the data is large so it
needs to be downloaded from the website given below and the path should be
updated in the codefile, wherever mentioned.


Dataset Required

Rainfall data: https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html

Max Temp. data: https://imdpune.gov.in/cmpg/Griddata/Max_1_Bin.html

Min. Temp. data: https://www.imdpune.gov.in/cmpg/Griddata/Min_1_Bin.html


    Rainfall_ind2023_rfp25.grd
    Rainfall_ind2024_rfp25.grd
    Maxtemp_MaxT_2023.GRD
    Maxtemp_MaxT_2024.GRD
    Mintemp_MinT_2023.GRD
    Mintemp_MinT_2024.GRD

The reader expects little-endian float32 annual grids: rainfall 129 x 135,
temperature 31 x 31. Run from the project folder so relative output paths agree.


## 3. Train — run selected cells in this order

    jupyter notebook baseline_preprocess.ipynb

1. First code cell: creates WG_Baseline_Normalized_Cube_Combined.nc from 2023/2024.
   Skip if that exact processed file was copied.
2. Model cell headed 'Spatiotemporal Memory: ConvLSTM Cell' (after the repair).
3. Loss cell defining PerfectClimateLoss.
4. Training cell headed 'The IMD-Only Foundation Dataset'.
5. Plotting cell headed 'Visualizing the AI's Predictions (Map-Accurate)'.

Training runs 100 epochs with batch size 32, hidden_dim=32, a 3-day lookback,
an 80/20 chronological train/validation split, AdamW, cosine scheduling, and
gradient clipping. It saves:

    IMD_Foundation_Model.pth    (FINAL epoch, not best validation checkpoint)
    loss_curve_ppt.png

Do NOT blindly Run All. The second preprocessing cell is an optional 2023-only
visualization; it does not overwrite the saved combined cube. The early dataset
cell creates a DIFFERENT 5-day, 70/15/15 setup. Its test_loader overlaps the later
training/validation periods, so the later 'held-out test' metrics are NOT an
independent test. Skip the quick test and multi-metric test cells until a consistent
held-out split is implemented. The last Optuna cell is optional and expensive:
50 trials, up to 20 epochs each; it does not automatically retrain/save its winner.

## 5. Load existing weights without retraining

Run the corrected model-definition cell, then run:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualStream_AttnUConvLSTM(ocean_vars=2, land_vars=1, hidden_dim=32).to(device)
    model.load_state_dict(torch.load('IMD_Foundation_Model.pth', map_location=device, weights_only=True))
    model.eval()

For inference, provide aligned float32 tensors using the SAME preprocessing:

    # x_therm: [B, 3, 2, 53, 29]; x_hydro: [B, 3, 1, 53, 29]
    with torch.no_grad():
        prediction = model(x_therm.to(device), x_hydro.to(device))
    # Output: [B, 1, 53, 29], next-day NORMALIZED rainfall, not millimetres.

Rainfall uses log1p then 2023-fitted min/max scaling; temperatures use 2023-fitted
z-scores. The notebook does not save these scalers separately: retain/recompute
them from the same raw data. Rainfall inversion is expm1(prediction *
(r_max - r_min) + r_min). The training code does not apply the visualization mask.

## Optional API demo — not model inference

    python -m uvicorn api_server:app --reload
