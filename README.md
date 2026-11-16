# Climate Digital Twin

A baseline rainfall-forecasting project for the Western Ghats using IMD gridded rainfall and temperature data.

## Project layout

- `notebooks/baseline_preprocess.ipynb` — preprocessing, model definition, training and evaluation workflow.
- `src/climate_twin/state_manager.py` — rolling state and scenario-perturbation utilities.
- `src/climate_twin/assimilation.py` — separate data-assimilation prototype.
- `examples/api_server.py` — FastAPI demonstration endpoints.
- `results/figures/` — figures used in the project presentation.
- `docs/hyperparameter_tuning.txt` — recorded hyperparameter-search result.

## Forecasting notebook

The notebook describes a next-day rainfall baseline for the Western Ghats. Its configuration uses three previous days of input, Tmax/Tmin and rainfall, a 53 × 29 regional grid, and a DualStream attention U-ConvLSTM model.

The notebook expects six annual IMD grid files:

```text
Rainfall_ind2023_rfp25.grd
Rainfall_ind2024_rfp25.grd
Maxtemp_MaxT_2023.GRD
Maxtemp_MaxT_2024.GRD
Mintemp_MinT_2023.GRD
Mintemp_MinT_2024.GRD
```

Download the data from the official IMD pages:

- [Daily gridded rainfall](https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html)
- [Maximum temperature](https://imdpune.gov.in/cmpg/Griddata/Max_1_Bin.html)
- [Minimum temperature](https://imdpune.gov.in/cmpg/Griddata/Min_1_Bin.html)

Update the file paths in the notebook to point to the downloaded files. The raw IMD datasets are not included in this repository.

## Environment

Create and activate a virtual environment, then install the notebook dependencies:

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook notebooks/baseline_preprocess.ipynb
```

Run the notebook selectively and follow the cell order and cautions documented in the notebook. Do not assume that every cell should be run sequentially: some cells are exploratory or use different data splits.

## API demonstration

The API in `examples/api_server.py` is a prototype. It currently initializes random tensors and returns mock predictions; it does not load the trained notebook model or perform real IMD inference. It is separate from the notebook training workflow.

Install the optional API dependencies and run it from the repository root:

```bash
pip install -r requirements-api.txt
pip install -e .
uvicorn api_server:app --app-dir examples --reload
```

## Current limitations

- Raw IMD data files and trained model weights are not committed.
- The notebook's data paths must be changed for the local machine.
- The API is a mock demonstration, not a deployed forecasting service.
- The recorded hyperparameter result is included as a note; it is not evidence of a completed, independently validated model-selection run.
- Forecast performance should only be reported after the data pipeline and evaluation split have been checked and a full training/evaluation run has been completed.

## Data and model files

Keep large raw datasets, processed cubes and model checkpoints outside Git unless there is a clear reason to version them. The `.gitignore` file excludes common local data, checkpoint and environment files.
