# Implementation notes

## What is in this repository

The notebook is the main baseline workflow for preprocessing IMD gridded data, defining the forecasting model, training and plotting. The Python state-management files are supporting prototypes and should not be described as a completed operational digital twin.

## Important distinction

The FastAPI application in `examples/api_server.py` uses random tensors for its initial state, incoming observations and forecast outputs. It is useful as an endpoint/UI integration demonstration only. It does not load the trained notebook model or query IMD.

`src/climate_twin/state_manager.py` contains the rolling-state and what-if perturbation methods used by that API example. `src/climate_twin/assimilation.py` is a separate assimilation implementation; it is not wired into the API example.

## Before presenting results

1. Download the required IMD files and update notebook paths.
2. Check the grid dimensions, date coverage and preprocessing assumptions against the downloaded files.
3. Run the notebook's preprocessing and training cells in the documented order.
4. Use a consistent chronological validation/test split before reporting final metrics.
5. Save the preprocessing scalers and trained checkpoint if repeatable inference is required.
6. Replace the API's random inputs and outputs with real observations and model inference before treating it as a live service.
