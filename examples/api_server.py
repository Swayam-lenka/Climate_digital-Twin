from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np

from climate_twin.state_manager import DigitalTwinStateManager

app = FastAPI(title="CLIMATE-TWIN Backend API", version="1.0")

# Demonstration state only. Replace these random tensors with aligned IMD
# observations and a trained forecasting model before using this as a service.
init_thermo = torch.rand(1, 14, 2, 64, 64)
init_hydro = torch.rand(1, 14, 1, 64, 64)
twin = DigitalTwinStateManager(init_thermo, init_hydro, current_date="2026-07-01")


class ScenarioRequest(BaseModel):
    delta_t: float = 0.0
    delta_h: float = 0.0
    target_date: str | None = None


@app.get("/")
def health_check():
    return {"status": "Demo active", "twin_date": twin.current_date.strftime("%Y-%m-%d")}


@app.post("/trigger_daily_assimilation")
def assimilate_next_day():
    # Mock observations: this endpoint does not fetch live IMD data.
    new_thermo_day = torch.rand(1, 1, 2, 64, 64)
    actual_hydro_day = torch.rand(1, 1, 1, 64, 64)

    new_date, error_map, corrected_state = twin.assimilate_and_correct(
        new_thermo_day,
        actual_hydro_day,
        kalman_gain=0.3,
    )
    next_prediction = torch.rand(1, 1, 1, 64, 64)
    twin.set_last_prediction(next_prediction)

    mse = (error_map ** 2).mean().item()
    health_score = max(0.0, 100.0 - (mse * 10))

    return {
        "message": "Mock data-assimilation demonstration completed",
        "new_system_date": new_date.strftime("%Y-%m-%d"),
        "twin_health_score": f"{round(health_score, 2)}%",
        "maps_for_ui": {
            "predicted_yesterday": twin.last_hydro_prediction.squeeze().tolist(),
            "actual_observed_today": actual_hydro_day.squeeze().tolist(),
            "calculated_error_map": error_map.squeeze().tolist(),
            "final_corrected_state": corrected_state.squeeze().tolist(),
        },
    }


@app.post("/simulate_scenario")
def run_scenario(request: ScenarioRequest):
    try:
        scen_t, scen_h = twin.generate_what_if_scenario(
            delta_t=request.delta_t,
            delta_h=request.delta_h,
        )
        # Mock outputs for the prototype; no trained model inference occurs.
        baseline_pred = torch.rand(64, 64)
        scenario_pred = torch.rand(64, 64)
        impact_matrix = scenario_pred - baseline_pred

        return {
            "baseline_grid": baseline_pred.tolist(),
            "scenario_grid": scenario_pred.tolist(),
            "impact_heatmap": impact_matrix.tolist(),
            "metrics": {
                "max_rain_increase_mm": round(impact_matrix.max().item(), 2),
                "max_rain_decrease_mm": round(impact_matrix.min().item(), 2),
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
