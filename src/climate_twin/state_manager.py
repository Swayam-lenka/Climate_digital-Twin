import datetime

import torch


class DigitalTwinStateManager:
    def __init__(self, initial_thermo_tensor, initial_hydro_tensor, current_date):
        """
        Initialize the rolling state.
        Expected tensor shapes: [Batch, Time, Channels, Latitude, Longitude].
        """
        self.thermo_state = initial_thermo_tensor.clone().detach()
        self.hydro_state = initial_hydro_tensor.clone().detach()
        self.current_date = datetime.datetime.strptime(current_date, "%Y-%m-%d")
        self.lookback_window = self.thermo_state.shape[1]
        self.last_hydro_prediction = self.hydro_state[:, -1:, ...].clone()

    def get_current_state(self):
        """Return the current temperature and rainfall history."""
        return self.thermo_state, self.hydro_state

    def set_last_prediction(self, prediction_tensor):
        """Store a forecast for comparison with the next observation."""
        self.last_hydro_prediction = prediction_tensor.clone().detach()

    def assimilate_new_observation(self, new_thermo_day, new_hydro_day):
        """Append a new observation and drop the oldest time step."""
        self.thermo_state = torch.cat(
            [self.thermo_state[:, 1:, ...], new_thermo_day], dim=1
        )
        self.hydro_state = torch.cat(
            [self.hydro_state[:, 1:, ...], new_hydro_day], dim=1
        )
        self.current_date += datetime.timedelta(days=1)
        return self.current_date

    def assimilate_and_correct(self, new_thermo_day, actual_hydro_day, kalman_gain=0.3):
        """
        Apply a simple fixed-gain correction to the previous rainfall forecast,
        then append the corrected field to the rolling state.
        """
        error_map = actual_hydro_day - self.last_hydro_prediction
        corrected_hydro_day = torch.clamp(
            self.last_hydro_prediction + kalman_gain * error_map, min=0.0
        )
        self.thermo_state = torch.cat(
            [self.thermo_state[:, 1:, ...], new_thermo_day], dim=1
        )
        self.hydro_state = torch.cat(
            [self.hydro_state[:, 1:, ...], corrected_hydro_day], dim=1
        )
        self.current_date += datetime.timedelta(days=1)
        return self.current_date, error_map, corrected_hydro_day

    def generate_what_if_scenario(self, delta_t=0.0, delta_h=0.0):
        """Return a perturbed copy without modifying the stored state."""
        scenario_thermo = self.thermo_state.clone()
        scenario_hydro = self.hydro_state.clone()

        if delta_t != 0.0:
            scenario_thermo += delta_t
        if delta_h != 0.0:
            scenario_hydro = torch.clamp(
                scenario_hydro * (1.0 + delta_h), min=0.0
            )
        return scenario_thermo, scenario_hydro
