# Analog Experiments

This folder contains utilities for running analog experiments on Google devices. It contains `AnalogTrajectory`, for defining qubit-frequency and coupler-strength trajectories, and the circuit builders `GenericAnalogCircuitBuilder` (which emits `AnalogDetuneQubit` and `AnalogDetuneCouplerOnly` operations for hardware) and `AnalogSimulationCircuitBuilder` (which emits a Trotterized circuit for simulation).

For an example of usage, see
[`analog_experiment_tutorial.ipynb`](../../../../docs/tutorials/google/analog_experiment_tutorial.ipynb).
