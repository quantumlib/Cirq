# Copyright 2020 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Workflow utilities for sampling and measurement collection."""

from cirq.work.collector import CircuitSampleJob as CircuitSampleJob, Collector as Collector
from cirq.work.pauli_sum_collector import PauliSumCollector as PauliSumCollector
from cirq.work.observable_settings import (
    InitObsSetting as InitObsSetting,
    _MeasurementSpec as _MeasurementSpec,
    observables_to_settings as observables_to_settings,
)
from cirq.work.observable_grouping import group_settings_greedy as group_settings_greedy
from cirq.work.observable_measurement_data import (
    ObservableMeasuredResult as ObservableMeasuredResult,
    BitstringAccumulator as BitstringAccumulator,
    flatten_grouped_results as flatten_grouped_results,
)
from cirq.work.observable_measurement import (
    VarianceStoppingCriteria as VarianceStoppingCriteria,
    RepetitionsStoppingCriteria as RepetitionsStoppingCriteria,
    measure_grouped_settings as measure_grouped_settings,
)
from cirq.work.observable_readout_calibration import (
    calibrate_readout_error as calibrate_readout_error,
)
from cirq.work.sampler import Sampler as Sampler
from cirq.work.zeros_sampler import ZerosSampler as ZerosSampler
