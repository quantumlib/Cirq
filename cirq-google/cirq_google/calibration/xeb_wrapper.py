# Copyright 2021 The Cirq Developers
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
import contextlib
import multiprocessing
import multiprocessing.pool
from typing import Optional, Union, Iterator

import cirq
import cirq.experiments.random_quantum_circuit_generation as rqcg
import cirq.experiments.xeb_fitting as xebf
import cirq.experiments.xeb_sampling as xebsamp
from cirq_google.calibration.phased_fsim import (
    PhasedFSimCalibrationResult,
    PhasedFSimCharacterization,
    LocalXEBPhasedFSimCalibrationRequest,
    LocalXEBPhasedFSimCalibrationOptions,
)


@contextlib.contextmanager
def _maybe_multiprocessing_pool(
    n_processes: Optional[int] = None,
) -> Iterator[Union['multiprocessing.pool.Pool', None]]:
    """Yield a multiprocessing.Pool as a context manager, unless n_processes=1; then yield None,
    which should disable multiprocessing in XEB apis."""
    if n_processes == 1:
        yield None
        return

    with multiprocessing.get_context('spawn').Pool(processes=n_processes) as pool:
        yield pool


def run_local_xeb_calibration(
    calibration: LocalXEBPhasedFSimCalibrationRequest, sampler: cirq.Sampler
) -> PhasedFSimCalibrationResult:
    """Run a calibration request using `cirq.experiments` XEB utilities and a sampler rather
    than `Engine.run_calibrations`.

    Args:
        calibration: A LocalXEBPhasedFSimCalibration request describing the XEB characterization
            to carry out.
        sampler: A sampler to execute circuits.
    """
    options: LocalXEBPhasedFSimCalibrationOptions = calibration.options
    circuit = cirq.Circuit([calibration.gate.on(*pair) for pair in calibration.pairs])

    # 2. Set up XEB experiment
    cycle_depths = options.cycle_depths
    circuits = rqcg.generate_library_of_2q_circuits(
        n_library_circuits=options.n_library_circuits,
        two_qubit_gate=calibration.gate,
        max_cycle_depth=max(cycle_depths),
    )
    combs_by_layer = rqcg.get_random_combinations_for_layer_circuit(
        n_library_circuits=len(circuits),
        n_combinations=options.n_combinations,
        layer_circuit=circuit,
    )

    # 3. Sample data
    sampled_df = xebsamp.sample_2q_xeb_circuits(
        sampler=sampler,
        circuits=circuits,
        cycle_depths=cycle_depths,
        combinations_by_layer=combs_by_layer,
    )

    # 4. Initial fidelities
    # initial_fids = xebf.benchmark_2q_xeb_fidelities(
    #     sampled_df=sampled_df,
    #     circuits=circuits,
    #     cycle_depths=cycle_depths,
    # )

    # 5. Characterize by fitting angles.
    if options.fsim_options.defaults_set():
        fsim_options = options.fsim_options
    else:
        fsim_options = options.fsim_options.with_defaults_from_gate(calibration.gate)

    pcircuits = [xebf.parameterize_circuit(circuit, fsim_options) for circuit in circuits]
    fatol = options.fatol if options.fatol is not None else 5e-3
    xatol = options.xatol if options.xatol is not None else 5e-3
    with _maybe_multiprocessing_pool(n_processes=options.n_processes) as pool:
        char_results = xebf.characterize_phased_fsim_parameters_with_xeb_by_pair(
            sampled_df=sampled_df,
            parameterized_circuits=pcircuits,
            cycle_depths=cycle_depths,
            options=fsim_options,
            pool=pool,
            fatol=fatol,
            xatol=xatol,
        )

    return PhasedFSimCalibrationResult(
        parameters={
            pair: PhasedFSimCharacterization(**param_dict)
            for pair, param_dict in char_results.final_params.items()
        },
        gate=calibration.gate,
        options=options,
    )
