# Copyright 2024 The Cirq Developers
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

"""Provides a method to do z-phase calibration for fermionic gates."""
from typing import Optional, Sequence, Union, ContextManager, TYPE_CHECKING
import multiprocessing
import concurrent.futures

import numpy as np

from cirq.experiments import xeb_fitting
from cirq.experiments.two_qubit_xeb import _parallel_two_qubit_xeb
from cirq import ops

if TYPE_CHECKING:
  import cirq

def z_phase_calibration_workflow(
    sampler: 'cirq.Sampler',
    q0: 'cirq.GridQubit',
    q1: 'cirq.GridQubit',
    two_qubit_gate: 'cirq.Gate' = ops.CZ,
    options: Optional[xeb_fitting.XEBPhasedFSimCharacterizationOptions] = None,
    n_repetitions: int = 10**4,
    n_combinations: int = 10,
    n_circuits: int = 20,
    cycle_depths: Sequence[int] = tuple(np.arange(3, 100, 20)),
    random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
    atol: float = 1e-3,
    pool: Optional[Union[multiprocessing.Pool, concurrent.futures.ThreadPoolExecutor]] = None,
):
    fids_df_0, circuits, sampled_df = _parallel_two_qubit_xeb(
        sampler=sampler,
        qubits=(q0, q1),
        entangling_gate=two_qubit_gate,
        n_repetitions=n_repetitions,
        cycle_depths=cycle_depths,
        n_circuits=n_circuits,
        n_combinations=n_combinations,
        random_state=random_state,
    )

    if options is None:
        options = xeb_fitting.XEBPhasedFSimCharacterizationOptions().with_defaults_from_gate(
            two_qubit_gate
        )

    p_circuits = [xeb_fitting.parameterize_circuit(circuit, options) for circuit in circuits]

    result = xeb_fitting.characterize_phased_fsim_parameters_with_xeb_by_pair(
        sampled_df=sampled_df,
        parameterized_circuits=p_circuits,
        cycle_depths=cycle_depths,
        options=options,
        fatol=atol,
        xatol=atol,
        pool=pool,
    )

    return result, xeb_fitting.before_and_after_characterization(
        fids_df_0, characterization_result=result
    )