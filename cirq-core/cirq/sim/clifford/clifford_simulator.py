# Copyright 2019 The Cirq Developers
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

"""An efficient simulator for Clifford circuits.

Allowed operations include:
	- X,Y,Z,H,S,CNOT,CZ
	- measurements in the computational basis

The quantum state is specified in two forms:
    1. In terms of stabilizer generators. These are a set of n Pauli operators
    {S_1,S_2,...,S_n} such that S_i |psi> = |psi>.

    This implementation is based on Aaronson and Gottesman,
    2004 (arXiv:quant-ph/0406196).

    2. In the CH-form defined by Bravyi et al, 2018 (arXiv:1808.00128).
    This representation keeps track of overall phase and enables access
    to state vector amplitudes.
"""

from typing import Any, Dict, List, Sequence, Union

import numpy as np

import cirq
from cirq import protocols, value
from cirq.protocols import act_on
from cirq.sim import clifford, simulator_base


class CliffordSimulator(
    simulator_base.SimulatorBase[
        'cirq.CliffordSimulatorStepResult',
        'cirq.CliffordTrialResult',
        'cirq.StabilizerChFormSimulationState',
    ]
):
    """An efficient simulator for Clifford circuits."""

    def __init__(
        self, seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None, split_untangled_states: bool = False
    ):
        """Creates instance of `CliffordSimulator`.

        Args:
            seed: The random seed to use for this simulator.
            split_untangled_states: Optimizes simulation by running separable
                states independently and merging those states at the end.
        """
        self.init = True
        super().__init__(seed=seed, split_untangled_states=split_untangled_states)

    @staticmethod
    def is_supported_operation(op: 'cirq.Operation') -> bool:
        """Checks whether given operation can be simulated by this simulator."""
        # TODO: support more general Pauli measurements
        return protocols.has_stabilizer_effect(op)

    def _create_partial_simulation_state(
        self,
        initial_state: Union[int, 'cirq.StabilizerChFormSimulationState'],
        qubits: Sequence['cirq.Qid'],
        classical_data: 'cirq.ClassicalDataStore',
    ) -> 'cirq.StabilizerChFormSimulationState':
        """Creates the StabilizerChFormSimulationState for a circuit.

        Args:
            initial_state: The initial state for the simulation in the
                computational basis. Represented as a big endian int.
            qubits: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            logs: A log of the results of measurement that is added to.
            classical_data: The shared classical data container for this
                simulation.

        Returns:
            StabilizerChFormSimulationState for the circuit.
        """
        if isinstance(initial_state, clifford.StabilizerChFormSimulationState):
            return initial_state

        return clifford.StabilizerChFormSimulationState(
            prng=self._prng,
            classical_data=classical_data,
            qubits=qubits,
            initial_state=initial_state,
        )

    def _create_step_result(
        self, sim_state: 'cirq.SimulationStateBase[clifford.StabilizerChFormSimulationState]'
    ):
        return CliffordSimulatorStepResult(sim_state=sim_state)

    def _create_simulator_trial_result(
        self,
        params: 'cirq.ParamResolver',
        measurements: Dict[str, np.ndarray],
        final_simulator_state: 'cirq.SimulationStateBase[cirq.StabilizerChFormSimulationState]',
    ):
        return CliffordTrialResult(
            params=params, measurements=measurements, final_simulator_state=final_simulator_state
        )


class CliffordTrialResult(
    simulator_base.SimulationTrialResultBase['clifford.StabilizerChFormSimulationState']
):
    def __init__(
        self,
        params: 'cirq.ParamResolver',
        measurements: Dict[str, np.ndarray],
        final_simulator_state: 'cirq.SimulationStateBase[cirq.StabilizerChFormSimulationState]',
    ) -> None:
        super().__init__(
            params=params, measurements=measurements, final_simulator_state=final_simulator_state
        )

    @property
    def final_state(self) -> 'cirq.CliffordState':
        state = self._get_merged_sim_state()
        clifford_state = CliffordState(state.qubit_map)
        clifford_state.ch_form = state.state.copy()
        return clifford_state

    def __str__(self) -> str:
        samples = super().__str__()
        final = self._get_merged_sim_state().state
        return f'measurements: {samples}\noutput state: {final}'

    def _repr_pretty_(self, p: Any, cycle: bool):
        """iPython (Jupyter) pretty print."""
        p.text("cirq.CliffordTrialResult(...)" if cycle else self.__str__())


class CliffordSimulatorStepResult(
    simulator_base.StepResultBase['cirq.StabilizerChFormSimulationState']
):
    """A `StepResult` that includes `StateVectorMixin` methods."""

    def __init__(
        self, sim_state: 'cirq.SimulationStateBase[clifford.StabilizerChFormSimulationState]'
    ):
        """Results of a step of the simulator.
        Attributes:
            sim_state: The qubit:SimulationState lookup for this step.
        """
        super().__init__(sim_state)
        self._clifford_state = None

    def __str__(self) -> str:
        def bitstring(vals):
            return ''.join('1' if v else '0' for v in vals)

        results = sorted([(key, bitstring(val)) for key, val in self.measurements.items()])

        if len(results) == 0:
            measurements = ''
        else:
            measurements = ' '.join([f'{key}={val}' for key, val in results]) + '\n'

        final = self.state

        return f'{measurements}{final}'

    def _repr_pretty_(self, p, cycle):
        """iPython (Jupyter) pretty print."""
        p.text("cirq.CliffordSimulatorStateResult(...)" if cycle else self.__str__())

    @property
    def state(self):
        if self._clifford_state is None:
            clifford_state = CliffordState(self._qubit_mapping)
            clifford_state.ch_form = self._merged_sim_state.state.copy()
            self._clifford_state = clifford_state
        return self._clifford_state


@value.value_equality
class CliffordState:
    """A state of the Clifford simulation.

    The state is stored using Bravyi's CH-form which allows access to the full
    state vector (including phase).

    Gates and measurements are applied to each representation in O(n^2) time.
    """

    def __init__(self, qubit_map, initial_state: Union[int, 'cirq.StabilizerStateChForm'] = 0):
        self.qubit_map = qubit_map
        self.n = len(qubit_map)

        self.ch_form = (
            initial_state
            if isinstance(initial_state, clifford.StabilizerStateChForm)
            else clifford.StabilizerStateChForm(self.n, initial_state)
        )

    def _json_dict_(self):
        return {'qubit_map': [(k, v) for k, v in self.qubit_map.items()], 'ch_form': self.ch_form}

    @classmethod
    def _from_json_dict_(cls, qubit_map, ch_form, **kwargs):
        state = cls(dict(qubit_map))
        state.ch_form = ch_form

        return state

    def _value_equality_values_(self) -> Any:
        return self.qubit_map, self.ch_form

    def copy(self) -> 'cirq.CliffordState':
        state = CliffordState(self.qubit_map)
        state.ch_form = self.ch_form.copy()

        return state

    def __repr__(self) -> str:
        return repr(self.ch_form)

    def __str__(self) -> str:
        """Return the state vector string representation of the state."""
        return str(self.ch_form)

    def to_numpy(self) -> np.ndarray:
        return self.ch_form.to_state_vector()

    def state_vector(self):
        return self.ch_form.state_vector()

    def apply_unitary(self, op: 'cirq.Operation'):
        ch_form_args = clifford.StabilizerChFormSimulationState(
            prng=np.random.RandomState(), qubits=self.qubit_map.keys(), initial_state=self.ch_form
        )
        try:
            act_on(op, ch_form_args)
        except TypeError:
            raise ValueError(f'{op.gate} cannot be run with Clifford simulator.')
        return

    def apply_measurement(
        self,
        op: 'cirq.Operation',
        measurements: Dict[str, List[int]],
        prng: np.random.RandomState,
        collapse_state_vector=True,
    ):
        if not isinstance(op.gate, cirq.MeasurementGate):
            raise TypeError(
                f'apply_measurement only supports cirq.MeasurementGate operations. '
                f'Found {op.gate} instead.'
            )

        if collapse_state_vector:
            state = self
        else:
            state = self.copy()

        classical_data = value.ClassicalDataDictionaryStore()
        ch_form_args = clifford.StabilizerChFormSimulationState(
            prng=prng,
            classical_data=classical_data,
            qubits=self.qubit_map.keys(),
            initial_state=state.ch_form,
        )
        act_on(op, ch_form_args)
        measurements.update({str(k): list(v[-1]) for k, v in classical_data.records.items()})
