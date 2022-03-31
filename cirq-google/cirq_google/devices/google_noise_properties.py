# Copyright 2022 The Cirq Developers
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


"""Class for representing noise on a Google device."""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Type
import numpy as np

import cirq, cirq_google
from cirq import _compat, devices
from cirq.devices import noise_utils
from cirq.transformers.heuristic_decompositions import gate_tabulation_math_utils


SINGLE_QUBIT_GATES: Set[Type['cirq.Gate']] = {
    cirq.ZPowGate,
    cirq.PhasedXZGate,
    cirq.MeasurementGate,
    cirq.ResetChannel,
}
SYMMETRIC_TWO_QUBIT_GATES: Set[Type['cirq.Gate']] = {
    cirq_google.SycamoreGate,
    cirq.FSimGate,
    cirq.PhasedFSimGate,
    cirq.ISwapPowGate,
    cirq.CZPowGate,
}
ASYMMETRIC_TWO_QUBIT_GATES: Set[Type['cirq.Gate']] = set()


@dataclass
class GoogleNoiseProperties(devices.SuperconductingQubitsNoiseProperties):
    """Noise-defining properties for a Google device.

    Inherited args:
        gate_times_ns: Dict[type, float] of gate types to their duration on
            quantum hardware. Used with t(1|phi)_ns to specify thermal noise.
        t1_ns: Dict[cirq.Qid, float] of qubits to their T_1 time, in ns.
        tphi_ns: Dict[cirq.Qid, float] of qubits to their T_phi time, in ns.
        readout_errors: Dict[cirq.Qid, np.ndarray] of qubits to their readout
            errors in matrix form: [P(read |1> from |0>), P(read |0> from |1>)].
            Used to prepend amplitude damping errors to measurements.
        gate_pauli_errors: dict of noise_utils.OpIdentifiers (a gate and the qubits it
            targets) to the Pauli error for that operation. Used to construct
            depolarizing error. Keys in this dict must have defined qubits.
        validate: If True, verifies that t1 and tphi qubits sets match, and
            that all symmetric two-qubit gates have errors which are
            symmetric over the qubits they affect. Defaults to True.

    Additional args:
        fsim_errors: Dict[noise_utils.OpIdentifier, cirq.PhasedFSimGate] of gate types
            (potentially on specific qubits) to the PhasedFSim fix-up operation
            for that gate. Defaults to no-op for all gates.
    """

    fsim_errors: Dict[noise_utils.OpIdentifier, cirq.PhasedFSimGate] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()

        # validate two qubit gate errors.
        self._validate_symmetric_errors('fsim_errors')

    @classmethod
    def single_qubit_gates(cls) -> Set[type]:
        return SINGLE_QUBIT_GATES

    @classmethod
    def symmetric_two_qubit_gates(cls) -> Set[type]:
        return SYMMETRIC_TWO_QUBIT_GATES

    @classmethod
    def asymmetric_two_qubit_gates(cls) -> Set[type]:
        return ASYMMETRIC_TWO_QUBIT_GATES

    @_compat.cached_property
    def _depolarizing_error(self) -> Dict[noise_utils.OpIdentifier, float]:
        depol_errors = super()._depolarizing_error

        def extract_entangling_error(match_id: noise_utils.OpIdentifier):
            """Gets the entangling error component of depol_errors[match_id]."""
            unitary_err = cirq.unitary(self.fsim_errors[match_id])
            fid = gate_tabulation_math_utils.unitary_entanglement_fidelity(unitary_err, np.eye(4))
            return 1 - fid

        # Subtract entangling angle error.
        for op_id in depol_errors:
            if op_id.gate_type not in self.two_qubit_gates():
                continue
            if op_id in self.fsim_errors:
                depol_errors[op_id] -= extract_entangling_error(op_id)
                continue
            # Find the closest matching supertype, if one is provided.
            # Gateset has similar behavior, but cannot be used here
            # because depol_errors is a dict, not a set.
            match_id = None
            candidate_parents = [
                parent_id for parent_id in self.fsim_errors if op_id.is_proper_subtype_of(parent_id)
            ]
            for parent_id in candidate_parents:
                if match_id is None or parent_id.is_proper_subtype_of(match_id):
                    match_id = parent_id
            if match_id is not None:
                depol_errors[op_id] -= extract_entangling_error(match_id)

        return depol_errors

    def build_noise_models(self) -> List['cirq.NoiseModel']:
        """Construct all NoiseModels associated with NoiseProperties."""
        noise_models = super().build_noise_models()

        # Insert entangling gate coherent errors after thermal error.
        if self.fsim_errors:
            fsim_ops = {op_id: gate.on(*op_id.qubits) for op_id, gate in self.fsim_errors.items()}
            noise_models.insert(1, devices.InsertionNoiseModel(ops_added=fsim_ops))

        return noise_models


class NoiseModelFromGoogleNoiseProperties(devices.NoiseModelFromNoiseProperties):
    """A noise model defined from noise properties of a Google device."""

    def is_virtual(self, op: cirq.Operation) -> bool:
        return isinstance(op.gate, cirq.ZPowGate) and cirq_google.PhysicalZTag not in op.tags

    # noisy_moments is implemented by the superclass.
