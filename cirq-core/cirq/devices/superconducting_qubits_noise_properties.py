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


"""Class for representing noise on a superconducting qubit device."""

import abc
from dataclasses import dataclass, field
from typing import Dict, TYPE_CHECKING, List, Set, Type

from cirq import _compat, ops, devices
from cirq.devices import noise_utils

if TYPE_CHECKING:
    import cirq


# TODO: missing per-device defaults
# Type-ignored because mypy cannot handle abstract dataclasses:
# https://github.com/python/mypy/issues/5374
@dataclass  # type: ignore
class SuperconductingQubitsNoiseProperties(devices.NoiseProperties, abc.ABC):
    """Noise-defining properties for a superconducting-qubit-based device.

    Args:
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
    """

    gate_times_ns: Dict[type, float]
    t1_ns: Dict['cirq.Qid', float]
    tphi_ns: Dict['cirq.Qid', float]
    readout_errors: Dict['cirq.Qid', List[float]]
    gate_pauli_errors: Dict[noise_utils.OpIdentifier, float]

    validate: bool = True
    _qubits: List['cirq.Qid'] = field(init=False, default_factory=list)

    def __post_init__(self):
        if not self.validate:
            return
        t1_qubits = set(self.t1_ns)
        tphi_qubits = set(self.tphi_ns)

        if t1_qubits != tphi_qubits:
            raise ValueError('Keys specified for T1 and Tphi are not identical.')

        # validate two qubit gate errors.
        self._validate_symmetric_errors('gate_pauli_errors')

    def _validate_symmetric_errors(self, field_name: str) -> None:
        gate_error_dict = getattr(self, field_name)
        for op_id in gate_error_dict:
            if len(op_id.qubits) != 2:
                # single qubit op_ids also present, or generic values are
                # specified. Skip these cases
                if len(op_id.qubits) > 2:
                    raise ValueError(
                        f'Found gate {op_id.gate_type} with {len(op_id.qubits)} qubits. '
                        'Symmetric errors can only apply to 2-qubit gates.'
                    )
            elif op_id.gate_type in self.symmetric_two_qubit_gates():
                op_id_swapped = noise_utils.OpIdentifier(op_id.gate_type, *op_id.qubits[::-1])
                if op_id_swapped not in gate_error_dict:
                    raise ValueError(
                        f'Operation {op_id} of field {field_name} has errors '
                        f'but its symmetric id {op_id_swapped} does not.'
                    )
            elif op_id.gate_type not in self.asymmetric_two_qubit_gates():
                # Asymmetric gates do not require validation.
                raise ValueError(
                    f'Found gate {op_id.gate_type} which does not appear in the '
                    'symmetric or asymmetric gate sets.'
                )

    @property
    def qubits(self) -> List['cirq.Qid']:
        """Qubits for which we have data"""
        if not self._qubits:
            self._qubits = sorted(self.t1_ns)
        return self._qubits

    @classmethod
    @abc.abstractmethod
    def single_qubit_gates(cls) -> Set[Type[ops.Gate]]:
        """Returns the set of single-qubit gates this class supports."""

    @classmethod
    @abc.abstractmethod
    def symmetric_two_qubit_gates(cls) -> Set[Type[ops.Gate]]:
        """Returns the set of symmetric two-qubit gates this class supports."""

    @classmethod
    @abc.abstractmethod
    def asymmetric_two_qubit_gates(cls) -> Set[Type[ops.Gate]]:
        """Returns the set of asymmetric two-qubit gates this class supports."""

    @classmethod
    def two_qubit_gates(cls) -> Set[Type[ops.Gate]]:
        """Returns the set of all two-qubit gates this class supports."""
        return cls.symmetric_two_qubit_gates() | cls.asymmetric_two_qubit_gates()

    @classmethod
    def expected_gates(cls) -> Set[Type[ops.Gate]]:
        """Returns the set of all gates this class supports."""
        return cls.single_qubit_gates() | cls.two_qubit_gates()

    def _get_pauli_error(self, p_error: float, op_id: noise_utils.OpIdentifier):
        time_ns = float(self.gate_times_ns[op_id.gate_type])
        for q in op_id.qubits:
            p_error -= noise_utils.decoherence_pauli_error(self.t1_ns[q], self.tphi_ns[q], time_ns)
        return p_error

    @_compat.cached_property
    def _depolarizing_error(self) -> Dict[noise_utils.OpIdentifier, float]:
        """Returns the portion of Pauli error from depolarization."""
        depol_errors = {}
        for op_id, p_error in self.gate_pauli_errors.items():
            gate_type = op_id.gate_type
            if issubclass(gate_type, ops.MeasurementGate):
                # Non-measurement error can be ignored on measurement gates.
                continue
            expected_qubits = 1 if gate_type in self.single_qubit_gates() else 2
            if len(op_id.qubits) != expected_qubits:
                raise ValueError(
                    f'Gate {gate_type} takes {expected_qubits} qubit(s), '
                    f'but {op_id.qubits} were given.'
                )
            depol_errors[op_id] = self._get_pauli_error(p_error, op_id)
        return depol_errors

    def build_noise_models(self) -> List['cirq.NoiseModel']:
        noise_models: List['cirq.NoiseModel'] = []

        if self.t1_ns:
            noise_models.append(
                devices.ThermalNoiseModel(
                    set(self.t1_ns.keys()),
                    self.gate_times_ns,
                    cool_rate_GHz={q: 1 / T1 for q, T1 in self.t1_ns.items()},
                    dephase_rate_GHz={q: 1 / Tp for q, Tp in self.tphi_ns.items()},
                )
            )

        depolarizing_error = self._depolarizing_error
        added_pauli_errors = {
            op_id: ops.depolarize(p_error, len(op_id.qubits)).on(*op_id.qubits)
            for op_id, p_error in depolarizing_error.items()
            if p_error > 0
        }

        # This adds per-qubit pauli error after ops on those qubits.
        noise_models.append(devices.InsertionNoiseModel(ops_added=added_pauli_errors))

        # This adds per-qubit measurement error BEFORE measurement on those qubits.
        if self.readout_errors:
            added_measure_errors: Dict[noise_utils.OpIdentifier, 'cirq.Operation'] = {}
            for qubit in self.readout_errors:
                p_00, p_11 = self.readout_errors[qubit]
                p = p_11 / (p_00 + p_11)
                gamma = p_11 / p
                added_measure_errors[
                    noise_utils.OpIdentifier(ops.MeasurementGate, qubit)
                ] = ops.generalized_amplitude_damp(p, gamma).on(qubit)

            noise_models.append(
                devices.InsertionNoiseModel(ops_added=added_measure_errors, prepend=True)
            )

        return noise_models
