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

from dataclasses import dataclass, field
from typing import Dict, TYPE_CHECKING, List, Set

from cirq import ops, protocols, devices
from cirq.devices.noise_utils import (
    OpIdentifier,
    decoherence_pauli_error,
)

if TYPE_CHECKING:
    import cirq

SINGLE_QUBIT_GATES: Set[type] = {
    ops.ZPowGate,
    ops.PhasedXZGate,
    ops.MeasurementGate,
    ops.ResetChannel,
}
SYMMETRIC_TWO_QUBIT_GATES: Set[type] = {
    ops.FSimGate,
    ops.PhasedFSimGate,
    ops.ISwapPowGate,
    ops.CZPowGate,
}
ASYMMETRIC_TWO_QUBIT_GATES: Set[type] = set()
TWO_QUBIT_GATES = SYMMETRIC_TWO_QUBIT_GATES | ASYMMETRIC_TWO_QUBIT_GATES


# TODO: missing per-device defaults
@dataclass
class SuperconductingQubitsNoiseProperties(devices.NoiseProperties):
    """Noise-defining properties for a quantum device.

    Args:
        gate_times_ns: Dict[type, float] of gate types to their duration on
            quantum hardware.
        t1_ns: Dict[cirq.Qid, float] of qubits to their T_1 time, in ns.
        tphi_ns: Dict[cirq.Qid, float] of qubits to their T_phi time, in ns.
        ro_fidelities: Dict[cirq.Qid, np.ndarray] of qubits to their readout
            fidelity matrix.
        gate_pauli_errors: Dict[OpIdentifier, float] of gate types
            (potentially on specific qubits) to the Pauli error for that gate.
        validate: If True, performs validation on input arguments. Defaults
            to True.
    """

    gate_times_ns: Dict[type, float]
    t1_ns: Dict['cirq.Qid', float]
    tphi_ns: Dict['cirq.Qid', float]
    ro_fidelities: Dict['cirq.Qid', List[float]]
    gate_pauli_errors: Dict[OpIdentifier, float]

    validate: bool = True
    _qubits: List['cirq.Qid'] = field(init=False, default_factory=list)
    _depolarizing_error: Dict[OpIdentifier, float] = field(init=False, default_factory=dict)

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
            elif op_id.gate_type not in self.two_qubit_gates():
                raise ValueError(
                    f'Found gate {op_id.gate_type} which does not appear in the '
                    'symmetric or asymmetric gate sets.'
                )
            else:
                # TODO: this assumes op is symmetric.
                # If asymmetric gates are added, we will need to update it.
                op_id_swapped = OpIdentifier(op_id.gate_type, *op_id.qubits[::-1])
                if op_id_swapped not in gate_error_dict:
                    raise ValueError(
                        f'Operation {op_id} of field {field_name} has errors '
                        f'but its symmetric id {op_id_swapped} does not.'
                    )

    @property
    def qubits(self) -> List['cirq.Qid']:
        """Qubits for which we have data"""
        if not self._qubits:
            self._qubits = sorted(self.t1_ns)
        return self._qubits

    @classmethod
    def single_qubit_gates(cls) -> Set[type]:
        return SINGLE_QUBIT_GATES

    @classmethod
    def two_qubit_gates(cls) -> Set[type]:
        return TWO_QUBIT_GATES

    @classmethod
    def expected_gates(cls) -> Set[type]:
        return cls.single_qubit_gates() | cls.two_qubit_gates()

    def get_depolarizing_error(self) -> Dict[OpIdentifier, float]:
        """Returns the portion of Pauli error from depolarization.

        The result of this method is memoized.
        """
        if self._depolarizing_error:
            return self._depolarizing_error

        depol_errors = {}
        for op_id, p_error in self.gate_pauli_errors.items():
            gate_type = op_id.gate_type
            if gate_type in self.single_qubit_gates():
                if issubclass(gate_type, ops.MeasurementGate):
                    # Non-measurement error can be ignored on measurement gates.
                    continue
                if len(op_id.qubits) != 1:
                    raise ValueError(
                        f'Gate {gate_type} only takes one qubit, but {op_id.qubits} were given.'
                    )
                time_ns = float(self.gate_times_ns[gate_type])
                q0 = op_id.qubits[0]
                # Subtract decoherence error.
                if q0 in self.t1_ns:
                    p_error -= decoherence_pauli_error(self.t1_ns[q0], self.tphi_ns[q0], time_ns)

            else:
                # This must be a 2-qubit gate.
                if gate_type not in self.two_qubit_gates():
                    raise ValueError(f'Gate {gate_type} is not in the supported gate list.')
                if len(op_id.qubits) != 2:
                    raise ValueError(
                        f'Gate {gate_type} takes two qubits, but {op_id.qubits} were given.'
                    )
                time_ns = float(self.gate_times_ns[gate_type])
                # Subtract decoherence error.
                q0, q1 = op_id.qubits
                if q0 in self.t1_ns:
                    p_error -= decoherence_pauli_error(self.t1_ns[q0], self.tphi_ns[q0], time_ns)
                if q1 in self.t1_ns:
                    p_error -= decoherence_pauli_error(self.t1_ns[q1], self.tphi_ns[q1], time_ns)

            depol_errors[op_id] = p_error
        # memoization is OK
        self._depolarizing_error = depol_errors
        return self._depolarizing_error

    def build_noise_models(self) -> List['cirq.NoiseModel']:
        noise_models: List['cirq.NoiseModel'] = []

        if set(self.t1_ns) != set(self.tphi_ns):
            raise ValueError(
                f'T1 data has qubits {set(self.t1_ns)}, but Tphi has qubits {set(self.tphi_ns)}.'
            )
        if self.t1_ns:  # level 1 sophistication
            noise_models.append(
                devices.ThermalNoiseModel(
                    set(self.t1_ns.keys()),
                    self.gate_times_ns,
                    cool_rate_GHz={q: 1 / T1 for q, T1 in self.t1_ns.items()},
                    dephase_rate_GHz={q: 1 / Tp for q, Tp in self.tphi_ns.items()},
                )
            )

        gate_types = set(op_id.gate_type for op_id in self.gate_pauli_errors)
        if not gate_types.issubset(self.expected_gates()):
            raise ValueError(
                'Some gates are not in the supported set.'
                f'\nGates: {gate_types}\nSupported: {self.expected_gates()}'
            )

        depolarizing_error = self.get_depolarizing_error()
        added_pauli_errors = {
            # TODO: handle op_id not having qubits.
            op_id: ops.depolarize(p_error, len(op_id.qubits)).on(*op_id.qubits)
            for op_id, p_error in depolarizing_error.items()
            if p_error > 0
        }

        # This adds per-qubit pauli error after ops on those qubits.
        noise_models.append(devices.InsertionNoiseModel(ops_added=added_pauli_errors))

        # This adds per-qubit measurement error BEFORE measurement on those qubits.
        if self.ro_fidelities:
            added_measure_errors: Dict[OpIdentifier, 'cirq.Operation'] = {}
            for qubit in self.ro_fidelities:
                p_00, p_11 = self.ro_fidelities[qubit]
                p = p_11 / (p_00 + p_11)
                gamma = p_11 / p
                added_measure_errors[
                    OpIdentifier(ops.MeasurementGate, qubit)
                ] = ops.generalized_amplitude_damp(p, gamma).on(qubit)

            noise_models.append(
                devices.InsertionNoiseModel(ops_added=added_measure_errors, prepend=True)
            )

        return noise_models

    def __str__(self) -> str:
        return 'SuperconductingQubitsNoiseProperties'

    def __repr__(self) -> str:
        args = []
        gate_times_repr = ', '.join(
            f'{key.__module__}.{key.__qualname__}: {val}' for key, val in self.gate_times_ns.items()
        )
        args.append(f'    gate_times_ns={{{gate_times_repr}}}')
        args.append(f'    t1_ns={self.t1_ns!r}')
        args.append(f'    tphi_ns={self.tphi_ns!r}')
        args.append(f'    ro_fidelities={self.ro_fidelities!r}')
        args.append(f'    gate_pauli_errors={self.gate_pauli_errors!r}')
        args_str = ',\n'.join(args)
        return f'cirq.SuperconductingQubitsNoiseProperties(\n{args_str}\n)'

    def _json_dict_(self):
        storage_gate_times = {
            protocols.json_cirq_type(key): val for key, val in self.gate_times_ns.items()
        }
        return {
            # JSON requires mappings to have keys of basic types.
            # Pairs must be sorted to ensure consistent serialization.
            'gate_times_ns': sorted(storage_gate_times.items(), key=str),
            't1_ns': sorted(self.t1_ns.items()),
            'tphi_ns': sorted(self.tphi_ns.items()),
            'ro_fidelities': sorted(self.ro_fidelities.items()),
            'gate_pauli_errors': sorted(self.gate_pauli_errors.items(), key=str),
            'validate': self.validate,
        }

    @classmethod
    def _from_json_dict_(
        cls, gate_times_ns, t1_ns, tphi_ns, ro_fidelities, gate_pauli_errors, validate, **kwargs
    ):
        gate_type_times = {protocols.cirq_type_from_json(gate): val for gate, val in gate_times_ns}
        return SuperconductingQubitsNoiseProperties(
            gate_times_ns=gate_type_times,
            t1_ns=dict(t1_ns),
            tphi_ns=dict(tphi_ns),
            ro_fidelities=dict(ro_fidelities),
            gate_pauli_errors=dict(gate_pauli_errors),
            validate=validate,
        )
