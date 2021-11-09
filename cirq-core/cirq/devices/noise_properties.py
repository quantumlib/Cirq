# pylint: disable=wrong-or-nonexistent-copyright-notice
from dataclasses import dataclass, field
from typing import Dict, Iterable, Sequence, TYPE_CHECKING, List, Set

from cirq import ops, protocols, devices
from cirq.devices.noise_utils import (
    OpIdentifier,
    PHYSICAL_GATE_TAG,
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


@dataclass
class NoiseProperties:
    """Noise-defining properties for a quantum device.

    Args:
        gate_times_ns: Dict[type, float] of gate types to their duration on
            Google hardware.
        T1_ns: Dict[cirq.Qid, float] of qubits to their T1 time, in ns.
        Tphi_ns: Dict[cirq.Qid, float] of qubits to their Tphi time, in ns.
        ro_fidelities: Dict[cirq.Qid, np.ndarray] of qubits to their readout
            fidelity matrix.
        gate_pauli_errors: Dict[OpIdentifier, float] of gate types
            (potentially on specific qubits) to the Pauli error for that gate.
        validate: If True, performs validation on input arguments. Defaults
            to True.
    """

    gate_times_ns: Dict[type, float]
    T1_ns: Dict['cirq.Qid', float]
    Tphi_ns: Dict['cirq.Qid', float]
    ro_fidelities: Dict['cirq.Qid', List[float]]
    gate_pauli_errors: Dict[OpIdentifier, float]

    validate: bool = True
    _qubits: List['cirq.Qid'] = field(init=False, default_factory=list)
    _depolarizing_error: Dict[OpIdentifier, float] = field(init=False, default_factory=dict)

    def __post_init__(self):
        if not self.validate:
            return
        t1_qubits = set(self.T1_ns)
        tphi_qubits = set(self.Tphi_ns)

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
                op_id_swapped = op_id.swapped()
                if op_id_swapped not in gate_error_dict:
                    raise ValueError(
                        f'Operation {op_id} of field {field_name} has errors '
                        f'but its symmetric id {op_id_swapped} does not.'
                    )

    @property
    def qubits(self) -> List['cirq.Qid']:
        """Qubits for which we have data"""
        if not self._qubits:
            self._qubits = sorted(self.T1_ns)
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

    @classmethod
    def canonical_gates(cls) -> Dict[type, 'cirq.Gate']:
        return {
            ops.ZPowGate: ops.ZPowGate(),
            ops.PhasedXZGate: ops.PhasedXZGate(x_exponent=0, z_exponent=0, axis_phase_exponent=0),
            ops.MeasurementGate: ops.MeasurementGate(num_qubits=1),
            ops.ResetChannel: ops.ResetChannel(),
            ops.FSimGate: ops.FSimGate(theta=0, phi=0),
            ops.PhasedFSimGate: ops.PhasedFSimGate(theta=0),
            ops.ISwapPowGate: ops.ISwapPowGate(),
            ops.CZPowGate: ops.CZPowGate(),
        }

    @classmethod
    def get_canonical_gate(cls, gate_type: type) -> 'cirq.Gate':
        return cls.canonical_gates()[gate_type]

    @classmethod
    def _identifier_to_op(cls, op_id: OpIdentifier) -> 'cirq.Operation':
        return cls.get_canonical_gate(op_id.gate_type).on(*op_id.qubits)

    @classmethod
    def _op_to_identifier(cls, op: 'cirq.Operation'):
        return OpIdentifier(type(op.gate), *op.qubits)

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
                if q0 in self.T1_ns:
                    p_error -= decoherence_pauli_error(self.T1_ns[q0], self.Tphi_ns[q0], time_ns)

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
                if q0 in self.T1_ns:
                    p_error -= decoherence_pauli_error(self.T1_ns[q0], self.Tphi_ns[q0], time_ns)
                if q1 in self.T1_ns:
                    p_error -= decoherence_pauli_error(self.T1_ns[q1], self.Tphi_ns[q1], time_ns)

            depol_errors[op_id] = p_error
        # memoization is OK
        self._depolarizing_error = depol_errors
        return self._depolarizing_error

    def build_noise_models(self) -> List['cirq.NoiseModel']:
        """Construct all NoiseModels associated with this NoiseProperties."""
        noise_models: List['cirq.NoiseModel'] = []

        if set(self.T1_ns) != set(self.Tphi_ns):
            raise ValueError(
                f'T1 data has qubits {set(self.T1_ns)}, but Tphi has qubits {set(self.Tphi_ns)}.'
            )
        if self.T1_ns:  # level 1 sophistication
            noise_models.append(
                devices.ThermalNoiseModel(
                    {q: 2 for q in self.T1_ns},
                    self.gate_times_ns,
                    cool_rate_GHz={q: 1 / T1 for q, T1 in self.T1_ns.items()},
                    dephase_rate_GHz={q: 1 / Tp for q, Tp in self.Tphi_ns.items()},
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
        return 'NoiseProperties'

    def __repr__(self) -> str:
        args = []
        gate_times_repr = ', '.join(
            f'{key.__module__}.{key.__qualname__}: {val}' for key, val in self.gate_times_ns.items()
        )
        args.append(f'    gate_times_ns={{{gate_times_repr}}}')
        args.append(f'    T1_ns={self.T1_ns!r}')
        args.append(f'    Tphi_ns={self.Tphi_ns!r}')
        args.append(f'    ro_fidelities={self.ro_fidelities!r}')
        args.append(f'    gate_pauli_errors={self.gate_pauli_errors!r}')
        args_str = ',\n'.join(args)
        return f'cirq.NoiseProperties(\n{args_str}\n)'

    def _json_dict_(self):
        storage_gate_times = {
            self.get_canonical_gate(key): val for key, val in self.gate_times_ns.items()
        }
        storage_pauli_errors = {
            self._identifier_to_op(op_id): val for op_id, val in self.gate_pauli_errors.items()
        }
        return {
            'cirq_type': 'NoiseProperties',
            # JSON requires mappings to have keys of basic types.
            # Pairs must be sorted to ensure consistent serialization.
            'gate_times_ns': sorted(storage_gate_times.items(), key=str),
            'T1_ns': sorted(self.T1_ns.items()),
            'Tphi_ns': sorted(self.Tphi_ns.items()),
            'ro_fidelities': sorted(self.ro_fidelities.items()),
            'gate_pauli_errors': sorted(storage_pauli_errors.items(), key=str),
            'validate': self.validate,
        }

    @classmethod
    def _from_json_dict_(
        cls, gate_times_ns, T1_ns, Tphi_ns, ro_fidelities, gate_pauli_errors, validate, **kwargs
    ):
        gate_type_times = {type(gate): val for gate, val in gate_times_ns}
        op_id_pauli_errors = {cls._op_to_identifier(op): val for op, val in gate_pauli_errors}
        return NoiseProperties(
            gate_times_ns=gate_type_times,
            T1_ns={k: v for k, v in T1_ns},
            Tphi_ns={k: v for k, v in Tphi_ns},
            ro_fidelities={k: v for k, v in ro_fidelities},
            gate_pauli_errors=op_id_pauli_errors,
            validate=validate,
        )


class NoiseModelFromNoiseProperties(devices.NoiseModel):
    def __init__(self, noise_properties: NoiseProperties) -> None:
        """Creates a Noise Model from a NoiseProperties object that can be used with a Simulator.

        Args:
            noise_properties: the NoiseProperties object to be converted to a Noise Model.

        Raises:
            ValueError: if no NoiseProperties object is specified.
        """
        self._noise_properties = noise_properties
        self.noise_models = self._noise_properties.build_noise_models()

    def virtual_predicate(self, op: 'cirq.Operation') -> bool:
        """Returns True if an operation is virtual.

        Device-specific subclasses should implement this method to mark any
        operations which their device handles outside the quantum hardware.
        """
        return False

    def noisy_moments(
        self, moments: Iterable['cirq.Moment'], system_qubits: Sequence['cirq.Qid']
    ) -> Sequence['cirq.OP_TREE']:
        # Hack to bypass import order constraints
        from cirq import circuits

        # Split multi-qubit measurements into single-qubit measurements.
        # These will be recombined after noise is applied.
        split_measure_moments = []
        multi_measurements = {}
        for moment in moments:
            split_measure_ops = []
            for op in moment:
                if not protocols.is_measurement(op):
                    split_measure_ops.append(op)
                    continue
                m_key = protocols.measurement_key_obj(op)
                multi_measurements[m_key] = op
                for q in op.qubits:
                    split_measure_ops.append(ops.measure(q, key=m_key))
            split_measure_moments.append(ops.Moment(split_measure_ops))

        new_moments = []
        for moment in split_measure_moments:
            virtual_ops = {op for op in moment if self.virtual_predicate(op)}
            physical_ops = [
                op.with_tags(PHYSICAL_GATE_TAG) for op in moment if op not in virtual_ops
            ]
            if virtual_ops:
                new_moments.append(ops.Moment(virtual_ops))
            if physical_ops:
                new_moments.append(ops.Moment(physical_ops))

        split_measure_circuit = circuits.Circuit(new_moments)

        # Add actual noise.
        noisy_circuit = split_measure_circuit.copy()
        for model in self.noise_models:
            noisy_circuit = noisy_circuit.with_noise(model)

        # Recombine measurements.
        final_moments = []
        for moment in noisy_circuit:
            combined_measure_ops = []
            restore_keys = set()
            for op in moment:
                if not protocols.is_measurement(op):
                    combined_measure_ops.append(op)
                    continue
                restore_keys.add(protocols.measurement_key_obj(op))
            for key in restore_keys:
                combined_measure_ops.append(multi_measurements[key])
            final_moments.append(ops.Moment(combined_measure_ops))
        return final_moments
