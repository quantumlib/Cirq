from dataclasses import dataclass, field
from typing import Dict, List, Set
import numpy as np

import cirq, cirq_google
from cirq import devices
from cirq.devices.noise_utils import (
    OpIdentifier,
    unitary_entanglement_fidelity,
)


@dataclass
class GoogleNoiseProperties(cirq.NoiseProperties):
    """Noise-defining properties for a Google device.

    Inherited args:
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

    Additional args:
        fsim_errors: Dict[OpIdentifier, cirq.PhasedFSimGate] of gate types
            (potentially on specific qubits) to the PhasedFSim fix-up operation
            for that gate. Defaults to no-op for all gates.
    """

    fsim_errors: Dict[OpIdentifier, cirq.PhasedFSimGate] = field(default_factory=dict)

    def __post_init__(self):
        if not self.validate:
            return
        t1_qubits = set(self.T1_ns)
        tphi_qubits = set(self.Tphi_ns)

        if t1_qubits != tphi_qubits:
            raise ValueError(f'Keys specified for T1 and Tphi are not' f' identical.')

        # validate two qubit gate errors.
        self._validate_symmetric_errors('gate_pauli_errors')
        if self.fsim_errors is not None:
            self._validate_symmetric_errors('fsim_errors')

    @classmethod
    def two_qubit_gates(cls) -> Set[type]:
        return super().two_qubit_gates() | {cirq_google.SycamoreGate}

    @classmethod
    def canonical_gates(cls) -> Dict[type, 'cirq.Gate']:
        return {
            **super().canonical_gates(),
            cirq_google.SycamoreGate: cirq_google.SycamoreGate(),
        }

    def get_depolarizing_error(self) -> Dict[OpIdentifier, float]:
        if self._depolarizing_error:
            return self._depolarizing_error

        depol_errors = super().get_depolarizing_error()

        for op_id in depol_errors:
            if op_id.gate_type not in self.two_qubit_gates():
                continue
            # Number of qubits is checked by the parent class.
            # Subtract entangling angle error.
            if op_id in self.fsim_errors:
                unitary_err = cirq.unitary(self.fsim_errors[op_id])
                fid = unitary_entanglement_fidelity(unitary_err, np.eye(4))
                depol_errors[op_id] -= 1 - fid

        # memoization is OK
        self._depolarizing_error = depol_errors
        return self._depolarizing_error

    def build_noise_models(self) -> List['cirq.NoiseModel']:
        """Construct all NoiseModels associated with NoiseProperties."""
        noise_models = super().build_noise_models()

        # Insert entangling gate coherent errors after thermal error.
        if self.fsim_errors:
            fsim_ops = {op_id: gate.on(*op_id.qubits) for op_id, gate in self.fsim_errors.items()}
            noise_models.insert(1, devices.InsertionNoiseModel(ops_added=fsim_ops))

        return noise_models

    def __str__(self) -> str:
        return 'GoogleNoiseProperties'

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
        args.append(f'    fsim_errors={self.fsim_errors!r}')
        args_str = ',\n'.join(args)
        return f'cirq_google.GoogleNoiseProperties(\n{args_str}\n)'

    def _json_dict_(self):
        json_dict = super()._json_dict_()
        json_dict['cirq_type'] = 'GoogleNoiseProperties'
        storage_fsim_errors = {
            self.get_canonical_gate(op_id.gate_type).on(*op_id.qubits): val
            for op_id, val in self.fsim_errors.items()
        }
        json_dict['fsim_errors'] = sorted(storage_fsim_errors.items(), key=str)
        return json_dict

    @classmethod
    def _from_json_dict_(
        cls,
        gate_times_ns,
        T1_ns,
        Tphi_ns,
        ro_fidelities,
        gate_pauli_errors,
        validate,
        fsim_errors,
        **kwargs,
    ):
        # TODO: overlaps op_ids with same qubits in different order
        gate_type_times = {type(gate): val for gate, val in gate_times_ns}
        op_id_pauli_errors = {}
        for op, val in gate_pauli_errors:
            op_id = cls._op_to_identifier(op)
            op_id_pauli_errors[op_id] = val
            op_id_pauli_errors[op_id.swapped()] = val
        op_id_fsim_errors = {}
        for op, val in fsim_errors:
            op_id = cls._op_to_identifier(op)
            op_id_fsim_errors[op_id] = val
            op_id_fsim_errors[op_id.swapped()] = val
        return GoogleNoiseProperties(
            gate_times_ns=gate_type_times,
            T1_ns={k: v for k, v in T1_ns},
            Tphi_ns={k: v for k, v in Tphi_ns},
            ro_fidelities={k: v for k, v in ro_fidelities},
            gate_pauli_errors=op_id_pauli_errors,
            validate=validate,
            fsim_errors=op_id_fsim_errors,
        )


class NoiseModelFromGoogleNoiseProperties(devices.NoiseModelFromNoiseProperties):
    """A noise model defined from noise properties of a Google device."""

    def virtual_predicate(self, op: cirq.Operation) -> bool:
        return isinstance(op.gate, cirq.ZPowGate) and cirq_google.PhysicalZTag not in op.tags

    # noisy_moments is implemented by the superclass.
