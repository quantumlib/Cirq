from dataclasses import dataclass, field
from typing import Dict, List, Set
import numpy as np

import cirq, cirq_google
from cirq import devices
from cirq.devices.noise_utils import (
    OpIdentifier,
    unitary_entanglement_fidelity,
)


SINGLE_QUBIT_GATES = {
    cirq.ZPowGate,
    cirq.PhasedXZGate,
    cirq.MeasurementGate,
    cirq.ResetChannel,
}
SYMMETRIC_TWO_QUBIT_GATES = {
    cirq_google.SycamoreGate,
    cirq.FSimGate,
    cirq.PhasedFSimGate,
    cirq.ISwapPowGate,
    cirq.CZPowGate,
}
ASYMMETRIC_TWO_QUBIT_GATES: Set[type] = set()
TWO_QUBIT_GATES = SYMMETRIC_TWO_QUBIT_GATES | ASYMMETRIC_TWO_QUBIT_GATES


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

    Additional args:
        fsim_errors: Dict[OpIdentifier, cirq.PhasedFSimGate] of gate types
            (potentially on specific qubits) to the PhasedFSim fix-up operation
            for that gate. Defaults to no-op for all gates.
    """

    fsim_errors: Dict[OpIdentifier, cirq.PhasedFSimGate] = field(default_factory=dict)

    def __attrs_post_init__(self):
        t1_qubits = set(self.T1_ns)
        tphi_qubits = set(self.Tphi_ns)

        if t1_qubits != tphi_qubits:
            raise ValueError(f'Keys specified for T1 and Tphi are not' f' identical.')

        # validate two qubit gate errors.
        self._validate_symmetric_errors('gate_pauli_errors')
        self._validate_symmetric_errors('fsim_errors')

    @classmethod
    def single_qubit_gates(cls) -> Set[type]:
        return SINGLE_QUBIT_GATES

    @classmethod
    def two_qubit_gates(cls) -> Set[type]:
        return TWO_QUBIT_GATES

    def get_depolarizing_error(self) -> Dict[OpIdentifier, float]:
        if self._depolarizing_error:
            return self._depolarizing_error

        depol_errors = super().get_depolarizing_error()

        for op_id in depol_errors:
            if op_id.gate not in self.two_qubit_gates():
                continue
            if len(op_id.qubits) != 2:
                raise ValueError(
                    f'Gate {op_id.gate} takes two qubits, but {op_id.qubits} were given.'
                )
            # Subtract entangling angle error.
            if op_id in self.fsim_errors:
                unitary_err = cirq.unitary(self.fsim_errors[op_id])
                fid = unitary_entanglement_fidelity(unitary_err, np.eye(4))
                rp_fsim = 1 - fid
            elif op_id.gate_only() in self.fsim_errors:
                unitary_err = cirq.unitary(self.fsim_errors[op_id.gate_only()])
                fid = unitary_entanglement_fidelity(unitary_err, np.eye(4))
                rp_fsim = 1 - fid
            else:
                rp_fsim = 0

            depol_errors[op_id] -= rp_fsim
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


class NoiseModelFromGoogleNoiseProperties(devices.NoiseModelFromNoiseProperties):
    """A noise model defined from noise properties of a Google device."""

    def virtual_predicate(self, op: cirq.Operation) -> bool:
        return isinstance(op.gate, cirq.ZPowGate) and cirq_google.PhysicalZTag not in op.tags

    # noisy_moments is implemented by the superclass.
