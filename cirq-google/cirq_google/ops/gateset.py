from typing import cast, Optional, Union, Type, List, Sequence

import cirq
from cirq_google.ops import fsim_gate_family, sycamore_gate
from cirq.protocols.decompose_protocol import DecomposeResult
import cirq_google.transformers.target_gatesets as cg_target_gatesets


def _build_equivalence_gate_families(device_gateset: cirq.Gateset) -> List[cirq.GateFamily]:
    fsim_gates = [
        cast(fsim_gate_family.POSSIBLE_FSIM_GATES, g)
        for g in device_gateset.gates
        if isinstance(g, tuple(fsim_gate_family.POSSIBLE_FSIM_GATE_TYPES))
    ]
    return [fsim_gate_family.FSimGateFamily(gates_to_accept=fsim_gates)]

    # TODO(verult) PhasedXZGateFamily


class Gateset(cirq.CompilationTargetGateset):
    """Represents the gateset of a Google device.

    This gateset accept circuits that use any Cirq gate representation of valid device gates.

    It can also be used as an argument in cirq.optimize_for_target_gateset() to transform your
    circuit into one that consists of valid gates for the target device. In cases where multiple
    `CompilationTargetGateset`s are possible for a given gateset, a pre-determined default will be
    applied, but all available `CompilationTargetGateset`s can be accessed via the `target_gatesets`
    property.

    Users typically interact with this class by accessing its instantiations via
    `GoogleDevice.metadata.gateset`.

    Examples:

    ```
    # View what gates and `CompilationTargetGatesets` are supported by the device.
    print(gateset)

    # Checks whether a gate is valid for the device.
    cirq.X in gateset

    # Checks whether the circuit only contains valid gates for the device.
    gateset.validate(circuit)

    # Transform the circuit into one that can be executed on the device.
    cirq.optimize_for_target_gateset(circuit, gateset=gateset)

    # Choose your own target gateset, if the device supports multiple.
    cirq.optimize_for_target_gateset(circuit, gateset=gateset.target_gatesets[1])
    """

    def __init__(
        self,
        device_name,
        *device_gates: Union[Type[cirq.Gate], cirq.Gate, cirq.GateFamily],
    ) -> None:
        self._device_name = device_name
        self._device_gateset = cirq.Gateset(*device_gates)
        equivalence_gate_families = _build_equivalence_gate_families(self._device_gateset)
        super().__init__(*(tuple(self._device_gateset.gates) + tuple(equivalence_gate_families)))
        self._build_target_gatesets()

    def _build_target_gatesets(self):
        self._target_gatesets: List[cirq.CompilationTargetGateset] = []

        if cirq.CZ in self._device_gateset:
            self._target_gatesets.append(cirq.CZTargetGateset())
        if cirq.SQRT_ISWAP in self._device_gateset and cirq.SQRT_ISWAP_INV in self._device_gateset:
            self._target_gatesets.append(cirq.SqrtIswapTargetGateset())
        if sycamore_gate.SYC in self._device_gateset:
            # TODO(verult) add tabulation
            self._target_gatesets.append(cg_target_gatesets.SycamoreTargetGateset())

    @property
    def num_qubits(self) -> int:
        if self.default_target_gateset is None:
            return 0
        return self.default_target_gateset.num_qubits

    def decompose_to_target_gateset(self, op: 'cirq.Operation', moment_idx: int) -> DecomposeResult:
        if self.default_target_gateset is None:
            return None
        return self.default_target_gateset.decompose_to_target_gateset(op, moment_idx)

    @property
    def target_gatesets(self) -> Sequence[cirq.CompilationTargetGateset]:
        return self._target_gatesets

    @property
    def default_target_gateset(self) -> Optional[cirq.CompilationTargetGateset]:
        return self._target_gatesets[0] if self._target_gatesets else None

    def __str__(self) -> str:
        header = f"Gateset for Google device '{self._device_name}':"
        header_line = '-' * len(header)
        target_gatesets = "None"
        if len(self._target_gatesets) > 0:
            target_gatesets = '\n\n'.join(
                ['#####\n' + str(gs) + '\n#####' for gs in self._target_gatesets]
            )
        return (
            f'{header}\n'
            + f'{header_line}\n\n'
            + 'Device Gates:\n\n'
            + '#####\n'
            + str(self._device_gateset)
            + '\n#####\n\nTarget Gatesets:\n\n'
            + target_gatesets
            + '\n'
        )
        # TODO(verult) consider including equivalence gatefamily information
