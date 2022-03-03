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

"""Target gateset used for approximately compiling under a given two qubit gate."""

from typing import cast, TYPE_CHECKING

from cirq import ops, protocols
from cirq.qis import measures
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
from cirq.transformers.heuristic_decompositions import two_qubit_gate_tabulation
from cirq.transformers.target_gatesets import compilation_target_gateset

if TYPE_CHECKING:
    import cirq


class ApproximateTwoQubitTargetGateset(compilation_target_gateset.TwoQubitCompilationTargetGateset):
    """Target gateset giving approximate compilations using provided base gate."""

    def __init__(
        self,
        base_gate: 'cirq.Gate',
        max_infidelity: float = 0.01,
        *,
        sample_scaling: int = 50,
        allow_missed_points: bool = True,
        random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
    ) -> None:

        """Initializes ApproximateTwoQubitTargetGateset


        This gateset builds a `cirq.GateTabulation` (kak decomposition)
        around the provided base_gate to do fidelity limited decompositions.

        Args:
            base_gate: `cirq.Gate` to use as two qubit entangler
            max_infidelity: Maximum acceptable infidelity per
                two qubit operation. Note that gate merging may
                decrease the number of two qubit operations.
            sample_scaling: Relative number of random gate products to use in the
                tabulation. The total number of random local unitaries scales as
                ~ $max_infidelity^{-3/2} * sample_scaling$. Must be positive.
            allow_missed_points: If True, the tabulation is allowed to conclude
                even if not all points in the Weyl chamber are expected to be
                compilable using 2 or 3 base gates. Otherwise an error is raised
                in this case.
            random_state: Random state or random state seed.

        Raises:
            ValueError: if base_gate is not a two qubit gate.
        """
        if base_gate.num_qubits() != 2:
            raise ValueError(
                "base_gate requires a two qubit gate. Given"
                f" {str(base_gate)} which is {base_gate.num_qubits()} qubits."
            )

        super().__init__(
            base_gate,
            ops.MeasurementGate,
            ops.AnyUnitaryGateFamily(1),
            name=f'Approximate{str(base_gate)}Gateset.',
        )
        self._base_gate = base_gate
        self._tabulation = two_qubit_gate_tabulation.two_qubit_gate_product_tabulation(
            protocols.unitary(base_gate),
            max_infidelity,
            sample_scaling=sample_scaling,
            allow_missed_points=allow_missed_points,
            random_state=random_state,
        )

    @property
    def base_gate(self) -> 'cirq.Gate':
        """Get the base_gate from initialization."""
        return self._base_gate

    @property
    def tabulation(self) -> two_qubit_gate_tabulation.TwoQubitGateTabulation:
        """Get the GateTabulation object associated with base_gate."""
        return self._tabulation

    def _decompose_two_qubit_operation(self, op: 'cirq.Operation', _) -> 'cirq.OP_TREE':
        if not protocols.has_unitary(op):
            return NotImplemented

        if protocols.has_kraus(op):
            e_fid = measures.entanglement_fidelity(cast(protocols.SupportsKraus, op))
            if e_fid > 1.0 - self._tabulation.max_expected_infidelity:
                return []  # we are close enough to identity.

        q0, q1 = op.qubits
        decomp = self._tabulation.compile_two_qubit_gate(protocols.unitary(op))
        ret = []
        for i in range(len(decomp.local_unitaries) - 1):
            mats = decomp.local_unitaries[i]
            for mat, q in zip(mats, [q0, q1]):
                phxz_gate = single_qubit_decompositions.single_qubit_matrix_to_phxz(mat)
                if phxz_gate is not None:
                    ret.append(phxz_gate(q))
            ret.append(self._base_gate(q0, q1))

        mats = decomp.local_unitaries[-1]
        for mat, q in zip(mats, [q0, q1]):
            phxz_gate = single_qubit_decompositions.single_qubit_matrix_to_phxz(mat)
            if phxz_gate is not None:
                ret.append(phxz_gate(q))

        return ret
