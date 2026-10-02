# Copyright 2026 The Cirq Developers
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

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, TYPE_CHECKING

from cirq import protocols, value
from cirq.ops import raw_types

if TYPE_CHECKING:
    import cirq


@value.value_equality
class PostSelectionGate(raw_types.Gate):
    r"""Projects onto a subspace of the computational basis and renormalizes.

    This gate simulates post-selection. It keeps only the part of the state that lies in the span
    of the given computational basis states and renormalizes the result. A pure state
    $|\psi\rangle$ becomes $P|\psi\rangle / \lVert P|\psi\rangle \rVert$ and a density matrix
    $\rho$ becomes $P \rho P / \mathrm{tr}(P \rho P)$, where $P$ is the projector onto the
    subspace.

    The gate is neither unitary nor a quantum channel (it does not preserve the trace), so it does
    not implement `cirq.unitary`, `cirq.kraus` or `cirq.mixture`. It can only be simulated, and
    only by simulators whose state representation supports post-selection, which currently are
    `cirq.Simulator` and `cirq.DensityMatrixSimulator`. It can not be run on a real device.

    Post-selection conditions the state, it does not sample an outcome. The projection is applied
    deterministically, so the probability of the post-selected outcome is not reflected in the
    number of repetitions returned by `run`.

    A subspace whose true probability is exactly zero is detected reliably. A subspace whose true
    probability is real but extremely small, comparable to the simulator's floating-point
    precision, may not be reliably distinguished from zero, or its post-selected state may be
    dominated by numerical noise rather than signal.
    """

    def __init__(self, qid_shape: Sequence[int], subspaces: Iterable[Sequence[int]]) -> None:
        r"""Creates a gate simulating post-selection.

        Args:
            qid_shape: The shape (dimensions) of the qudits this gate acts on.
            subspaces: The computational basis states spanning the post-selection subspace, each
                given as one value per qudit. For example, projecting two qutrits onto the span of
                |00> and |12> is written as `[(0, 0), (1, 2)]`.

        Raises:
            ValueError: If there are no subspaces, or if a subspace does not fit `qid_shape`.
        """
        qid_shape = tuple(qid_shape)
        unique_subspaces = tuple(sorted({tuple(subspace) for subspace in subspaces}))
        if not unique_subspaces:
            raise ValueError('At least one subspace is required for post-selection.')
        for subspace in unique_subspaces:
            if len(subspace) != len(qid_shape):
                raise ValueError(f'Subspace {subspace} does not match qid_shape {qid_shape}.')
            if not all(0 <= digit < dim for digit, dim in zip(subspace, qid_shape)):
                raise ValueError(f'Subspace {subspace} is out of range for qid_shape {qid_shape}.')
        self._qid_shape = qid_shape
        self._subspaces = unique_subspaces

    def _qid_shape_(self) -> tuple[int, ...]:
        return self._qid_shape

    def _act_on_(self, sim_state: cirq.SimulationStateBase, qubits: Sequence[cirq.Qid]) -> bool:
        from cirq.sim import SimulationState

        if not isinstance(sim_state, SimulationState):
            return NotImplemented
        sim_state.post_select(qubits, self._subspaces)
        return True

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        subspaces = '|'.join(
            ''.join(str(digit) for digit in subspace) for subspace in self._subspaces
        )
        return protocols.CircuitDiagramInfo(
            wire_symbols=(f'PostSelect({subspaces})',) * len(self._qid_shape)
        )

    def _value_equality_values_(self) -> Any:
        return self._subspaces, self._qid_shape

    def _json_dict_(self) -> dict[str, Any]:
        return {'qid_shape': self._qid_shape, 'subspaces': self._subspaces}

    @classmethod
    def _from_json_dict_(cls, qid_shape, subspaces, **kwargs) -> PostSelectionGate:
        return cls(qid_shape=qid_shape, subspaces=subspaces)

    def __repr__(self) -> str:
        return (
            f'cirq.PostSelectionGate(qid_shape={self._qid_shape!r}, '
            f'subspaces={self._subspaces!r})'
        )
