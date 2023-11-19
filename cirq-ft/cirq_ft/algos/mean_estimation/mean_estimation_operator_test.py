# Copyright 2023 The Cirq Developers
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

from typing import Optional, Sequence, Tuple

import cirq
import cirq_ft
import numpy as np
import pytest
from attr import frozen
from cirq_ft import infra
from cirq._compat import cached_property
from cirq_ft.algos.mean_estimation import CodeForRandomVariable, MeanEstimationOperator
from cirq_ft.infra import bit_tools


@frozen
class BernoulliSynthesizer(cirq_ft.PrepareOracle):
    r"""Synthesizes the state $sqrt(1 - p)|00..00> + sqrt(p)|11..11>$"""

    p: float
    nqubits: int

    @cached_property
    def selection_registers(self) -> Tuple[cirq_ft.SelectionRegister, ...]:
        return (cirq_ft.SelectionRegister('q', self.nqubits, 2),)

    def decompose_from_registers(  # type:ignore[override]
        self, context, q: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        theta = np.arccos(np.sqrt(1 - self.p))
        yield cirq.ry(2 * theta).on(q[0])
        yield [cirq.CNOT(q[0], q[i]) for i in range(1, len(q))]


@frozen
class BernoulliEncoder(cirq_ft.SelectOracle):
    r"""Encodes Bernoulli random variable y0/y1 as $Enc|ii..i>|0> = |ii..i>|y_{i}>$ where i=0/1."""

    p: float
    y: Tuple[int, int]
    selection_bitsize: int
    target_bitsize: int
    control_val: Optional[int] = None

    @cached_property
    def control_registers(self) -> Tuple[cirq_ft.Register, ...]:
        return () if self.control_val is None else (cirq_ft.Register('control', 1),)

    @cached_property
    def selection_registers(self) -> Tuple[cirq_ft.SelectionRegister, ...]:
        return (cirq_ft.SelectionRegister('q', self.selection_bitsize, 2),)

    @cached_property
    def target_registers(self) -> Tuple[cirq_ft.Register, ...]:
        return (cirq_ft.Register('t', self.target_bitsize),)

    def decompose_from_registers(  # type:ignore[override]
        self, context, q: Sequence[cirq.Qid], t: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        y0_bin = bit_tools.iter_bits(self.y[0], self.target_bitsize)
        y1_bin = bit_tools.iter_bits(self.y[1], self.target_bitsize)

        for y0, y1, tq in zip(y0_bin, y1_bin, t):
            if y0:
                yield cirq.X(tq).controlled_by(  # pragma: no cover
                    *q, control_values=[0] * self.selection_bitsize  # pragma: no cover
                )  # pragma: no cover
            if y1:
                yield cirq.X(tq).controlled_by(*q, control_values=[1] * self.selection_bitsize)

    def controlled(self, *args, **kwargs):
        cv = kwargs['control_values'][0]
        return BernoulliEncoder(self.p, self.y, self.selection_bitsize, self.target_bitsize, cv)

    @cached_property
    def mu(self) -> float:
        return self.p * self.y[1] + (1 - self.p) * self.y[0]

    @cached_property
    def s_square(self) -> float:
        return self.p * (self.y[1] ** 2) + (1 - self.p) * (self.y[0] ** 2)


def overlap(v1: np.ndarray, v2: np.ndarray) -> float:
    return np.abs(np.vdot(v1, v2)) ** 2


def satisfies_theorem_321(
    synthesizer: cirq_ft.PrepareOracle,
    encoder: cirq_ft.SelectOracle,
    c: float,
    s: float,
    mu: float,
    arctan_bitsize: int,
):
    r"""Verifies Theorem 3.21 of https://arxiv.org/abs/2208.07544

    Pr[∣sin(θ/2)∣ ∈ ∣µ∣ / √(1 + s ** 2) . [1 / (1 + cs), 1 / (1 - cs)]] >= (1 - 2 / c**2)
    """
    code = CodeForRandomVariable(synthesizer=synthesizer, encoder=encoder)
    mean_gate = MeanEstimationOperator(code, arctan_bitsize=arctan_bitsize)

    # Compute a reduced unitary for mean_op.
    u = cirq.unitary(mean_gate)
    assert cirq.is_unitary(u)

    # Compute the final state vector obtained using the synthesizer `Prep |0>`
    prep_op = synthesizer.on_registers(**infra.get_named_qubits(synthesizer.signature))
    prep_state = cirq.Circuit(prep_op).final_state_vector()

    expected_hav = abs(mu) * np.sqrt(1 / (1 + s**2))
    expected_hav_low = expected_hav / (1 + c * s)
    expected_hav_high = expected_hav / (1 - c * s)

    overlap_sum = 0.0
    eigvals, eigvects = cirq.linalg.unitary_eig(u)
    for eig_val, eig_vect in zip(eigvals, eigvects.T):
        theta = np.abs(np.angle(eig_val))
        hav_theta = np.sin(theta / 2)
        overlap_prob = overlap(prep_state, eig_vect)
        if expected_hav_low <= hav_theta <= expected_hav_high:
            overlap_sum += overlap_prob
    return overlap_sum >= 1 - 2 / (c**2) > 0


@pytest.mark.parametrize('selection_bitsize', [1, 2])
@pytest.mark.parametrize(
    'p, y_1, target_bitsize, c',
    [
        (1 / 100 * 1 / 100, 3, 2, 100 / 7),
        (1 / 50 * 1 / 50, 2, 2, 50 / 4),
        (1 / 50 * 1 / 50, 1, 1, 50 / 10),
        (1 / 4 * 1 / 4, 1, 1, 1.5),
    ],
)
def test_mean_estimation_bernoulli(
    p: int, y_1: int, selection_bitsize: int, target_bitsize: int, c: float, arctan_bitsize: int = 5
):
    synthesizer = BernoulliSynthesizer(p, selection_bitsize)
    encoder = BernoulliEncoder(p, (0, y_1), selection_bitsize, target_bitsize)
    s = np.sqrt(encoder.s_square)
    # For hav_theta interval to be reasonably wide, 1/(1-cs) term should be <=2; thus cs <= 0.5.
    # The theorem assumes that C >= 1 and s <= 1 / c.
    assert c * s <= 0.5 and c >= 1 >= s

    assert satisfies_theorem_321(
        synthesizer=synthesizer,
        encoder=encoder,
        c=c,
        s=s,
        mu=encoder.mu,
        arctan_bitsize=arctan_bitsize,
    )


@frozen
class GroverSynthesizer(cirq_ft.PrepareOracle):
    r"""Prepare a uniform superposition over the first $2^n$ elements."""

    n: int

    @cached_property
    def selection_registers(self) -> Tuple[cirq_ft.SelectionRegister, ...]:
        return (cirq_ft.SelectionRegister('selection', self.n),)

    def decompose_from_registers(  # type:ignore[override]
        self, *, context, selection: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        yield cirq.H.on_each(*selection)

    def __pow__(self, power):
        if power in [+1, -1]:
            return self
        return NotImplemented  # pragma: no cover


@frozen
class GroverEncoder(cirq_ft.SelectOracle):
    """Enc|marked_item>|0> --> |marked_item>|marked_val>"""

    n: int
    marked_item: int
    marked_val: int

    @cached_property
    def control_registers(self) -> Tuple[cirq_ft.Register, ...]:
        return ()

    @cached_property
    def selection_registers(self) -> Tuple[cirq_ft.SelectionRegister, ...]:
        return (cirq_ft.SelectionRegister('selection', self.n),)

    @cached_property
    def target_registers(self) -> Tuple[cirq_ft.Register, ...]:
        return (cirq_ft.Register('target', self.marked_val.bit_length()),)

    def decompose_from_registers(  # type:ignore[override]
        self, context, *, selection: Sequence[cirq.Qid], target: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        selection_cv = [
            *bit_tools.iter_bits(self.marked_item, infra.total_bits(self.selection_registers))
        ]
        yval_bin = [*bit_tools.iter_bits(self.marked_val, infra.total_bits(self.target_registers))]

        for b, q in zip(yval_bin, target):
            if b:
                yield cirq.X(q).controlled_by(*selection, control_values=selection_cv)

    @cached_property
    def mu(self) -> float:
        return self.marked_val / 2**self.n

    @cached_property
    def s_square(self) -> float:
        return (self.marked_val**2) / 2**self.n


@pytest.mark.parametrize('n, marked_val, c', [(5, 1, 4), (4, 1, 2), (2, 1, np.sqrt(2))])
def test_mean_estimation_grover(
    n: int, marked_val: int, c: float, marked_item: int = 1, arctan_bitsize: int = 5
):
    synthesizer = GroverSynthesizer(n)
    encoder = GroverEncoder(n, marked_item=marked_item, marked_val=marked_val)
    s = np.sqrt(encoder.s_square)
    assert c * s < 1 and c >= 1 >= s

    assert satisfies_theorem_321(
        synthesizer=synthesizer,
        encoder=encoder,
        c=c,
        s=s,
        mu=encoder.mu,
        arctan_bitsize=arctan_bitsize,
    )


def test_mean_estimation_operator_consistent_protocols():
    p, selection_bitsize, y_1, target_bitsize, arctan_bitsize = 0.1, 2, 1, 1, 4
    synthesizer = BernoulliSynthesizer(p, selection_bitsize)
    encoder = BernoulliEncoder(p, (0, y_1), selection_bitsize, target_bitsize)
    code = CodeForRandomVariable(synthesizer=synthesizer, encoder=encoder)
    mean_gate = MeanEstimationOperator(code, arctan_bitsize=arctan_bitsize)
    op = mean_gate.on_registers(**infra.get_named_qubits(mean_gate.signature))

    # Test controlled gate.
    equals_tester = cirq.testing.EqualsTester()
    equals_tester.add_equality_group(
        mean_gate.controlled(),
        mean_gate.controlled(num_controls=1),
        mean_gate.controlled(control_values=(1,)),
        op.controlled_by(cirq.q("control")).gate,
    )
    equals_tester.add_equality_group(
        mean_gate.controlled(control_values=(0,)),
        mean_gate.controlled(num_controls=1, control_values=(0,)),
        op.controlled_by(cirq.q("control"), control_values=(0,)).gate,
    )
    with pytest.raises(NotImplementedError, match="Cannot create a controlled version"):
        _ = mean_gate.controlled(num_controls=2)

    # Test with_power
    assert mean_gate.with_power(5) ** 2 == MeanEstimationOperator(
        code, arctan_bitsize=arctan_bitsize, power=10
    )
    # Test diagrams
    expected_symbols = ['U_ko'] * cirq.num_qubits(mean_gate)
    assert cirq.circuit_diagram_info(mean_gate).wire_symbols == tuple(expected_symbols)
    control_symbols = ['@']
    assert cirq.circuit_diagram_info(mean_gate.controlled()).wire_symbols == tuple(
        control_symbols + expected_symbols
    )
    control_symbols = ['@(0)']
    assert cirq.circuit_diagram_info(
        mean_gate.controlled(control_values=(0,))
    ).wire_symbols == tuple(control_symbols + expected_symbols)
    expected_symbols[-1] = 'U_ko^2'
    assert cirq.circuit_diagram_info(
        mean_gate.with_power(2).controlled(control_values=(0,))
    ).wire_symbols == tuple(control_symbols + expected_symbols)
