# Copyright 2024 The Cirq Developers
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

from collections.abc import Mapping
from typing import cast

import numpy as np

from cirq import circuits, ops
from cirq.transformers import transformer_api


def _gate_in_moment(gate: ops.Gate, moment: circuits.Moment) -> bool:
    """Check whether `gate` is in `moment`."""
    return any(op.gate == gate for op in moment)


@transformer_api.transformer
class DepolarizingNoiseTransformer:
    """Add local depolarizing noise after two-qubit gates in a specified circuit. More specifically,
    with probability p, append a random non-identity two-qubit Pauli operator after each specified
    two-qubit gate.

    Attrs:
        p: The probability with which to add noise.
        target_gate: Add depolarizing nose after this type of gate
    """

    def __init__(
        self, p: float | Mapping[tuple[ops.Qid, ops.Qid], float], target_gate: ops.Gate = ops.CZ
    ):
        """Initialize the depolarizing noise transformer with some depolarizing probability and
        target gate.

        Args:
            p: The depolarizing probability, either a single float or a mapping from pairs of qubits
               to floats.
           target_gate: The gate after which to add depolarizing noise.

        Raises:
            TypeError: If `p` is not either be a float or a mapping from sorted qubit pairs to
                       floats.
        """

        if not isinstance(p, (Mapping, float)):
            raise TypeError(  # pragma: no cover
                "p must either be a float or a mapping from"  # pragma: no cover
                + "sorted qubit pairs to floats"  # pragma: no cover
            )  # pragma: no cover
        self.p = p
        self.p_func = (
            (lambda _: p)
            if isinstance(p, (int, float))
            else (lambda pair: cast(Mapping, p).get(pair, 0.0))
        )
        self.target_gate = target_gate

    def __call__(
        self,
        circuit: circuits.AbstractCircuit,
        rng: np.random.Generator | None = None,
        *,
        context: transformer_api.TransformerContext | None = None,
    ):
        """Apply the transformer to the given circuit.

        Args:
            circuit: The circuit to add noise to.
            context: Not used; to satisfy transformer API.

        Returns:
            The transformed circuit.
        """
        if rng is None:
            rng = np.random.default_rng()
        target_gate = self.target_gate

        # add random Pauli gates with probability p after each of the specified gate
        assert target_gate.num_qubits() == 2, "`target_gate` must be a two-qubit gate."
        paulis = [ops.I, ops.X, ops.Y, ops.Z]
        new_moments = []
        for moment in circuit:
            new_moments.append(moment)
            if _gate_in_moment(target_gate, moment):
                # add a new moment with the Paulis
                target_pairs = {
                    tuple(sorted(op.qubits)) for op in moment.operations if op.gate == target_gate
                }
                added_moment_ops = []
                for pair in target_pairs:
                    pair_sorted_tuple = (pair[0], pair[1])
                    p_i = self.p_func(pair_sorted_tuple)
                    apply = rng.choice([True, False], p=[p_i, 1 - p_i])
                    if apply:
                        choices = [
                            (pauli_a(pair[0]), pauli_b(pair[1]))
                            for pauli_a in paulis
                            for pauli_b in paulis
                        ][1:]
                        pauli_to_apply = rng.choice(np.array(choices, dtype=object))
                        added_moment_ops.append(pauli_to_apply)
                if len(added_moment_ops) > 0:
                    new_moments.append(circuits.Moment(*added_moment_ops))
        return circuits.Circuit.from_moments(*new_moments)
