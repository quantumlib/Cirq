# Copyright 2025 The Cirq Developers
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

"""A pauli insertion transformer."""

from __future__ import annotations

import inspect

import numpy as np

from cirq import circuits, ops
from cirq.transformers import transformer_api

_PAULIS: tuple[ops.Gate] = (ops.I, ops.X, ops.Y, ops.Z)  # type: ignore[has-type]


@transformer_api.transformer
class PauliInsertionTransformer:
    r"""Creates a pauli insertion transformer.

    A pauli insertion operation samples paulis from $\{I, X, Y, Z\}^2$ with the given
    probabilities and adds it before the target 2Q gate/operation. This procedure is commonly
    used in zero noise extrapolation (ZNE), see appendix D of https://arxiv.org/abs/2503.20870.
    """

    def __init__(
        self,
        target: ops.Gate | ops.GateFamily | ops.Gateset | type[ops.Gate],
        probabilities: np.ndarray | None = None,
    ):
        """Makes a pauli insertion transformer that samples 2Q paulis with the given probabilities.

        Args:
            target: The target gate, gatefamily, gateset, or type (e.g. ZZPowGAte).
            probabilities: Optional ndarray representing the probabilities of sampling 2Q paulis.
                The order of the paulis is IXYZ. If None, assume uniform distribution.
        Returns:
            A gauge transformer.
        """
        if probabilities is None:
            probabilities = np.ones((4, 4)) / 16
        probabilities = np.asarray(probabilities)
        assert probabilities.shape == (4, 4)
        assert np.isclose(probabilities.sum(), 1)

        if inspect.isclass(target):
            self.target = ops.GateFamily(target)
        elif isinstance(target, ops.Gate):
            self.target = ops.Gateset(target)
        else:
            assert isinstance(target, (ops.Gateset, ops.GateFamily))
            self.target = target
        self._flat_probs = probabilities.reshape(-1)

    def __call__(
        self,
        circuit: circuits.AbstractCircuit,
        *,
        rng_or_seed: np.random.Generator | int | None = None,
        context: transformer_api.TransformerContext | None = None,
    ):
        context = (
            context
            if isinstance(context, transformer_api.TransformerContext)
            else transformer_api.TransformerContext()
        )
        rng = (
            rng_or_seed
            if isinstance(rng_or_seed, np.random.Generator)
            else np.random.default_rng(rng_or_seed)
        )

        if context.deep:
            raise ValueError(f"this transformer doesn't support deep {context=}")

        tags_to_ignore = frozenset(context.tags_to_ignore)
        new_circuit: list[circuits.Moment] = []
        for moment in circuit:
            if any(tag in tags_to_ignore for tag in moment.tags):
                new_circuit.append(moment)
                continue
            new_moment = []
            for op in moment:
                if any(tag in tags_to_ignore for tag in op.tags):
                    continue
                if op not in self.target:
                    continue
                pair = np.unravel_index(rng.choice(16, p=self._flat_probs), (4, 4))
                for pauli_index, q in zip(pair, op.qubits):
                    if new_circuit and (q not in new_circuit[-1].qubits):
                        new_circuit[-1] += _PAULIS[pauli_index](q)
                    else:
                        new_moment.append(_PAULIS[pauli_index](q))
            if new_moment:
                new_circuit.append(circuits.Moment(new_moment))
            new_circuit.append(moment)
        return circuits.Circuit.from_moments(*new_circuit)
