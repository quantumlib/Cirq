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

"""Transformers that symbolize operations."""

from __future__ import annotations

import re
from typing import Hashable, TYPE_CHECKING

import attrs
import sympy
from attrs import validators

from cirq import ops
from cirq.transformers import transformer_api, transformer_primitives

if TYPE_CHECKING:
    import cirq


@attrs.frozen
class SymbolizeTag:
    prefix: str = attrs.field(
        validator=validators.and_(validators.instance_of(str), validators.min_len(1))
    )


@transformer_api.transformer
def symbolize_single_qubit_gates_by_indexed_tags(
    circuit: cirq.AbstractCircuit,
    *,
    context: cirq.TransformerContext | None = None,
    symbolize_tag: SymbolizeTag = SymbolizeTag(prefix="TO-PHXZ"),
) -> cirq.Circuit:
    """Symbolizes single qubit operations by indexed tags prefixed by symbolize_tag.prefix.

    Example:
        >>> from cirq import transformers
        >>> q0, q1 = cirq.LineQubit.range(2)
        >>> c = cirq.Circuit(
        ...         cirq.X(q0).with_tags("phxz_0"),
        ...         cirq.CZ(q0,q1),
        ...         cirq.Y(q0).with_tags("phxz_1"),
        ...         cirq.X(q0))
        >>> print(c)
        0: ───X[phxz_0]───@───Y[phxz_1]───X───
                          │
        1: ───────────────@───────────────────
        >>> new_circuit = cirq.symbolize_single_qubit_gates_by_indexed_tags(
        ...     c, symbolize_tag=transformers.SymbolizeTag(prefix="phxz"))
        >>> print(new_circuit)
        0: ───PhXZ(a=a0,x=x0,z=z0)───@───PhXZ(a=a1,x=x1,z=z1)───X───
                                     │
        1: ──────────────────────────@──────────────────────────────

    Args:
        circuit: Input circuit to apply the transformations on. The input circuit is not mutated.
        context: `cirq.TransformerContext` storing common configurable options for transformers.
        symbolize_tag: The tag info used to symbolize the phxz gate. Prefix is required.

    Returns:
        Copy of the transformed input circuit.
    """

    def _map_func(op: cirq.Operation, _):
        """Maps an op with tag `{tag_prefix}_i` to a symbolized `PhasedXZGate(xi,zi,ai)`."""
        tags: set[Hashable] = set(op.tags)
        tag_id: None | int = None
        for tag in tags:
            if re.fullmatch(f"{symbolize_tag.prefix}_\\d+", str(tag)):
                if tag_id is None:
                    tag_id = int(str(tag).rsplit("_", maxsplit=-1)[-1])
                else:
                    raise ValueError(f"Multiple tags are prefixed with {symbolize_tag.prefix}.")
        if tag_id is None:
            return op
        tags.remove(f"{symbolize_tag.prefix}_{tag_id}")
        phxz_params = {
            "x_exponent": sympy.Symbol(f"x{tag_id}"),
            "z_exponent": sympy.Symbol(f"z{tag_id}"),
            "axis_phase_exponent": sympy.Symbol(f"a{tag_id}"),
        }

        return ops.PhasedXZGate(**phxz_params).on(*op.qubits).with_tags(*tags)

    return transformer_primitives.map_operations(
        circuit.freeze(),
        _map_func,
        deep=context.deep if context else False,
        tags_to_ignore=context.tags_to_ignore if context else [],
    ).unfreeze(copy=False)
