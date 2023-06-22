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

import cirq
import numpy as np
import pytest
from cirq_ft.infra.decompose_protocol import _fredkin, _try_decompose_from_known_decompositions


def test_fredkin_unitary():
    c, t1, t2 = cirq.LineQid.for_gate(cirq.FREDKIN)
    context = cirq.DecompositionContext(cirq.ops.SimpleQubitManager())
    np.testing.assert_allclose(
        cirq.Circuit(_fredkin((c, t1, t2), context)).unitary(),
        cirq.unitary(cirq.FREDKIN(c, t1, t2)),
        atol=1e-8,
    )


@pytest.mark.parametrize('gate', [cirq.FREDKIN, cirq.FREDKIN**-1])
def test_decompose_fredkin(gate):
    c, t1, t2 = cirq.LineQid.for_gate(cirq.FREDKIN)
    op = cirq.FREDKIN(c, t1, t2)
    context = cirq.DecompositionContext(cirq.ops.SimpleQubitManager())
    want = tuple(cirq.flatten_op_tree(_fredkin((c, t1, t2), context)))
    assert want == _try_decompose_from_known_decompositions(op, context)

    op = cirq.FREDKIN(c, t1, t2).with_classical_controls('key')
    classical_controls = op.classical_controls
    want = tuple(
        o.with_classical_controls(*classical_controls)
        for o in cirq.flatten_op_tree(_fredkin((c, t1, t2), context))
    )
    assert want == _try_decompose_from_known_decompositions(op, context)
