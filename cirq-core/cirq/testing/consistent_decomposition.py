# Copyright 2018 The Cirq Developers
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

from typing import Any

import numpy as np

from cirq import devices, protocols, ops, circuits
from cirq.testing import lin_alg_utils


def assert_decompose_is_consistent_with_unitary(val: Any, ignoring_global_phase: bool = False):
    """Uses `val._unitary_` to check `val._phase_by_`'s behavior."""
    # pylint: disable=unused-variable
    __tracebackhide__ = True
    # pylint: enable=unused-variable

    expected = protocols.unitary(val, None)
    if expected is None:
        # If there's no unitary, it's vacuously consistent.
        return
    if isinstance(val, ops.Operation):
        qubits = val.qubits
        dec = protocols.decompose_once(val, default=None)
    else:
        qubits = tuple(devices.LineQid.for_gate(val))
        dec = protocols.decompose_once_with_qubits(val, qubits, default=None)
    if dec is None:
        # If there's no decomposition, it's vacuously consistent.
        return

    c = circuits.Circuit(dec)
    if len(c.all_qubits().difference(qubits)):
        # The decomposition contains ancilla qubits.
        ancilla = tuple(c.all_qubits().difference(qubits))
        qubit_order = ancilla + qubits
        actual = c.unitary(qubit_order=qubit_order)
        qid_shape = protocols.qid_shape(qubits)
        vol = np.prod(qid_shape, dtype=np.int64)
        actual = actual[:vol, :vol]
    else:
        actual = c.unitary(qubit_order=qubits)
    if ignoring_global_phase:
        lin_alg_utils.assert_allclose_up_to_global_phase(actual, expected, atol=1e-8)
    else:
        # coverage: ignore
        np.testing.assert_allclose(actual, expected, atol=1e-8)


def _known_gate_with_no_decomposition(val: Any):
    """Checks whether `val` is a known gate with no default decomposition to default gateset."""
    if isinstance(val, ops.MatrixGate):
        return protocols.qid_shape(val) not in [(2,), (2,) * 2, (2,) * 3]
    if isinstance(val, ops.BaseDensePauliString) and not protocols.has_unitary(val):
        return True
    if isinstance(val, ops.ControlledGate):
        if protocols.is_parameterized(val):
            return True
        if isinstance(val.sub_gate, ops.MatrixGate) and protocols.num_qubits(val.sub_gate) > 1:
            return True
        if val.control_qid_shape != (2,) * val.num_controls():
            return True
        if isinstance(val.control_values, ops.SumOfProducts):
            return True
        return _known_gate_with_no_decomposition(val.sub_gate)
    return False


def assert_decompose_ends_at_default_gateset(val: Any, ignore_known_gates: bool = True):
    """Asserts that cirq.decompose(val) ends at default cirq gateset or a known gate."""
    args = () if isinstance(val, ops.Operation) else (tuple(devices.LineQid.for_gate(val)),)
    dec_once = protocols.decompose_once(val, [val(*args[0]) if args else val], *args)
    for op in [*ops.flatten_to_ops(protocols.decompose(d) for d in dec_once)]:
        assert (_known_gate_with_no_decomposition(op.gate) and ignore_known_gates) or (
            op in protocols.decompose_protocol.DECOMPOSE_TARGET_GATESET
        ), f'{val} decomposed to {op}, which is not part of default cirq target gateset.'
