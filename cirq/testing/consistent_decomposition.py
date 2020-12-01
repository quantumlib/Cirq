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

    actual = circuits.Circuit(dec).unitary(qubit_order=qubits)

    if ignoring_global_phase:
        lin_alg_utils.assert_allclose_up_to_global_phase(actual, expected, atol=1e-8)
    else:
        # coverage: ignore
        np.testing.assert_allclose(actual, expected, atol=1e-8)
