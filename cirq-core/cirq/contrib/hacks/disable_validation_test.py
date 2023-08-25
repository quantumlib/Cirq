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

import pytest

import cirq
from cirq.contrib.hacks.disable_validation import disable_op_validation


def test_disable_op_validation():
    q0, q1 = cirq.LineQubit.range(2)

    # Fails normally.
    with pytest.raises(ValueError, match='Wrong number'):
        _ = cirq.H(q0, q1)

    # Fails if responsibility is not accepted.
    with pytest.raises(ValueError, match='mysterious and terrible'):
        with disable_op_validation():
            # This does not run - the with condition errors out first.
            _ = cirq.H(q0, q1)  # pragma: no cover

    # Passes, skipping validation.
    with disable_op_validation(accept_debug_responsibility=True):
        op = cirq.H(q0, q1)
        assert op.qubits == (q0, q1)

    # Validation is restored even on error.
    with pytest.raises(AssertionError):
        with disable_op_validation(accept_debug_responsibility=True):
            assert q0 == q1

    # Future developer: DO NOT REMOVE. This is NOT a duplicate!
    # It only LOOKS like a duplicate because this is a gross hack :D
    with pytest.raises(ValueError, match='Wrong number'):
        _ = cirq.H(q0, q1)
