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

from typing import Any, Optional, Sequence

from cirq import protocols, value
from cirq.testing.circuit_compare import (
        assert_apply_unitary_to_tensor_is_consistent_with_unitary)
from cirq.testing.consistent_decomposition import (
        assert_decompose_is_consistent_with_unitary)
from cirq.testing.consistent_phase_by import (
        assert_phase_by_is_consistent_with_unitary)
from cirq.testing.equivalent_repr_eval import assert_equivalent_repr


def assert_implements_consistent_protocols(
        val: Any,
        *,
        qubit_count: Optional[int] = None,
        exponents: Sequence[Any] = (-1,
                                    -0.5,
                                    -0.25,
                                    -0.1,
                                    0,
                                    0.1,
                                    0.25,
                                    0.5,
                                    1,
                                    value.Symbol('s'))):
    """Checks that a value is internally consistent and has a good __repr__."""

    _assert_meets_standards_helper(val, qubit_count)

    for exponent in exponents:
        p = protocols.pow(val, exponent, None)
        if p is not None:
            _assert_meets_standards_helper(val**exponent, qubit_count)


def _assert_meets_standards_helper(val: Any, qubit_count: Optional[int]):
    if protocols.has_unitary(val):
        assert_apply_unitary_to_tensor_is_consistent_with_unitary(
                val, qubit_count=qubit_count)
        if getattr(val, '_decompose_', None) is not None:
            assert_decompose_is_consistent_with_unitary(val)
        if getattr(val, '_phase_by_', None) is not None:
            assert_phase_by_is_consistent_with_unitary(val)

    assert_equivalent_repr(val)
