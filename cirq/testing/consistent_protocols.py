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

from typing import Any, Optional, Sequence, Type, Union

from cirq import ops, protocols, value
from cirq.testing.circuit_compare import (
        assert_has_consistent_apply_unitary)
from cirq.testing.consistent_decomposition import (
        assert_decompose_is_consistent_with_unitary)
from cirq.testing.consistent_phase_by import (
        assert_phase_by_is_consistent_with_unitary)
from cirq.testing.consistent_qasm import (
        assert_qasm_is_consistent_with_unitary)
from cirq.testing.equivalent_repr_eval import assert_equivalent_repr


def assert_implements_consistent_protocols(
        val: Any,
        *,
        exponents: Sequence[Any] = (
            0, 1, -1, 0.5, 0.25, -0.5, 0.1, value.Symbol('s')),
        qubit_count: Optional[int] = None,
        setup_code: str = 'import cirq\nimport numpy as np'
        ) -> None:
    """Checks that a value is internally consistent and has a good __repr__."""

    _assert_meets_standards_helper(val, qubit_count, setup_code)

    for exponent in exponents:
        p = protocols.pow(val, exponent, None)
        if p is not None:
            _assert_meets_standards_helper(
                    val**exponent, qubit_count, setup_code)


def assert_eigengate_implements_consistent_protocols(
        eigen_gate_type: Type[ops.EigenGate],
        *,
        exponents: Sequence[Union[value.Symbol, float]] = (
            0, 1, -1, 0.5, 0.25, -0.5, 0.1, value.Symbol('s')),
        global_shifts: Sequence[float] = (0, 0.5, -0.5, 0.1),
        qubit_count: Optional[int] = None,
        setup_code: str = 'import cirq\nimport numpy as np'
        ) -> None:
    """Checks that an EigenGate subclass is internally consistent and has a
    good __repr__."""
    for exponent in exponents:
        for shift in global_shifts:
            _assert_meets_standards_helper(
                    eigen_gate_type(exponent=exponent, global_shift=shift),
                    qubit_count,
                    setup_code)


def assert_eigen_shifts_is_consistent_with_eigen_components(
        val: ops.EigenGate) -> None:
    assert val._eigen_shifts() == [e[0] for e in val._eigen_components()]


def _assert_meets_standards_helper(val: Any,
                                   qubit_count: Optional[int],
                                   setup_code: str) -> None:
    assert_has_consistent_apply_unitary(val, qubit_count=qubit_count)
    assert_qasm_is_consistent_with_unitary(val)
    assert_decompose_is_consistent_with_unitary(val)
    assert_phase_by_is_consistent_with_unitary(val)
    assert_equivalent_repr(val, setup_code=setup_code)
    if isinstance(val, ops.EigenGate):
        assert_eigen_shifts_is_consistent_with_eigen_components(val)
