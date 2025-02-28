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

import itertools
from typing import Any, Dict, Optional, Sequence, Type

import numpy as np
import sympy

from cirq import ops, protocols, value
from cirq.testing.consistent_act_on import assert_all_implemented_act_on_effects_match_unitary
from cirq.testing.circuit_compare import (
    assert_has_consistent_apply_unitary,
    assert_has_consistent_qid_shape,
)
from cirq.testing.consistent_decomposition import (
    assert_decompose_is_consistent_with_unitary,
    assert_decompose_ends_at_default_gateset,
)
from cirq.testing.consistent_phase_by import assert_phase_by_is_consistent_with_unitary
from cirq.testing.consistent_qasm import assert_qasm_is_consistent_with_unitary
from cirq.testing.consistent_pauli_expansion import (
    assert_pauli_expansion_is_consistent_with_unitary,
)
from cirq.testing.consistent_resolve_parameters import assert_consistent_resolve_parameters
from cirq.testing.consistent_specified_has_unitary import assert_specifies_has_unitary_if_unitary
from cirq.testing.equivalent_repr_eval import assert_equivalent_repr
from cirq.testing.consistent_controlled_gate_op import (
    assert_controlled_and_controlled_by_identical,
    assert_controlled_unitary_consistent,
)
from cirq.testing.consistent_unitary import assert_unitary_is_consistent


def assert_implements_consistent_protocols(
    val: Any,
    *,
    exponents: Sequence[Any] = (0, 1, -1, 0.25, sympy.Symbol('s')),
    qubit_count: Optional[int] = None,
    ignoring_global_phase: bool = False,
    setup_code: str = 'import cirq\nimport numpy as np\nimport sympy',
    global_vals: Optional[Dict[str, Any]] = None,
    local_vals: Optional[Dict[str, Any]] = None,
    ignore_decompose_to_default_gateset: bool = False,
) -> None:
    """Checks that a value is internally consistent and has a good __repr__."""
    global_vals = global_vals or {}
    local_vals = local_vals or {}

    _assert_meets_standards_helper(
        val,
        ignoring_global_phase=ignoring_global_phase,
        setup_code=setup_code,
        global_vals=global_vals,
        local_vals=local_vals,
        ignore_decompose_to_default_gateset=ignore_decompose_to_default_gateset,
    )

    for exponent in exponents:
        p = protocols.pow(val, exponent, None)
        if p is not None:
            _assert_meets_standards_helper(
                val**exponent,
                ignoring_global_phase=ignoring_global_phase,
                setup_code=setup_code,
                global_vals=global_vals,
                local_vals=local_vals,
                ignore_decompose_to_default_gateset=ignore_decompose_to_default_gateset,
            )


def assert_eigengate_implements_consistent_protocols(
    eigen_gate_type: Type[ops.EigenGate],
    *,
    exponents: Sequence[value.TParamVal] = (0, 1, -1, 0.25, sympy.Symbol('s')),
    global_shifts: Sequence[float] = (0, -0.5, 0.1),
    qubit_count: Optional[int] = None,
    ignoring_global_phase: bool = False,
    setup_code: str = 'import cirq\nimport numpy as np\nimport sympy',
    global_vals: Optional[Dict[str, Any]] = None,
    local_vals: Optional[Dict[str, Any]] = None,
    ignore_decompose_to_default_gateset: bool = False,
) -> None:
    """Checks that an EigenGate subclass is internally consistent and has a
    good __repr__."""
    # pylint: disable=unused-variable
    __tracebackhide__ = True
    # pylint: enable=unused-variable

    for exponent in exponents:
        for shift in global_shifts:
            _assert_meets_standards_helper(
                eigen_gate_type(exponent=exponent, global_shift=shift),
                ignoring_global_phase=ignoring_global_phase,
                setup_code=setup_code,
                global_vals=global_vals,
                local_vals=local_vals,
                ignore_decompose_to_default_gateset=ignore_decompose_to_default_gateset,
            )


def assert_eigen_shifts_is_consistent_with_eigen_components(val: ops.EigenGate) -> None:
    # pylint: disable=unused-variable
    __tracebackhide__ = True
    # pylint: enable=unused-variable
    if not protocols.is_parameterized(val):
        assert val._eigen_shifts() == [
            e[0] for e in val._eigen_components()
        ], "_eigen_shifts not consistent with _eigen_components"


def assert_has_consistent_trace_distance_bound(val: Any) -> None:
    # pylint: disable=unused-variable
    __tracebackhide__ = True
    # pylint: enable=unused-variable
    u = protocols.unitary(val, default=None)
    val_from_trace = protocols.trace_distance_bound(val)
    assert 0.0 <= val_from_trace <= 1.0
    if u is not None:

        class Unitary:
            def _unitary_(self):
                return u

        val_from_unitary = protocols.trace_distance_bound(Unitary())

        assert val_from_trace >= val_from_unitary or np.isclose(val_from_trace, val_from_unitary)


def _assert_meets_standards_helper(
    val: Any,
    *,
    ignoring_global_phase: bool,
    setup_code: str,
    global_vals: Optional[Dict[str, Any]],
    local_vals: Optional[Dict[str, Any]],
    ignore_decompose_to_default_gateset: bool,
) -> None:
    __tracebackhide__ = True  # pylint: disable=unused-variable

    assert_consistent_resolve_parameters(val)
    assert_specifies_has_unitary_if_unitary(val)
    assert_has_consistent_qid_shape(val)
    assert_has_consistent_apply_unitary(val)
    assert_all_implemented_act_on_effects_match_unitary(val)
    assert_qasm_is_consistent_with_unitary(val)
    assert_has_consistent_trace_distance_bound(val)
    assert_decompose_is_consistent_with_unitary(val, ignoring_global_phase=ignoring_global_phase)
    assert_unitary_is_consistent(val, ignoring_global_phase=ignoring_global_phase)
    if not ignore_decompose_to_default_gateset:
        assert_decompose_ends_at_default_gateset(val)
    assert_phase_by_is_consistent_with_unitary(val)
    assert_pauli_expansion_is_consistent_with_unitary(val)
    assert_equivalent_repr(
        val, setup_code=setup_code, global_vals=global_vals, local_vals=local_vals
    )
    assert protocols.measurement_key_objs(val) == protocols.measurement_key_names(val)
    if isinstance(val, ops.EigenGate):
        assert_eigen_shifts_is_consistent_with_eigen_components(val)
    if isinstance(val, ops.Gate) and protocols.has_mixture(val):
        assert_controlled_and_controlled_by_identical(val)
        if protocols.has_unitary(val):
            assert_controlled_unitary_consistent(val)


def assert_commutes_magic_method_consistent_with_unitaries(
    *vals: Sequence[Any], atol: float = 1e-8
) -> None:
    if any(isinstance(val, ops.Operation) for val in vals):
        raise TypeError('`_commutes_` need not be consistent with unitaries for `Operation`.')
    unitaries = [protocols.unitary(val, None) for val in vals]
    pairs = itertools.permutations(zip(vals, unitaries), 2)
    for (left_val, left_unitary), (right_val, right_unitary) in pairs:
        if left_unitary is None or right_unitary is None:
            continue
        commutes = protocols.commutes(left_val, right_val, atol=atol, default=None)
        if commutes is None:
            continue
        assert commutes == protocols.commutes(left_unitary, right_unitary)
