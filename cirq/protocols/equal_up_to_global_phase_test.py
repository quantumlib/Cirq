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
import numpy as np
import cirq


def test_equal_up_to_global_phase_primitives():
    assert cirq.equal_up_to_global_phase(1.0 + 1j, 1.0 + 1j, atol=1e-09)
    assert cirq.equal_up_to_global_phase(1.0 + 1j, 1.0 - 1j, atol=1e-09)
    assert cirq.equal_up_to_global_phase(np.exp(1j*3.3), 1.0, atol=1e-09)
    assert not cirq.equal_up_to_global_phase(1j, 1e-10 + 1j, atol=1e-11)
    assert cirq.equal_up_to_global_phase(0.0, 1e-10, atol=1e-09)
    assert cirq.equal_up_to_global_phase(0, 0, atol=1e-09)
    # mixed types - see FIXME
    # assert cirq.equal_up_to_global_phase(0, 1e-10j, atol=1e-09)
    # assert cirq.equal_up_to_global_phase(0.0, 1e-10j, atol=1e-09)


def test_approx_eq_list():
    assert cirq.equal_up_to_global_phase([], [], atol=0.0)
    assert not cirq.equal_up_to_global_phase([], [[]], atol=0.0)

    assert cirq.equal_up_to_global_phase([1j, 1], [1j, 1])
    assert cirq.equal_up_to_global_phase([1j, 1j], [1, 1])
    assert cirq.equal_up_to_global_phase([1j, 1], [1, -1j])
    assert not cirq.equal_up_to_global_phase([1j, -1j], [1, 1])
    assert not cirq.equal_up_to_global_phase([1j, 1], [1, 1j])

    assert not cirq.equal_up_to_global_phase([1j, 1], [1j, 1, 0])


# def test_approx_eq_iterables():
#     def gen_1_1():
#         yield 1
#         yield 1
#     assert cirq.approx_eq((1, 1), [1, 1], atol=0.0)
#     assert cirq.approx_eq((1, 1), gen_1_1(), atol=0.0)
#     assert cirq.approx_eq(gen_1_1(), [1, 1], atol=0.0)
#
#

# Dummy container class implementing _equal_up_to_global_phase_
# for homogeneous comparison
class A:

    def __init__(self, val):
        self.val = [val]

    def _equal_up_to_global_phase_(self, other, atol):
        if not isinstance(self, type(other)):
            return NotImplemented
        return cirq.equal_up_to_global_phase(
            self.val[0], other.val[0], atol=atol
        )


# Dummy container class implementing _equal_up_to_global_phase_
# for heterogeneous comparison
class B:

    def __init__(self, val):
        self.val = [val]

    def _equal_up_to_global_phase_(self, other, atol):
        if not isinstance(self.val[0], type(other)):
            return NotImplemented
        return cirq.equal_up_to_global_phase(self.val[0], other, atol=atol)


def test_approx_eq_supported():
    assert cirq.equal_up_to_global_phase(A(0.1), A(0.1j), atol=0.1)
    assert not cirq.equal_up_to_global_phase(A(0.0), A(0.1j), atol=0.0)
    assert cirq.equal_up_to_global_phase(B(0.0j), 1e-8j, atol=1e-8)
    assert cirq.equal_up_to_global_phase(1e-8j, B(0.0j), atol=1e-8)
    assert not cirq.equal_up_to_global_phase(1e-8j, B(0.0), atol=1e-10)
