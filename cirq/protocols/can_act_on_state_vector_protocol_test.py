# Copyright 2020 The Cirq Developers
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


def test_inconclusive():

    class No:
        pass

    assert not cirq.can_act_on_state_vector(object())
    assert not cirq.can_act_on_state_vector('boo')
    assert not cirq.can_act_on_state_vector(No())


def test_via_can_act():

    class No1:

        def _can_act_on_state_vector_(self):
            return NotImplemented

    class No2:

        def _can_act_on_state_vector_(self):
            return None

    class No3:

        def _can_act_on_state_vector_(self):
            return False

    class Yes1:

        def _can_act_on_state_vector_(self):
            return True

    assert not cirq.can_act_on_state_vector(No1())
    assert not cirq.can_act_on_state_vector(No2())
    assert not cirq.can_act_on_state_vector(No3())
    assert cirq.can_act_on_state_vector(Yes1())


def test_common_cases():
    q = cirq.LineQubit(0)
    assert cirq.can_act_on_state_vector(cirq.X)
    assert cirq.can_act_on_state_vector(cirq.X(q))
    assert cirq.can_act_on_state_vector(cirq.MeasurementGate(num_qubits=1))
    assert cirq.can_act_on_state_vector(cirq.measure(q))
    assert cirq.can_act_on_state_vector(cirq.depolarize(0.1))
    assert cirq.can_act_on_state_vector(cirq.depolarize(0.1).on(q))
    assert cirq.can_act_on_state_vector(cirq.reset(q))
    assert cirq.can_act_on_state_vector(cirq.ResetChannel())


def test_via_decompose():

    class YesAtom(cirq.Operation):

        @property
        def qubits(self):
            return cirq.LineQubit.range(2)

        def with_qubits(self):
            raise NotImplementedError()

        def _can_act_on_state_vector_(self):
            return True

    class NoAtom(cirq.Operation):

        @property
        def qubits(self):
            return cirq.LineQubit.range(2)

        def with_qubits(self):
            raise NotImplementedError()

        def _can_act_on_state_vector_(self):
            return False

    class Yes1:

        def _decompose_(self):
            return []

    class Yes2:

        def _decompose_(self):
            return [YesAtom(), YesAtom()]

    class No1:

        def _decompose_(self):
            return [NoAtom()]

    class No2:

        def _decompose_(self):
            return None

    class No3:

        def _decompose_(self):
            return NotImplemented

    class No4:

        def _decompose_(self):
            return [NoAtom(), YesAtom()]

    assert cirq.can_act_on_state_vector(Yes1())
    assert cirq.can_act_on_state_vector(Yes2())
    assert not cirq.can_act_on_state_vector(No1())
    assert not cirq.can_act_on_state_vector(No2())
    assert not cirq.can_act_on_state_vector(No3())
    assert not cirq.can_act_on_state_vector(No4())


def test_only_calls_decompose_once():

    calls = 0

    class NoAtom(cirq.Operation):

        @property
        def qubits(self):
            raise NotImplementedError()

        def with_qubits(self):
            raise NotImplementedError()

        def _can_act_on_state_vector_(self):
            return False

    class Count(cirq.Gate):

        def num_qubits(self) -> int:
            return 1

        def _decompose_(self, qubits):
            nonlocal calls
            calls += 1
            return [NoAtom()]

    assert not cirq.can_act_on_state_vector(Count().on(cirq.LineQubit(0)))
    assert calls == 1


def test_via_has_unitary():

    class No1:

        def _has_unitary_(self):
            return NotImplemented

    class No2:

        def _has_unitary_(self):
            return None

    class No3:

        def _has_unitary_(self):
            return False

    class Yes1:

        def _has_unitary_(self):
            return True

    class Yes2:

        def _unitary_(self):
            return np.eye(2)

    assert not cirq.can_act_on_state_vector(No1())
    assert not cirq.can_act_on_state_vector(No2())
    assert not cirq.can_act_on_state_vector(No3())
    assert cirq.can_act_on_state_vector(Yes1())
    assert cirq.can_act_on_state_vector(Yes2())


def test_order():

    class Yes1:

        def _can_act_on_state_vector_(self):
            return True

        def _has_unitary_(self):
            assert False

        def _decompose_(self):
            assert False

    class Yes2:

        def _can_act_on_state_vector_(self):
            return None

        def _has_unitary_(self):
            return True

        def _decompose_(self):
            assert False

    class Yes3:

        def _can_act_on_state_vector_(self):
            return None

        def _has_unitary_(self):
            return False

        def _decompose_(self):
            return []

    assert cirq.can_act_on_state_vector(Yes1())
    assert cirq.can_act_on_state_vector(Yes2())
    assert cirq.can_act_on_state_vector(Yes3())
