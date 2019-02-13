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
import pytest

import cirq


def test_circuit_diagram_info_value_wrapping():
    single_info = cirq.CircuitDiagramInfo(('Single',))

    class ReturnInfo:
        def _circuit_diagram_info_(self, args):
            return single_info

    class ReturnTuple:
        def _circuit_diagram_info_(self, args):
            return 'Single',

    class ReturnList:
        def _circuit_diagram_info_(self, args):
            return ('Single' for _ in range(1))

    class ReturnGenerator:
        def _circuit_diagram_info_(self, args):
            return ['Single']

    class ReturnString:
        def _circuit_diagram_info_(self, args):
            return 'Single'

    assert (cirq.circuit_diagram_info(ReturnInfo()) ==
            cirq.circuit_diagram_info(ReturnTuple()) ==
            cirq.circuit_diagram_info(ReturnString()) ==
            cirq.circuit_diagram_info(ReturnList()) ==
            cirq.circuit_diagram_info(ReturnGenerator()) ==
            single_info)

    double_info = cirq.CircuitDiagramInfo(('Single', 'Double',))

    class ReturnDoubleInfo:
        def _circuit_diagram_info_(self, args):
            return double_info

    class ReturnDoubleTuple:
        def _circuit_diagram_info_(self, args):
            return 'Single', 'Double'

    assert (cirq.circuit_diagram_info(ReturnDoubleInfo()) ==
            cirq.circuit_diagram_info(ReturnDoubleTuple()) ==
            double_info)


def test_circuit_diagram_info_validate():
    with pytest.raises(ValueError):
        _ = cirq.CircuitDiagramInfo('X')


def test_circuit_diagram_info_repr():
    cirq.testing.assert_equivalent_repr(
        cirq.CircuitDiagramInfo(('X', 'Y'), 2))


def test_circuit_diagram_info_eq():
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: cirq.CircuitDiagramInfo(('X',)))
    eq.add_equality_group(cirq.CircuitDiagramInfo(('X', 'Y')),
                          cirq.CircuitDiagramInfo(('X', 'Y'), 1))
    eq.add_equality_group(cirq.CircuitDiagramInfo(('Z',), 2))
    eq.add_equality_group(cirq.CircuitDiagramInfo(('Z', 'Z'), 2))
    eq.add_equality_group(cirq.CircuitDiagramInfo(('Z',), 3))


def test_circuit_diagram_info_pass_fail():
    class C:
        pass

    class D:
        def _circuit_diagram_info_(self, args):
            return NotImplemented

    class E:
        def _circuit_diagram_info_(self, args):
            return cirq.CircuitDiagramInfo(('X',))

    assert cirq.circuit_diagram_info(C(), default=None) is None
    assert cirq.circuit_diagram_info(D(), default=None) is None
    assert cirq.circuit_diagram_info(
        E(), default=None) == cirq.CircuitDiagramInfo(('X',))

    with pytest.raises(TypeError, match='no _circuit_diagram_info'):
        _ = cirq.circuit_diagram_info(C())
    with pytest.raises(TypeError, match='returned NotImplemented'):
        _ = cirq.circuit_diagram_info(D())
    assert cirq.circuit_diagram_info(E()) == cirq.CircuitDiagramInfo(('X',))


def test_circuit_diagram_info_args_eq():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.CircuitDiagramInfoArgs.UNINFORMED_DEFAULT)
    eq.add_equality_group(cirq.CircuitDiagramInfoArgs(
        known_qubits=None,
        known_qubit_count=None,
        use_unicode_characters=False,
        precision=None,
        qubit_map=None))
    eq.add_equality_group(cirq.CircuitDiagramInfoArgs(
        known_qubits=None,
        known_qubit_count=None,
        use_unicode_characters=True,
        precision=None,
        qubit_map=None))
    eq.add_equality_group(cirq.CircuitDiagramInfoArgs(
        known_qubits=cirq.LineQubit.range(3),
        known_qubit_count=3,
        use_unicode_characters=False,
        precision=None,
        qubit_map=None))
    eq.add_equality_group(cirq.CircuitDiagramInfoArgs(
        known_qubits=cirq.LineQubit.range(2),
        known_qubit_count=2,
        use_unicode_characters=False,
        precision=None,
        qubit_map=None))
    eq.add_equality_group(cirq.CircuitDiagramInfoArgs(
        known_qubits=cirq.LineQubit.range(2),
        known_qubit_count=2,
        use_unicode_characters=False,
        precision=None,
        qubit_map={cirq.LineQubit(0): 5, cirq.LineQubit(1): 7}))


def test_circuit_diagram_info_args_repr():
    cirq.testing.assert_equivalent_repr(
        cirq.CircuitDiagramInfoArgs(
            known_qubits=cirq.LineQubit.range(2),
            known_qubit_count=2,
            use_unicode_characters=True,
            precision=5,
            qubit_map={cirq.LineQubit(0): 5, cirq.LineQubit(1): 7}))
