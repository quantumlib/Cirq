# Copyright 2019 The Cirq Developers
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
import abc
import inspect

import pytest

import cirq
import cirq.protocols
import io


def assert_roundtrip(obj, text_should_be=None):
    buffer = io.StringIO()
    cirq.protocols.to_json(obj, buffer)

    if text_should_be is not None:
        buffer.seek(0)
        text = buffer.read()

        print()
        print(text)

        assert text == text_should_be

    buffer.seek(0)
    obj2 = cirq.protocols.read_json(buffer)
    assert obj == obj2


def test_line_qubit_roundtrip():
    q1 = cirq.LineQubit(12)
    assert_roundtrip(q1, text_should_be="""{
  "cirq_type": "LineQubit",
  "x": 12
}""")


def test_gridqubit_roundtrip():
    q = cirq.GridQubit(15, 18)
    assert_roundtrip(q, text_should_be="""{
  "cirq_type": "GridQubit",
  "row": 15,
  "col": 18
}""")


def test_op_roundtrip():
    q = cirq.LineQubit(5)
    op1 = cirq.Rx(.123).on(q)
    assert_roundtrip(op1, text_should_be="""{
  "cirq_type": "GateOperation",
  "gate": {
    "cirq_type": "XPowGate",
    "exponent": 0.03915211600060625,
    "global_shift": -0.5
  },
  "qubits": [
    {
      "cirq_type": "LineQubit",
      "x": 5
    }
  ]
}""")


TEST_OBJECTS = {
    'CCXPowGate': [cirq.CCNOT, cirq.TOFFOLI, cirq.CCX ** 0.123],
    'CCZPowGate': cirq.CCZ,
    'CNotPowGate': cirq.CNOT,
    'CSwapGate': cirq.CSWAP,
    'CZPowGate': cirq.CZ,
    'FSimGate': cirq.FSimGate(theta=0.123, phi=.456),
    'GateOperation': [
        cirq.CCNOT(*cirq.LineQubit.range(3)),
        cirq.CCZ(*cirq.LineQubit.range(3)),
        cirq.CNOT(*cirq.LineQubit.range(2)),
        # TODO: https://github.com/quantumlib/Cirq/issues/1972
        # cirq.CSWAP(*cirq.LineQubit.range(3)),
        cirq.CZ(*cirq.LineQubit.range(2))
    ],
    'GlobalPhaseOperation': cirq.GlobalPhaseOperation(-1j),
    'GridQubit': cirq.GridQubit(10, 11),
    'HPowGate': [cirq.H, cirq.H ** 0.123],
    'ISwapPowGate': [cirq.ISWAP, cirq.ISWAP ** 0.123],
    'IdentityGate': [cirq.I, cirq.IdentityGate(num_qubits=5)],
    'LineQubit': [cirq.LineQubit(0), cirq.LineQubit(123)],
    'MeasurementGate': [
        cirq.MeasurementGate(num_qubits=3, key='z'),
        cirq.MeasurementGate(num_qubits=3, key='z',
                             invert_mask=(True, False, True)),
    ],
    '_PauliX': cirq.X,
    '_PauliY': cirq.Y,
    '_PauliZ': cirq.Z,
    'SwapPowGate': [cirq.SWAP, cirq.SWAP ** 0.5],
    'XPowGate': cirq.X ** 0.123,
    'XXPowGate': [cirq.XX, cirq.XX ** 0.123],
    'YPowGate': cirq.Y ** 0.456,
    'YYPowGate': [cirq.YY, cirq.YY ** 0.456],
    'ZPowGate': cirq.Z ** 0.789,
    'ZZPowGate': [cirq.ZZ, cirq.ZZ ** 0.789],
    'complex': [1 + 2j],
}


def _get_all_public_classes():
    for cls_name, cls_cls in inspect.getmembers(cirq, inspect.isclass):
        if cls_name.startswith('_'):
            continue

        if inspect.isabstract(cls_cls) or issubclass(cls_cls, abc.ABCMeta):
            continue

        yield cls_name, cls_cls

    # extras
    yield '_PauliX', cirq.ops.pauli_gates._PauliX
    yield '_PauliY', cirq.ops.pauli_gates._PauliY
    yield '_PauliZ', cirq.ops.pauli_gates._PauliZ
    yield 'complex', complex


NOT_YET_SERIALIZABLE = [
    'AmplitudeDampingChannel',
    'ApplyChannelArgs',
    'ApplyUnitaryArgs',
    'ApproxPauliStringExpectation',
    'AsymmetricDepolarizingChannel',
    'AxisAngleDecomposition',
    'BitFlipChannel',
    'Circuit',  # TODO
    'CircuitDag',
    'CircuitDiagramInfo',
    'CircuitDiagramInfoArgs',
    'CircuitSampleJob',
    'ComputeDisplaysResult',
    'ConstantQubitNoiseModel',
    'ControlledGate',  # TODO
    'ControlledOperation',  # TODO
    'ConvertToCzAndSingleGates',
    'ConvertToIonGates',
    'ConvertToNeutralAtomGates',
    'DensityMatrixSimulator',
    'DensityMatrixSimulatorState',
    'DensityMatrixStepResult',
    'DensityMatrixTrialResult',
    'DepolarizingChannel',
    'DropEmptyMoments',
    'DropNegligible',
    'Duration',
    'EjectPhasedPaulis',
    'EjectZ',
    'ExpandComposite',
    'GeneralizedAmplitudeDampingChannel',
    'Heatmap',
    'InsertStrategy',
    'InterchangeableQubitsGate',
    'IonDevice',
    'KakDecomposition',
    'LinearCombinationOfGates',  # TODO
    'LinearCombinationOfOperations',  # TODO
    'LinearDict',  # TODO
    'Linspace',
    'MergeInteractions',
    'MergeSingleQubitGates',
    'Moment',  # TODO
    'NamedQubit',  # TODO
    'NeutralAtomDevice',
    'ParallelGateOperation',
    'ParamResolver',
    'Pauli',  # TODO, should be excluded
    'PauliInteractionGate',
    'PauliString',  # TODO
    'PauliStringExpectation',
    'PauliStringPhasor',
    'PauliSum',  # TODO
    'PauliSumCollector',
    'PauliTransform',
    'PeriodicValue',
    'PhaseDampingChannel',
    'PhaseFlipChannel',
    'PhasedXPowGate',  # TODO
    'PointOptimizationSummary',
    'PointOptimizer',
    'Points',
    'Product',
    'QasmArgs',
    'QasmOutput',
    'QubitOrder',
    'ResetChannel',
    'Schedule',
    'ScheduledOperation',
    'SimulationTrialResult',
    'Simulator',
    'SingleQubitCliffordGate',
    'SingleQubitGate',
    'SingleQubitMatrixGate',
    'SingleQubitPauliStringGateOperation',
    'SparseSimulatorStep',
    'StateVectorMixin',
    'SupportsApplyChannel',
    'SupportsApproximateEquality',
    'SupportsChannel',
    'SupportsCircuitDiagramInfo',
    'SupportsConsistentApplyUnitary',
    'SupportsDecompose',
    'SupportsDecomposeWithQubits',
    'SupportsExplicitNumQubits',
    'SupportsExplicitQidShape',
    'SupportsMixture',
    'SupportsParameterization',
    'SupportsPhase',
    'SupportsQasm',
    'SupportsQasmWithArgs',
    'SupportsQasmWithArgsAndQubits',
    'SupportsTraceDistanceBound',
    'SupportsUnitary',
    'TextDiagramDrawer',
    'ThreeQubitGate',  # TODO
    'Timestamp',
    'TrialResult',
    'TwoQubitGate',  # TODO
    'TwoQubitMatrixGate',
    'Unique',
    'WaveFunctionSimulatorState',
    'WaveFunctionTrialResult',
    'Zip',
]


@pytest.mark.parametrize('cirq_type,cls', _get_all_public_classes())
def test_all_roundtrip(cirq_type: str, cls):
    if cirq_type == 'CSwapGate':
        return pytest.xfail(reason='https://github.com/quantumlib/Cirq/issues/1972')

    if cirq_type in NOT_YET_SERIALIZABLE:
        return pytest.xfail(reason="Not serializable (yet)")

    objs = TEST_OBJECTS[cirq_type]
    if not isinstance(objs, list):
        objs = [objs]

    for obj in objs:
        assert isinstance(obj, cls)

        # more strict: must be exact (no subclasses)
        assert type(obj) == cls
        assert_roundtrip(obj)
