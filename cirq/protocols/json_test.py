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


QUBITS = cirq.LineQubit.range(5)
Q0, Q1, Q2, Q3, Q4 = QUBITS

TEST_OBJECTS = {
    'CCNOT': cirq.CCNOT,
    'CCX': cirq.CCX,
    'CCXPowGate': cirq.CCXPowGate(exponent=0.123, global_shift=0.456),
    'CCZ': cirq.CCZ,
    'CCZPowGate': cirq.CCZPowGate(exponent=0.123, global_shift=0.456),
    'CNOT': cirq.CNOT,
    'CNotPowGate': cirq.CNotPowGate(exponent=0.123, global_shift=0.456),
    'CX': cirq.CX,
    'CSWAP': cirq.CSWAP,
    'CSwapGate': cirq.CSwapGate(),
    'CZ': cirq.CZ,
    'CZPowGate': cirq.CZPowGate(exponent=0.123, global_shift=0.456),
    'Circuit': [
        cirq.Circuit.from_ops(cirq.H.on_each(QUBITS),
                              cirq.measure(*QUBITS)),
    ],
    'FREDKIN': cirq.FREDKIN,
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
    'H': cirq.H,
    'HPowGate': [cirq.HPowGate(exponent=-8), cirq.H ** 0.123],
    'I': cirq.I,
    'ISWAP': cirq.ISWAP,
    'ISwapPowGate': [cirq.ISwapPowGate(exponent=-8), cirq.ISWAP ** 0.123],
    'IdentityGate': [cirq.I, cirq.IdentityGate(num_qubits=5)],
    'LineQubit': [cirq.LineQubit(0), cirq.LineQubit(123)],
    'MeasurementGate': [
        cirq.MeasurementGate(num_qubits=3, key='z'),
        cirq.MeasurementGate(num_qubits=3, key='z',
                             invert_mask=(True, False, True)),
    ],
    'Moment': [
        cirq.Moment(operations=[cirq.X(Q0), cirq.Y(Q1), cirq.Z(Q2)]),
    ],
    'X': cirq.X,
    'Y': cirq.Y,
    'Z': cirq.Z,
    'S': cirq.S,
    'SWAP': cirq.SWAP,
    'SwapPowGate': [cirq.SwapPowGate(), cirq.SWAP ** 0.5],
    'T': cirq.T,
    'TOFFOLI': cirq.TOFFOLI,
    'UnconstrainedDevice': cirq.UnconstrainedDevice,
    'XPowGate': cirq.X ** 0.123,
    'XX': cirq.XX,
    'XXPowGate': [cirq.XXPowGate(), cirq.XX ** 0.123],
    'YPowGate': cirq.Y ** 0.456,
    'YY': cirq.YY,
    'YYPowGate': [cirq.YYPowGate(), cirq.YY ** 0.456],
    'ZPowGate': cirq.Z ** 0.789,
    'ZZ': cirq.ZZ,
    'ZZPowGate': [cirq.ZZPowGate(), cirq.ZZ ** 0.789],
    'complex': [1 + 2j],
}


def _get_all_public_classes():
    for cls_name, cls_cls in inspect.getmembers(cirq):
        if inspect.isfunction(cls_cls) or inspect.ismodule(cls_cls):
            continue


        if not inspect.isclass(cls_cls):
            print(cls_name, '- not a class, but lets test anyway')
            cls_cls = cls_cls.__class__


        if cls_name.startswith('_'):
            continue

        if (inspect.isclass(cls_cls)
                and (inspect.isabstract(cls_cls)
                     or issubclass(cls_cls, abc.ABCMeta))):
            continue

        yield cls_name, cls_cls

    # extra
    yield 'complex', complex


NOT_YET_SERIALIZABLE = [
    'AmplitudeDampingChannel',
    'ApplyChannelArgs',
    'ApplyUnitaryArgs',
    'ApproxPauliStringExpectation',
    'AsymmetricDepolarizingChannel',
    'AxisAngleDecomposition',
    'BitFlipChannel',
    'CircuitDag',
    'CircuitDiagramInfo',
    'CircuitDiagramInfoArgs',
    'CircuitSampleJob',
    'ComputeDisplaysResult',
    'ConstantQubitNoiseModel',
    'ControlledGate',  # TODO
    'ControlledOperation',  # TODO
    'CONTROL_TAG',
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
    'IonDevice', # TODO
    'KakDecomposition',
    'LinearCombinationOfGates',  # TODO
    'LinearCombinationOfOperations',  # TODO
    'LinearDict',  # TODO
    'Linspace',
    'MergeInteractions',
    'MergeSingleQubitGates',
    'NamedQubit',  # TODO
    'NeutralAtomDevice', # TODO
    'NO_NOISE',
    'OP_TREE',
    'ParallelGateOperation',
    'ParamResolver',
    'ParamResolverOrSimilarType', # to-not-do: type
    'PAULI_BASIS', # TODO
    'Pauli',  # TODO, should be excluded
    'PauliInteractionGate',
    'PauliString',  # TODO
    'PauliStringExpectation',
    'PauliStringPhasor',
    'PauliSum',  # TODO
    'PauliSumLike', # to-not-do: type
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
    'QubitOrderOrList', # to-not-do: type
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
    'Sweepable', # to-not-do: type
    'TextDiagramDrawer',
    'ThreeQubitGate',  # TODO
    'Timestamp',
    'TrialResult',
    'TwoQubitGate',  # TODO
    'TwoQubitMatrixGate',
    'UnitSweep',
    'Unique',
    'WaveFunctionSimulatorState',
    'WaveFunctionTrialResult',
    'Zip',
    'reset',
]


@pytest.mark.parametrize('cirq_type,cls', _get_all_public_classes())
def test_all_roundtrip(cirq_type: str, cls):
    if cirq_type == 'CSwapGate' or cirq_type == 'CSWAP' or cirq_type == 'FREDKIN':
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
