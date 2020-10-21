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
import inspect

import io
import json
import os
import pathlib
import textwrap
from typing import Tuple, Iterator, Type, List, Set, Any

import pytest

import numpy as np
import pandas as pd
import sympy

import cirq
from cirq._compat import proper_repr, proper_eq
from cirq.testing import assert_json_roundtrip_works
from cirq.protocols.json_serialization import RESOLVER_CACHE

TEST_DATA_PATH = pathlib.Path(__file__).parent / 'json_test_data'
TEST_DATA_REL = 'cirq/protocols/json_test_data'


def test_line_qubit_roundtrip():
    q1 = cirq.LineQubit(12)
    assert_json_roundtrip_works(q1,
                                text_should_be="""{
  "cirq_type": "LineQubit",
  "x": 12
}""")


def test_gridqubit_roundtrip():
    q = cirq.GridQubit(15, 18)
    assert_json_roundtrip_works(q,
                                text_should_be="""{
  "cirq_type": "GridQubit",
  "row": 15,
  "col": 18
}""")


def test_op_roundtrip():
    q = cirq.LineQubit(5)
    op1 = cirq.rx(.123).on(q)
    assert_json_roundtrip_works(op1,
                                text_should_be="""{
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


def test_op_roundtrip_filename(tmpdir):
    filename = f'{tmpdir}/op.json'
    q = cirq.LineQubit(5)
    op1 = cirq.rx(.123).on(q)
    cirq.to_json(op1, filename)
    assert os.path.exists(filename)
    op2 = cirq.read_json(filename)
    assert op1 == op2


def test_fail_to_resolve():
    buffer = io.StringIO()
    buffer.write("""
    {
      "cirq_type": "MyCustomClass",
      "data": [1, 2, 3]
    }
    """)
    buffer.seek(0)

    with pytest.raises(ValueError) as e:
        cirq.read_json(buffer)
    assert e.match("Could not resolve type 'MyCustomClass' "
                   "during deserialization")


QUBITS = cirq.LineQubit.range(5)
Q0, Q1, Q2, Q3, Q4 = QUBITS

# TODO: Include cirq.rx in the Circuit test case file.
# Github issue: https://github.com/quantumlib/Cirq/issues/2014
# Note that even the following doesn't work because theta gets
# multiplied by 1/pi:
#   cirq.Circuit(cirq.rx(sympy.Symbol('theta')).on(Q0)),

SHOULDNT_BE_SERIALIZED = [
    # Intermediate states with work buffers and unknown external prng guts.
    'ActOnCliffordTableauArgs',
    'ActOnStabilizerCHFormArgs',
    'ActOnStateVectorArgs',
    'ApplyChannelArgs',
    'ApplyMixtureArgs',
    'ApplyUnitaryArgs',

    # Circuit optimizers are function-like. Only attributes
    # are ignore_failures, tolerance, and other feature flags
    'ConvertToCzAndSingleGates',
    'ConvertToIonGates',
    'ConvertToNeutralAtomGates',
    'ConvertToSqrtIswapGates',
    'ConvertToSycamoreGates',
    'ConvertToXmonGates',
    'DropEmptyMoments',
    'DropNegligible',
    'EjectPhasedPaulis',
    'EjectZ',
    'ExpandComposite',
    'MergeInteractions',
    'MergeSingleQubitGates',
    'PointOptimizer',
    'SynchronizeTerminalMeasurements',

    # global objects
    'CONTROL_TAG',
    'PAULI_BASIS',
    'PAULI_STATES',

    # abstract, but not inspect.isabstract():
    'Device',
    'InterchangeableQubitsGate',
    'Pauli',
    'SingleQubitGate',
    'ThreeQubitGate',
    'TwoQubitGate',
    'ABCMetaImplementAnyOneOf',

    # protocols:
    'SupportsActOn',
    'SupportsApplyChannel',
    'SupportsApplyMixture',
    'SupportsApproximateEquality',
    'SupportsChannel',
    'SupportsCircuitDiagramInfo',
    'SupportsCommutes',
    'SupportsConsistentApplyUnitary',
    'SupportsDecompose',
    'SupportsDecomposeWithQubits',
    'SupportsEqualUpToGlobalPhase',
    'SupportsExplicitHasUnitary',
    'SupportsExplicitNumQubits',
    'SupportsExplicitQidShape',
    'SupportsJSON',
    'SupportsMeasurementKey',
    'SupportsMixture',
    'SupportsParameterization',
    'SupportsPauliExpansion',
    'SupportsPhase',
    'SupportsQasm',
    'SupportsQasmWithArgs',
    'SupportsQasmWithArgsAndQubits',
    'SupportsTraceDistanceBound',
    'SupportsUnitary',

    # mypy types:
    'CIRCUIT_LIKE',
    'DURATION_LIKE',
    'JsonResolver',
    'NOISE_MODEL_LIKE',
    'OP_TREE',
    'PAULI_GATE_LIKE',
    'PAULI_STRING_LIKE',
    'ParamResolverOrSimilarType',
    'PauliSumLike',
    'QubitOrderOrList',
    'RANDOM_STATE_OR_SEED_LIKE',
    'STATE_VECTOR_LIKE',
    'Sweepable',
    'TParamKey',
    'TParamVal',
    'ParamDictType',

    # utility:
    'AnnealSequenceSearchStrategy',
    'CliffordSimulator',
    'DeserializingArg',
    'GateOpDeserializer',
    'GateOpSerializer',
    'GreedySequenceSearchStrategy',
    'SerializingArg',
    'Simulator',
    'StabilizerSampler',
    'Unique',
    'DEFAULT_RESOLVERS',

    # Quantum Engine
    'Engine',
    'EngineJob',
    'EngineProcessor',
    'EngineProgram',
    'EngineTimeSlot',
    'QuantumEngineSampler',
    'NAMED_GATESETS',

    # enums
    'ProtoVersion'
]


def _get_all_public_classes(module) -> Iterator[Tuple[str, Type]]:
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) or inspect.ismodule(obj):
            continue

        if name in SHOULDNT_BE_SERIALIZED:
            continue

        if not inspect.isclass(obj):
            # singletons, for instance
            obj = obj.__class__

        if name.startswith('_'):
            continue

        if inspect.isclass(obj) and inspect.isabstract(obj):
            continue

        # assert name != 'XPowGate'
        yield name, obj


def _get_all_names() -> Iterator[str]:

    def not_module_or_function(x):
        return not (inspect.ismodule(x) or inspect.isfunction(x))

    for name, _ in inspect.getmembers(cirq, not_module_or_function):
        yield name
    for name, _ in inspect.getmembers(cirq.google, not_module_or_function):
        yield name


def test_shouldnt_be_serialized_no_superfluous():
    # everything in the list should be ignored for a reason
    names = set(_get_all_names())
    for name in SHOULDNT_BE_SERIALIZED:
        assert name in names


def test_not_yet_serializable_no_superfluous():
    # everything in the list should be ignored for a reason
    names = set(_get_all_names())
    for name in NOT_YET_SERIALIZABLE:
        assert name in names


def test_mutually_exclusive_blacklist():
    assert len(set(SHOULDNT_BE_SERIALIZED) & set(NOT_YET_SERIALIZABLE)) == 0


NOT_YET_SERIALIZABLE = [
    'AsymmetricDepolarizingChannel',
    'AxisAngleDecomposition',
    'Calibration',
    'CircuitDag',
    'CircuitDiagramInfo',
    'CircuitDiagramInfoArgs',
    'CircuitSampleJob',
    'CliffordSimulatorStepResult',
    'CliffordState',
    'CliffordTrialResult',
    'ConstantQubitNoiseModel',
    'DensityMatrixSimulator',
    'DensityMatrixSimulatorState',
    'DensityMatrixStepResult',
    'DensityMatrixTrialResult',
    'ExpressionMap',
    'FSIM_GATESET',
    'Heatmap',
    'InsertStrategy',
    'IonDevice',
    'KakDecomposition',
    'LinearCombinationOfGates',
    'LinearCombinationOfOperations',
    'Linspace',
    'ListSweep',
    'NeutralAtomDevice',
    'ParallelGateOperation',
    'PauliInteractionGate',
    'PauliStringPhasor',
    'PauliSum',
    'PauliSumCollector',
    'PauliTransform',
    'PeriodicValue',
    'PointOptimizationSummary',
    'Points',
    'Product',
    'QasmArgs',
    'QasmOutput',
    'QubitOrder',
    'QubitPermutationGate',
    'QuilFormatter',
    'QuilOutput',
    'SerializableDevice',
    'SerializableGateSet',
    'SimulationTrialResult',
    'SingleQubitCliffordGate',
    'SparseSimulatorStep',
    'SQRT_ISWAP_GATESET',
    'StateVectorMixin',
    'SYC_GATESET',
    'Sycamore',
    'Sycamore23',
    'TextDiagramDrawer',
    'ThreeQubitDiagonalGate',
    'Timestamp',
    'TwoQubitDiagonalGate',
    'UnitSweep',
    'StateVectorSimulatorState',
    'StateVectorTrialResult',
    'WaveFunctionSimulatorState',
    'WaveFunctionTrialResult',
    'XmonDevice',
    'XMON',
    'ZerosSampler',
    'Zip',
]


def _find_classes_that_should_serialize() -> Set[Tuple[str, Type]]:
    result: Set[Tuple[str, Type]] = set()
    result.update(_get_all_public_classes(cirq))
    result.update(_get_all_public_classes(cirq.google))
    result.update(_get_all_public_classes(cirq.work))

    for k, v in RESOLVER_CACHE.cirq_class_resolver_dictionary.items():
        t = v if isinstance(v, type) else None
        result.add((k, t))
    return result


def test_builtins():
    assert_json_roundtrip_works(True)
    assert_json_roundtrip_works(1)
    assert_json_roundtrip_works(1 + 2j)
    assert_json_roundtrip_works({
        'test': [123, 5.5],
        'key2': 'asdf',
        '3': None,
        '0.0': [],
    })


def test_numpy():
    x = np.ones(1)[0]

    assert_json_roundtrip_works(x.astype(np.bool))
    assert_json_roundtrip_works(x.astype(np.int8))
    assert_json_roundtrip_works(x.astype(np.int16))
    assert_json_roundtrip_works(x.astype(np.int32))
    assert_json_roundtrip_works(x.astype(np.int64))
    assert_json_roundtrip_works(x.astype(np.uint8))
    assert_json_roundtrip_works(x.astype(np.uint16))
    assert_json_roundtrip_works(x.astype(np.uint32))
    assert_json_roundtrip_works(x.astype(np.uint64))
    assert_json_roundtrip_works(x.astype(np.float32))
    assert_json_roundtrip_works(x.astype(np.float64))
    assert_json_roundtrip_works(x.astype(np.complex64))
    assert_json_roundtrip_works(x.astype(np.complex128))

    assert_json_roundtrip_works(np.ones((11, 5)))
    assert_json_roundtrip_works(np.arange(3))


def test_pandas():
    assert_json_roundtrip_works(
        pd.DataFrame(data=[[1, 2, 3], [4, 5, 6]],
                     columns=['x', 'y', 'z'],
                     index=[2, 5]))
    assert_json_roundtrip_works(pd.Index([1, 2, 3], name='test'))
    assert_json_roundtrip_works(
        pd.MultiIndex.from_tuples([(1, 2), (3, 4), (5, 6)],
                                  names=['alice', 'bob']))

    assert_json_roundtrip_works(
        pd.DataFrame(index=pd.Index([1, 2, 3], name='test'),
                     data=[[11, 21.0], [12, 22.0], [13, 23.0]],
                     columns=['a', 'b']))
    assert_json_roundtrip_works(
        pd.DataFrame(index=pd.MultiIndex.from_tuples([(1, 2), (2, 3), (3, 4)],
                                                     names=['x', 'y']),
                     data=[[11, 21.0], [12, 22.0], [13, 23.0]],
                     columns=pd.Index(['a', 'b'], name='c')))


def test_sympy():
    # Raw values.
    assert_json_roundtrip_works(sympy.Symbol('theta'))
    assert_json_roundtrip_works(sympy.Integer(5))
    assert_json_roundtrip_works(sympy.Rational(2, 3))
    assert_json_roundtrip_works(sympy.Float(1.1))

    # Basic operations.
    s = sympy.Symbol('s')
    t = sympy.Symbol('t')
    assert_json_roundtrip_works(t + s)
    assert_json_roundtrip_works(t * s)
    assert_json_roundtrip_works(t / s)
    assert_json_roundtrip_works(t - s)
    assert_json_roundtrip_works(t**s)

    # Linear combinations.
    assert_json_roundtrip_works(t * 2)
    assert_json_roundtrip_works(4 * t + 3 * s + 2)


def _write_test_data(key: str, *test_instances: Any):
    """Helper method for creating initial test data."""
    # coverage: ignore
    cirq.to_json(test_instances, TEST_DATA_PATH / f'{key}.json')
    with open(TEST_DATA_PATH / f'{key}.repr', 'w') as f:
        f.write('[\n')
        for e in test_instances:
            f.write(proper_repr(e))
            f.write(',\n')
        f.write(']')


@pytest.mark.parametrize('cirq_obj_name,cls',
                         _find_classes_that_should_serialize())
def test_json_test_data_coverage(cirq_obj_name: str, cls):
    if cirq_obj_name in NOT_YET_SERIALIZABLE:
        return pytest.xfail(reason="Not serializable (yet)")

    json_path = TEST_DATA_PATH / f'{cirq_obj_name}.json'
    json_path2 = TEST_DATA_PATH / f'{cirq_obj_name}.json_inward'

    if not json_path.exists() and not json_path2.exists():
        # coverage: ignore
        raise NotImplementedError(
            textwrap.fill(
                f"Hello intrepid developer. There is a new public or "
                f"serializable object named '{cirq_obj_name}' that does not "
                f"have associated test data.\n"
                f"\n"
                f"You must create the file\n"
                f"    cirq/protocols/json_test_data/{cirq_obj_name}.json\n"
                f"and the file\n"
                f"    cirq/protocols/json_test_data/{cirq_obj_name}.repr\n"
                f"in order to guarantee this public object is, and will "
                f"remain, serializable.\n"
                f"\n"
                f"The content of the .repr file should be the string returned "
                f"by `repr(obj)` where `obj` is a test {cirq_obj_name} value "
                f"or list of such values. To get this to work you may need to "
                f"implement a __repr__ method for {cirq_obj_name}. The repr "
                f"must be a parsable python expression that evaluates to "
                f"something equal to `obj`."
                f"\n"
                f"The content of the .json file should be the string returned "
                f"by `cirq.to_json(obj)` where `obj` is the same object or "
                f"list of test objects.\n"
                f"To get this to work you likely need "
                f"to add {cirq_obj_name} to the "
                f"`cirq_class_resolver_dictionary` method in "
                f"the cirq/protocols/json_serialization.py source file. "
                f"You may also need to add a _json_dict_ method to "
                f"{cirq_obj_name}. In some cases you will also need to add a "
                f"_from_json_dict_ method to {cirq_obj_name}."
                f"\n"
                f"For more information on JSON serialization, please read the "
                f"docstring for protocols.SupportsJSON. If this object or "
                f"class is not appropriate for serialization, add its name to "
                f"the SHOULDNT_BE_SERIALIZED list in the "
                f"cirq/protocols/json_serialization_test.py source file."))

    repr_file = TEST_DATA_PATH / f'{cirq_obj_name}.repr'
    if repr_file.exists() and cls is not None:
        objs = _eval_repr_data_file(repr_file)
        if not isinstance(objs, list):
            objs = [objs]

        for obj in objs:
            assert type(obj) == cls, (
                f"Value in {TEST_DATA_REL}/{cirq_obj_name}.repr must be of "
                f"exact type {cls}, or a list of instances of that type. But "
                f"the value (or one of the list entries) had type "
                f"{type(obj)}.\n"
                f"\n"
                f"If using a value of the wrong type is intended, move the "
                f"value to {TEST_DATA_REL}/{cirq_obj_name}.repr_inward\n"
                f"\n"
                f"Value with wrong type:\n{obj!r}.")


def test_to_from_strings():
    x_json_text = """{
  "cirq_type": "_PauliX",
  "exponent": 1.0,
  "global_shift": 0.0
}"""
    assert cirq.to_json(cirq.X) == x_json_text
    assert cirq.read_json(json_text=x_json_text) == cirq.X

    with pytest.raises(ValueError, match='specify ONE'):
        cirq.read_json(io.StringIO(), json_text=x_json_text)


def _eval_repr_data_file(path: pathlib.Path):
    return eval(path.read_text(), {
        'cirq': cirq,
        'pd': pd,
        'sympy': sympy,
        'np': np,
    }, {})


def assert_repr_and_json_test_data_agree(repr_path: pathlib.Path,
                                         json_path: pathlib.Path,
                                         inward_only: bool):
    if not repr_path.exists() and not json_path.exists():
        return

    rel_repr_path = f'{TEST_DATA_REL}/{repr_path.name}'
    rel_json_path = f'{TEST_DATA_REL}/{json_path.name}'

    try:
        json_from_file = json_path.read_text()
        json_obj = cirq.read_json(json_text=json_from_file)
    except Exception as ex:  # coverage: ignore
        # coverage: ignore
        raise IOError(
            f'Failed to parse test json data from {rel_json_path}.') from ex

    try:
        repr_obj = _eval_repr_data_file(repr_path)
    except Exception as ex:  # coverage: ignore
        # coverage: ignore
        raise IOError(
            f'Failed to parse test repr data from {rel_repr_path}.') from ex

    assert proper_eq(json_obj, repr_obj), (
        f'The json data from {rel_json_path} did not parse '
        f'into an object equivalent to the repr data from {rel_repr_path}.\n'
        f'\n'
        f'json object: {json_obj!r}\n'
        f'repr object: {repr_obj!r}\n')

    if not inward_only:
        json_from_cirq = cirq.to_json(repr_obj)
        json_from_cirq_obj = json.loads(json_from_cirq)
        json_from_file_obj = json.loads(json_from_file)

        assert json_from_cirq_obj == json_from_file_obj, (
            f'The json produced by cirq no longer agrees with the json in the '
            f'{rel_json_path} test data file.\n'
            f'\n'
            f'You must either fix the cirq code to continue to produce the '
            f'same output, or you must move the old test data to '
            f'{rel_json_path}_inward and create a fresh {rel_json_path} file.\n'
            f'\n'
            f'test data json:\n'
            f'{json_from_file}\n'
            f'\n'
            f'cirq produced json:\n'
            f'{json_from_cirq}\n')


def all_test_data_keys() -> List[str]:
    seen = set()
    for file in TEST_DATA_PATH.iterdir():
        name = file.name
        if name.endswith('.json') or name.endswith('.repr'):
            seen.add(file.name[:-len('.json')])
        elif name.endswith('.json_inward') or name.endswith('.repr_inward'):
            seen.add(file.name[:-len('.json_inward')])
    return sorted(seen)


@pytest.mark.parametrize('key', all_test_data_keys())
def test_json_and_repr_data(key: str):
    assert_repr_and_json_test_data_agree(
        repr_path=TEST_DATA_PATH / f'{key}.repr',
        json_path=TEST_DATA_PATH / f'{key}.json',
        inward_only=False)
    assert_repr_and_json_test_data_agree(
        repr_path=TEST_DATA_PATH / f'{key}.repr_inward',
        json_path=TEST_DATA_PATH / f'{key}.json_inward',
        inward_only=True)


def test_pathlib_paths(tmpdir):
    path = pathlib.Path(tmpdir) / 'op.json'
    cirq.to_json(cirq.X, path)
    assert cirq.read_json(path) == cirq.X


def test_json_serializable_dataclass():

    @cirq.json_serializable_dataclass
    class MyDC:
        q: cirq.LineQubit
        desc: str

    my_dc = MyDC(cirq.LineQubit(4), 'hi mom')

    def custom_resolver(name):
        if name == 'MyDC':
            return MyDC

    assert_json_roundtrip_works(my_dc,
                                text_should_be="\n".join([
                                    '{',
                                    '  "cirq_type": "MyDC",',
                                    '  "q": {',
                                    '    "cirq_type": "LineQubit",',
                                    '    "x": 4',
                                    '  },',
                                    '  "desc": "hi mom"',
                                    '}',
                                ]),
                                resolvers=[custom_resolver] +
                                cirq.DEFAULT_RESOLVERS)


def test_json_serializable_dataclass_parenthesis():

    @cirq.json_serializable_dataclass()
    class MyDC:
        q: cirq.LineQubit
        desc: str

    def custom_resolver(name):
        if name == 'MyDC':
            return MyDC

    my_dc = MyDC(cirq.LineQubit(4), 'hi mom')

    assert_json_roundtrip_works(my_dc,
                                resolvers=[custom_resolver] +
                                cirq.DEFAULT_RESOLVERS)


def test_json_serializable_dataclass_namespace():

    @cirq.json_serializable_dataclass(namespace='cirq.experiments')
    class QuantumVolumeParams:
        width: int
        depth: int
        circuit_i: int

    qvp = QuantumVolumeParams(width=5, depth=5, circuit_i=0)

    def custom_resolver(name):
        if name == 'cirq.experiments.QuantumVolumeParams':
            return QuantumVolumeParams

    assert_json_roundtrip_works(qvp,
                                resolvers=[custom_resolver] +
                                cirq.DEFAULT_RESOLVERS)
