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
import contextlib

import datetime
import io
import json
import os
import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest
import sympy

import cirq
from cirq._compat import proper_eq
from cirq.protocols import json_serialization
from cirq.testing import assert_json_roundtrip_works
from cirq.testing.json import ModuleJsonTestSpec, spec_for

REPO_ROOT = pathlib.Path(__file__).parent.parent.parent.parent
TESTED_MODULES = ['cirq_google', 'cirq.protocols', 'non_existent_should_be_fine']


def _get_testspecs_for_modules():
    modules = []
    for m in TESTED_MODULES:
        try:
            modules.append(spec_for(m))
        except ModuleNotFoundError:
            # for optional modules it is okay to skip
            pass
    return modules


MODULE_TEST_SPECS = _get_testspecs_for_modules()


def test_line_qubit_roundtrip():
    q1 = cirq.LineQubit(12)
    assert_json_roundtrip_works(
        q1,
        text_should_be="""{
  "cirq_type": "LineQubit",
  "x": 12
}""",
    )


def test_gridqubit_roundtrip():
    q = cirq.GridQubit(15, 18)
    assert_json_roundtrip_works(
        q,
        text_should_be="""{
  "cirq_type": "GridQubit",
  "row": 15,
  "col": 18
}""",
    )


def test_op_roundtrip():
    q = cirq.LineQubit(5)
    op1 = cirq.rx(0.123).on(q)
    assert_json_roundtrip_works(
        op1,
        text_should_be="""{
  "cirq_type": "GateOperation",
  "gate": {
    "cirq_type": "Rx",
    "rads": 0.123
  },
  "qubits": [
    {
      "cirq_type": "LineQubit",
      "x": 5
    }
  ]
}""",
    )


def test_op_roundtrip_filename(tmpdir):
    filename = f'{tmpdir}/op.json'
    q = cirq.LineQubit(5)
    op1 = cirq.rx(0.123).on(q)
    cirq.to_json(op1, filename)
    assert os.path.exists(filename)
    op2 = cirq.read_json(filename)
    assert op1 == op2

    gzip_filename = f'{tmpdir}/op.gz'
    cirq.to_json_gzip(op1, gzip_filename)
    assert os.path.exists(gzip_filename)
    op3 = cirq.read_json_gzip(gzip_filename)
    assert op1 == op3


def test_op_roundtrip_file_obj(tmpdir):
    filename = f'{tmpdir}/op.json'
    q = cirq.LineQubit(5)
    op1 = cirq.rx(0.123).on(q)
    with open(filename, 'w+') as file:
        cirq.to_json(op1, file)
        assert os.path.exists(filename)
        file.seek(0)
        op2 = cirq.read_json(file)
        assert op1 == op2

    gzip_filename = f'{tmpdir}/op.gz'
    with open(gzip_filename, 'w+b') as gzip_file:
        cirq.to_json_gzip(op1, gzip_file)
        assert os.path.exists(gzip_filename)
        gzip_file.seek(0)
        op3 = cirq.read_json_gzip(gzip_file)
        assert op1 == op3


def test_fail_to_resolve():
    buffer = io.StringIO()
    buffer.write(
        """
    {
      "cirq_type": "MyCustomClass",
      "data": [1, 2, 3]
    }
    """
    )
    buffer.seek(0)

    with pytest.raises(ValueError) as e:
        cirq.read_json(buffer)
    assert e.match("Could not resolve type 'MyCustomClass' during deserialization")


QUBITS = cirq.LineQubit.range(5)
Q0, Q1, Q2, Q3, Q4 = QUBITS

# TODO: Include cirq.rx in the Circuit test case file.
# Github issue: https://github.com/quantumlib/Cirq/issues/2014
# Note that even the following doesn't work because theta gets
# multiplied by 1/pi:
#   cirq.Circuit(cirq.rx(sympy.Symbol('theta')).on(Q0)),

### MODULE CONSISTENCY tests


@pytest.mark.parametrize('mod_spec', MODULE_TEST_SPECS)
def test_shouldnt_be_serialized_no_superfluous(mod_spec: ModuleJsonTestSpec):
    # everything in the list should be ignored for a reason
    names = set(mod_spec.get_all_names())
    missing_names = set(mod_spec.should_not_be_serialized).difference(names)
    assert len(missing_names) == 0, (
        f"Defined as \"should't be serialized\", "
        f"but missing from {mod_spec}: \n"
        f"{missing_names}"
    )


@pytest.mark.parametrize('mod_spec', MODULE_TEST_SPECS)
def test_not_yet_serializable_no_superfluous(mod_spec: ModuleJsonTestSpec):
    # everything in the list should be ignored for a reason
    names = set(mod_spec.get_all_names())
    missing_names = set(mod_spec.not_yet_serializable).difference(names)
    assert len(missing_names) == 0, (
        f"Defined as Not yet serializable, " f"but missing from {mod_spec}: \n" f"{missing_names}"
    )


@pytest.mark.parametrize('mod_spec', MODULE_TEST_SPECS)
def test_mutually_exclusive_blacklist(mod_spec: ModuleJsonTestSpec):
    common = set(mod_spec.should_not_be_serialized) & set(mod_spec.not_yet_serializable)
    assert len(common) == 0, (
        f"Defined in both {mod_spec.name} 'Not yet serializable' "
        f" and 'Should not be serialized' lists: {common}"
    )


@pytest.mark.parametrize('mod_spec', MODULE_TEST_SPECS)
def test_resolver_cache_vs_should_not_serialize(mod_spec: ModuleJsonTestSpec):
    resolver_cache_types = set([n for (n, _) in mod_spec.get_resolver_cache_types()])
    common = set(mod_spec.should_not_be_serialized) & resolver_cache_types

    assert len(common) == 0, (
        f"Defined in both {mod_spec.name} Resolver "
        f"Cache and should not be serialized:"
        f"{common}"
    )


@pytest.mark.parametrize('mod_spec', MODULE_TEST_SPECS)
def test_resolver_cache_vs_not_yet_serializable(mod_spec: ModuleJsonTestSpec):
    resolver_cache_types = set([n for (n, _) in mod_spec.get_resolver_cache_types()])
    common = set(mod_spec.not_yet_serializable) & resolver_cache_types

    assert len(common) == 0, (
        f"Issue with the JSON config of {mod_spec.name}.\n"
        f"Types are listed in both"
        f" {mod_spec.name}.json_resolver_cache.py and in the 'not_yet_serializable' list in"
        f" {mod_spec.test_data_path}/spec.py: "
        f"\n {common}"
    )


def test_builtins():
    assert_json_roundtrip_works(True)
    assert_json_roundtrip_works(1)
    assert_json_roundtrip_works(1 + 2j)
    assert_json_roundtrip_works(
        {
            'test': [123, 5.5],
            'key2': 'asdf',
            '3': None,
            '0.0': [],
        }
    )


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
        pd.DataFrame(data=[[1, 2, 3], [4, 5, 6]], columns=['x', 'y', 'z'], index=[2, 5])
    )
    assert_json_roundtrip_works(pd.Index([1, 2, 3], name='test'))
    assert_json_roundtrip_works(
        pd.MultiIndex.from_tuples([(1, 2), (3, 4), (5, 6)], names=['alice', 'bob'])
    )

    assert_json_roundtrip_works(
        pd.DataFrame(
            index=pd.Index([1, 2, 3], name='test'),
            data=[[11, 21.0], [12, 22.0], [13, 23.0]],
            columns=['a', 'b'],
        )
    )
    assert_json_roundtrip_works(
        pd.DataFrame(
            index=pd.MultiIndex.from_tuples([(1, 2), (2, 3), (3, 4)], names=['x', 'y']),
            data=[[11, 21.0], [12, 22.0], [13, 23.0]],
            columns=pd.Index(['a', 'b'], name='c'),
        )
    )


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
    assert_json_roundtrip_works(t ** s)

    # Linear combinations.
    assert_json_roundtrip_works(t * 2)
    assert_json_roundtrip_works(4 * t + 3 * s + 2)

    assert_json_roundtrip_works(sympy.pi)
    assert_json_roundtrip_works(sympy.E)
    assert_json_roundtrip_works(sympy.EulerGamma)


class SBKImpl(cirq.SerializableByKey):
    """A test implementation of SerializableByKey."""

    def __init__(
        self,
        name: str,
        data_list: Optional[List] = None,
        data_tuple: Optional[Tuple] = None,
        data_dict: Optional[Dict] = None,
    ):
        self.name = name
        self.data_list = data_list or []
        self.data_tuple = data_tuple or ()
        self.data_dict = data_dict or {}

    def __eq__(self, other):
        if not isinstance(other, SBKImpl):
            return False
        return (
            self.name == other.name
            and self.data_list == other.data_list
            and self.data_tuple == other.data_tuple
            and self.data_dict == other.data_dict
        )

    def _json_dict_(self):
        return {
            "cirq_type": "SBKImpl",
            "name": self.name,
            "data_list": self.data_list,
            "data_tuple": self.data_tuple,
            "data_dict": self.data_dict,
        }

    @classmethod
    def _from_json_dict_(cls, name, data_list, data_tuple, data_dict, **kwargs):
        return cls(name, data_list, tuple(data_tuple), data_dict)


def test_context_serialization():
    def custom_resolver(name):
        if name == 'SBKImpl':
            return SBKImpl

    test_resolvers = [custom_resolver] + cirq.DEFAULT_RESOLVERS

    sbki_empty = SBKImpl('sbki_empty')
    assert_json_roundtrip_works(sbki_empty, resolvers=test_resolvers)

    sbki_list = SBKImpl('sbki_list', data_list=[sbki_empty, sbki_empty])
    assert_json_roundtrip_works(sbki_list, resolvers=test_resolvers)

    sbki_tuple = SBKImpl('sbki_tuple', data_tuple=(sbki_list, sbki_list))
    assert_json_roundtrip_works(sbki_tuple, resolvers=test_resolvers)

    sbki_dict = SBKImpl('sbki_dict', data_dict={'a': sbki_tuple, 'b': sbki_tuple})
    assert_json_roundtrip_works(sbki_dict, resolvers=test_resolvers)

    sbki_json = str(cirq.to_json(sbki_dict))
    # There should be exactly one context item for each previous SBKImpl.
    assert sbki_json.count('"cirq_type": "_SerializedContext"') == 4
    # There should be exactly two key items for each of sbki_(empty|list|tuple),
    # plus one for the top-level sbki_dict.
    assert sbki_json.count('"cirq_type": "_SerializedKey"') == 7
    # The final object should be a _SerializedKey for sbki_dict.
    final_obj_idx = sbki_json.rfind('{')
    final_obj = sbki_json[final_obj_idx : sbki_json.find('}', final_obj_idx) + 1]
    assert (
        final_obj
        == """{
      "cirq_type": "_SerializedKey",
      "key": 4
    }"""
    )

    list_sbki = [sbki_dict]
    assert_json_roundtrip_works(list_sbki, resolvers=test_resolvers)

    dict_sbki = {'a': sbki_dict}
    assert_json_roundtrip_works(dict_sbki, resolvers=test_resolvers)

    assert sbki_list != json_serialization._SerializedKey(sbki_list)

    # Serialization keys have unique suffixes.
    sbki_other_list = SBKImpl('sbki_list', data_list=[sbki_list])
    assert_json_roundtrip_works(sbki_other_list, resolvers=test_resolvers)


def test_internal_serializer_types():
    sbki = SBKImpl('test_key')
    key = 1
    test_key = json_serialization._SerializedKey(key)
    test_context = json_serialization._SerializedContext(sbki, 1)
    test_serialization = json_serialization._ContextualSerialization(sbki)

    key_json = test_key._json_dict_()
    with pytest.raises(TypeError, match='_from_json_dict_'):
        _ = json_serialization._SerializedKey._from_json_dict_(**key_json)

    context_json = test_context._json_dict_()
    with pytest.raises(TypeError, match='_from_json_dict_'):
        _ = json_serialization._SerializedContext._from_json_dict_(**context_json)

    serialization_json = test_serialization._json_dict_()
    with pytest.raises(TypeError, match='_from_json_dict_'):
        _ = json_serialization._ContextualSerialization._from_json_dict_(**serialization_json)


@pytest.mark.parametrize(
    'mod_spec,cirq_obj_name,cls',
    [
        (mod_spec, o, n)
        for mod_spec in MODULE_TEST_SPECS
        for (o, n) in mod_spec.find_classes_that_should_serialize()
    ],
)
def test_json_test_data_coverage(mod_spec: ModuleJsonTestSpec, cirq_obj_name: str, cls):
    if cirq_obj_name == "SerializableByKey":
        pytest.skip(
            "SerializableByKey does not follow common serialization rules. "
            "It is tested separately in test_context_serialization."
        )

    if cirq_obj_name in mod_spec.not_yet_serializable:
        return pytest.xfail(reason="Not serializable (yet)")

    test_data_path = mod_spec.test_data_path
    rel_path = test_data_path.relative_to(REPO_ROOT)
    mod_path = mod_spec.name.replace(".", "/")
    rel_resolver_cache_path = f"{mod_path}/json_resolver_cache.py"
    json_path = test_data_path / f'{cirq_obj_name}.json'
    json_path2 = test_data_path / f'{cirq_obj_name}.json_inward'
    deprecation_deadline = mod_spec.deprecated.get(cirq_obj_name)

    if not json_path.exists() and not json_path2.exists():
        # coverage: ignore
        pytest.fail(
            f"Hello intrepid developer. There is a new public or "
            f"serializable object named '{cirq_obj_name}' in the module '{mod_spec.name}' "
            f"that does not have associated test data.\n"
            f"\n"
            f"You must create the file\n"
            f"    {rel_path}/{cirq_obj_name}.json\n"
            f"and the file\n"
            f"    {rel_path}/{cirq_obj_name}.repr\n"
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
            f"`_class_resolver_dictionary` method in "
            f"the {rel_resolver_cache_path} source file. "
            f"You may also need to add a _json_dict_ method to "
            f"{cirq_obj_name}. In some cases you will also need to add a "
            f"_from_json_dict_ class method to the {cirq_obj_name} class."
            f"\n"
            f"For more information on JSON serialization, please read the "
            f"docstring for cirq.protocols.SupportsJSON. If this object or "
            f"class is not appropriate for serialization, add its name to "
            f"the `should_not_be_serialized` list in the TestSpec defined in the "
            f"{rel_path}/spec.py source file."
        )

    repr_file = test_data_path / f'{cirq_obj_name}.repr'
    if repr_file.exists() and cls is not None:
        objs = _eval_repr_data_file(repr_file, deprecation_deadline=deprecation_deadline)
        if not isinstance(objs, list):
            objs = [objs]

        for obj in objs:
            assert type(obj) == cls, (
                f"Value in {test_data_path}/{cirq_obj_name}.repr must be of "
                f"exact type {cls}, or a list of instances of that type. But "
                f"the value (or one of the list entries) had type "
                f"{type(obj)}.\n"
                f"\n"
                f"If using a value of the wrong type is intended, move the "
                f"value to {test_data_path}/{cirq_obj_name}.repr_inward\n"
                f"\n"
                f"Value with wrong type:\n{obj!r}."
            )


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


def test_to_from_json_gzip():
    a, b = cirq.LineQubit.range(2)
    test_circuit = cirq.Circuit(cirq.H(a), cirq.CX(a, b))
    gzip_data = cirq.to_json_gzip(test_circuit)
    unzip_circuit = cirq.read_json_gzip(gzip_raw=gzip_data)
    assert test_circuit == unzip_circuit

    with pytest.raises(ValueError):
        _ = cirq.read_json_gzip(io.StringIO(), gzip_raw=gzip_data)
    with pytest.raises(ValueError):
        _ = cirq.read_json_gzip()


def _eval_repr_data_file(path: pathlib.Path, deprecation_deadline: Optional[str]):
    ctx_manager = (
        cirq.testing.assert_deprecated(deadline=deprecation_deadline, count=None)
        if deprecation_deadline
        else contextlib.suppress()
    )
    with ctx_manager:
        imports = {
            'cirq': cirq,
            'datetime': datetime,
            'pd': pd,
            'sympy': sympy,
            'np': np,
            'datetime': datetime,
        }
        try:
            import cirq_google

            imports['cirq_google'] = cirq_google
        except ImportError:
            pass
        obj = eval(
            path.read_text(),
            imports,
            {},
        )
    return obj


def assert_repr_and_json_test_data_agree(
    mod_spec: ModuleJsonTestSpec,
    repr_path: pathlib.Path,
    json_path: pathlib.Path,
    inward_only: bool,
    deprecation_deadline: Optional[str],
):
    if not repr_path.exists() and not json_path.exists():
        return

    rel_repr_path = f'{repr_path.relative_to(REPO_ROOT)}'
    rel_json_path = f'{json_path.relative_to(REPO_ROOT)}'

    try:
        json_from_file = json_path.read_text()
        ctx_manager = (
            cirq.testing.assert_deprecated(deadline=deprecation_deadline, count=None)
            if deprecation_deadline
            else contextlib.suppress()
        )
        with ctx_manager:
            json_obj = cirq.read_json(json_text=json_from_file)
    except ValueError as ex:  # coverage: ignore
        # coverage: ignore
        if "Could not resolve type" in str(ex):
            mod_path = mod_spec.name.replace(".", "/")
            rel_resolver_cache_path = f"{mod_path}/json_resolver_cache.py"
            # coverage: ignore
            pytest.fail(
                f"{rel_json_path} can't be parsed to JSON.\n"
                f"Maybe an entry is missing from the "
                f" `_class_resolver_dictionary` method in {rel_resolver_cache_path}?"
            )
        else:
            raise ValueError(f"deprecation: {deprecation_deadline} - got error: {ex}")
    except AssertionError as ex:  # coverage: ignore
        # coverage: ignore
        raise ex
    except Exception as ex:  # coverage: ignore
        # coverage: ignore
        raise IOError(f'Failed to parse test json data from {rel_json_path}.') from ex

    try:
        repr_obj = _eval_repr_data_file(repr_path, deprecation_deadline)
    except Exception as ex:  # coverage: ignore
        # coverage: ignore
        raise IOError(f'Failed to parse test repr data from {rel_repr_path}.') from ex

    assert proper_eq(json_obj, repr_obj), (
        f'The json data from {rel_json_path} did not parse '
        f'into an object equivalent to the repr data from {rel_repr_path}.\n'
        f'\n'
        f'json object: {json_obj!r}\n'
        f'repr object: {repr_obj!r}\n'
    )

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
            f'{json_from_cirq}\n'
        )


@pytest.mark.parametrize(
    'mod_spec, abs_path',
    [(m, abs_path) for m in MODULE_TEST_SPECS for abs_path in m.all_test_data_keys()],
)
def test_json_and_repr_data(mod_spec: ModuleJsonTestSpec, abs_path: str):
    assert_repr_and_json_test_data_agree(
        mod_spec=mod_spec,
        repr_path=pathlib.Path(f'{abs_path}.repr'),
        json_path=pathlib.Path(f'{abs_path}.json'),
        inward_only=False,
        deprecation_deadline=mod_spec.deprecated.get(os.path.basename(abs_path)),
    )
    assert_repr_and_json_test_data_agree(
        mod_spec=mod_spec,
        repr_path=pathlib.Path(f'{abs_path}.repr_inward'),
        json_path=pathlib.Path(f'{abs_path}.json_inward'),
        inward_only=True,
        deprecation_deadline=mod_spec.deprecated.get(os.path.basename(abs_path)),
    )


def test_pathlib_paths(tmpdir):
    path = pathlib.Path(tmpdir) / 'op.json'
    cirq.to_json(cirq.X, path)
    assert cirq.read_json(path) == cirq.X

    gzip_path = pathlib.Path(tmpdir) / 'op.gz'
    cirq.to_json_gzip(cirq.X, gzip_path)
    assert cirq.read_json_gzip(gzip_path) == cirq.X


def test_json_serializable_dataclass():
    @cirq.json_serializable_dataclass
    class MyDC:
        q: cirq.LineQubit
        desc: str

    my_dc = MyDC(cirq.LineQubit(4), 'hi mom')

    def custom_resolver(name):
        if name == 'MyDC':
            return MyDC

    assert_json_roundtrip_works(
        my_dc,
        text_should_be="\n".join(
            [
                '{',
                '  "cirq_type": "MyDC",',
                '  "q": {',
                '    "cirq_type": "LineQubit",',
                '    "x": 4',
                '  },',
                '  "desc": "hi mom"',
                '}',
            ]
        ),
        resolvers=[custom_resolver] + cirq.DEFAULT_RESOLVERS,
    )


def test_json_serializable_dataclass_parenthesis():
    @cirq.json_serializable_dataclass()
    class MyDC:
        q: cirq.LineQubit
        desc: str

    def custom_resolver(name):
        if name == 'MyDC':
            return MyDC

    my_dc = MyDC(cirq.LineQubit(4), 'hi mom')

    assert_json_roundtrip_works(my_dc, resolvers=[custom_resolver] + cirq.DEFAULT_RESOLVERS)


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

    assert_json_roundtrip_works(qvp, resolvers=[custom_resolver] + cirq.DEFAULT_RESOLVERS)
