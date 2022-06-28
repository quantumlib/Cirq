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

import collections

import numpy as np
import pandas as pd
import pytest

import cirq
import cirq.testing
from cirq.study.result import _pack_digits


def test_result_init():
    assert cirq.ResultDict(params=cirq.ParamResolver({}), measurements=None).repetitions == 0
    assert cirq.ResultDict(params=cirq.ParamResolver({}), measurements={}).repetitions == 0


def test_default_repetitions():
    class MyResult(cirq.Result):
        def __init__(self, records):
            self._records = records

        @property
        def params(self):
            raise NotImplementedError()

        @property
        def measurements(self):
            raise NotImplementedError()

        @property
        def records(self):
            return self._records

        @property
        def data(self):
            raise NotImplementedError()

    assert MyResult({}).repetitions == 0
    assert MyResult({'a': np.zeros((5, 2, 3))}).repetitions == 5


def test_repr():
    v = cirq.ResultDict(
        params=cirq.ParamResolver({'a': 2}), measurements={'xy': np.array([[1, 0], [0, 1]])}
    )
    cirq.testing.assert_equivalent_repr(v)

    v = cirq.ResultDict(
        params=cirq.ParamResolver({'a': 2}),
        records={'xy': np.array([[[0, 0], [0, 1]], [[1, 0], [1, 1]]])},
    )
    cirq.testing.assert_equivalent_repr(v)


def test_construct_from_measurements():
    r = cirq.ResultDict(
        params=None,
        measurements={'a': np.array([[0, 0], [1, 1]]), 'b': np.array([[0, 0, 0], [1, 1, 1]])},
    )
    assert np.all(r.measurements['a'] == np.array([[0, 0], [1, 1]]))
    assert np.all(r.measurements['b'] == np.array([[0, 0, 0], [1, 1, 1]]))
    assert np.all(r.records['a'] == np.array([[[0, 0]], [[1, 1]]]))
    assert np.all(r.records['b'] == np.array([[[0, 0, 0]], [[1, 1, 1]]]))


def test_construct_from_repeated_measurements():
    r = cirq.ResultDict(
        params=None,
        records={
            'a': np.array([[[0, 0], [0, 1]], [[1, 0], [1, 1]]]),
            'b': np.array([[[0, 0, 0]], [[1, 1, 1]]]),
        },
    )
    with pytest.raises(ValueError):
        _ = r.measurements
    assert np.all(r.records['a'] == np.array([[[0, 0], [0, 1]], [[1, 0], [1, 1]]]))
    assert np.all(r.records['b'] == np.array([[[0, 0, 0]], [[1, 1, 1]]]))
    assert r.repetitions == 2

    r2 = cirq.ResultDict(
        params=None,
        records={'a': np.array([[[0, 0]], [[1, 1]]]), 'b': np.array([[[0, 0, 0]], [[1, 1, 1]]])},
    )
    assert np.all(r2.measurements['a'] == np.array([[0, 0], [1, 1]]))
    assert np.all(r2.measurements['b'] == np.array([[0, 0, 0], [1, 1, 1]]))
    assert np.all(r2.records['a'] == np.array([[[0, 0]], [[1, 1]]]))
    assert np.all(r2.records['b'] == np.array([[[0, 0, 0]], [[1, 1, 1]]]))
    assert r2.repetitions == 2


def test_empty_measurements():
    assert cirq.ResultDict(params=None).repetitions == 0
    assert cirq.ResultDict(params=None, measurements={}).repetitions == 0
    assert cirq.ResultDict(params=None, records={}).repetitions == 0


def test_str():
    result = cirq.ResultDict(
        params=cirq.ParamResolver({}),
        measurements={
            'ab': np.array([[0, 1], [0, 1], [0, 1], [1, 0], [0, 1]]),
            'c': np.array([[0], [0], [1], [0], [1]]),
        },
    )
    assert str(result) == 'ab=00010, 11101\nc=00101'

    result = cirq.ResultDict(
        params=cirq.ParamResolver({}),
        measurements={
            'ab': np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
            'c': np.array([[0], [1], [2], [3], [4]]),
        },
    )
    assert str(result) == 'ab=13579, 2 4 6 8 10\nc=01234'


def test_df():
    result = cirq.ResultDict(
        params=cirq.ParamResolver({}),
        measurements={
            'ab': np.array([[0, 1], [0, 1], [0, 1], [1, 0], [0, 1]], dtype=bool),
            'c': np.array([[0], [0], [1], [0], [1]], dtype=bool),
        },
    )
    remove_end_measurements = pd.DataFrame(
        data={'ab': [1, 1, 2], 'c': [0, 1, 0]}, index=[1, 2, 3], dtype=np.int64
    )

    pd.testing.assert_frame_equal(result.data.iloc[1:-1], remove_end_measurements)

    # Frequency counting.
    df = result.data
    assert len(df[df['ab'] == 1]) == 4
    assert df.c.value_counts().to_dict() == {0: 3, 1: 2}


def test_df_large():
    result = cirq.ResultDict(
        params=cirq.ParamResolver({}),
        measurements={
            'a': np.array([[0 for _ in range(76)]] * 10_000, dtype=bool),
            'd': np.array([[1 for _ in range(76)]] * 10_000, dtype=bool),
        },
    )

    assert np.all(result.data['a'] == 0)
    assert np.all(result.data['d'] == 0xFFF_FFFFFFFF_FFFFFFFF)
    assert result.data['a'].dtype == object
    assert result.data['d'].dtype == object


def test_histogram():
    result = cirq.ResultDict(
        params=cirq.ParamResolver({}),
        measurements={
            'ab': np.array([[0, 1], [0, 1], [0, 1], [1, 0], [0, 1]], dtype=bool),
            'c': np.array([[0], [0], [1], [0], [1]], dtype=bool),
        },
    )

    assert result.histogram(key='ab') == collections.Counter({1: 4, 2: 1})
    assert result.histogram(key='ab', fold_func=tuple) == collections.Counter(
        {(False, True): 4, (True, False): 1}
    )
    assert result.histogram(key='ab', fold_func=lambda e: None) == collections.Counter({None: 5})
    assert result.histogram(key='c') == collections.Counter({0: 3, 1: 2})


def test_multi_measurement_histogram():
    result = cirq.ResultDict(
        params=cirq.ParamResolver({}),
        measurements={
            'ab': np.array([[0, 1], [0, 1], [0, 1], [1, 0], [0, 1]], dtype=bool),
            'c': np.array([[0], [0], [1], [0], [1]], dtype=bool),
        },
    )

    assert result.multi_measurement_histogram(keys=['ab']) == collections.Counter(
        {(1,): 4, (2,): 1}
    )
    assert result.multi_measurement_histogram(keys=['c']) == collections.Counter({(0,): 3, (1,): 2})
    assert result.multi_measurement_histogram(keys=['ab', 'c']) == collections.Counter(
        {(1, 0): 2, (1, 1): 2, (2, 0): 1}
    )

    assert result.multi_measurement_histogram(
        keys=[], fold_func=lambda e: None
    ) == collections.Counter({None: 5})
    assert result.multi_measurement_histogram(
        keys=['ab'], fold_func=lambda e: None
    ) == collections.Counter({None: 5})
    assert result.multi_measurement_histogram(
        keys=['ab', 'c'], fold_func=lambda e: None
    ) == collections.Counter({None: 5})

    assert result.multi_measurement_histogram(
        keys=['ab', 'c'], fold_func=lambda e: tuple(tuple(f) for f in e)
    ) == collections.Counter(
        {((False, True), (False,)): 2, ((False, True), (True,)): 2, ((True, False), (False,)): 1}
    )


def test_result_equality():
    et = cirq.testing.EqualsTester()
    et.add_equality_group(
        cirq.ResultDict(params=cirq.ParamResolver({}), measurements={'a': np.array([[0]] * 5)}),
        cirq.ResultDict(params=cirq.ParamResolver({}), records={'a': np.array([[[0]]] * 5)}),
    )
    et.add_equality_group(
        cirq.ResultDict(params=cirq.ParamResolver({}), measurements={'a': np.array([[0]] * 6)})
    )
    et.add_equality_group(
        cirq.ResultDict(params=cirq.ParamResolver({}), measurements={'a': np.array([[1]] * 5)})
    )
    et.add_equality_group(
        cirq.ResultDict(params=cirq.ParamResolver({}), records={'a': np.array([[[0], [1]]] * 5)})
    )


def test_result_addition_valid():
    a = cirq.ResultDict(
        params=cirq.ParamResolver({'ax': 1}),
        measurements={
            'q0': np.array([[0, 1], [1, 0], [0, 1]], dtype=bool),
            'q1': np.array([[0], [0], [1]], dtype=bool),
        },
    )
    b = cirq.ResultDict(
        params=cirq.ParamResolver({'ax': 1}),
        measurements={'q0': np.array([[0, 1]], dtype=bool), 'q1': np.array([[0]], dtype=bool)},
    )

    c = a + b
    np.testing.assert_array_equal(c.measurements['q0'], np.array([[0, 1], [1, 0], [0, 1], [0, 1]]))
    np.testing.assert_array_equal(c.measurements['q1'], np.array([[0], [0], [1], [0]]))

    # Add results with repeated measurements.
    a = cirq.ResultDict(
        params=cirq.ParamResolver({'ax': 1}),
        records={
            'q0': np.array([[[0, 1]], [[1, 0]], [[0, 1]]], dtype=bool),
            'q1': np.array([[[0], [0]], [[0], [1]], [[1], [0]]], dtype=bool),
        },
    )
    b = cirq.ResultDict(
        params=cirq.ParamResolver({'ax': 1}),
        records={'q0': np.array([[[0, 1]]], dtype=bool), 'q1': np.array([[[1], [1]]], dtype=bool)},
    )

    c = a + b
    np.testing.assert_array_equal(
        c.records['q0'], np.array([[[0, 1]], [[1, 0]], [[0, 1]], [[0, 1]]])
    )
    np.testing.assert_array_equal(
        c.records['q1'], np.array([[[0], [0]], [[0], [1]], [[1], [0]], [[1], [1]]])
    )


def test_result_addition_invalid():
    a = cirq.ResultDict(
        params=cirq.ParamResolver({'ax': 1}),
        measurements={
            'q0': np.array([[0, 1], [1, 0], [0, 1]], dtype=bool),
            'q1': np.array([[0], [0], [1]], dtype=bool),
        },
    )
    b = cirq.ResultDict(
        params=cirq.ParamResolver({'bad': 1}),
        measurements={
            'q0': np.array([[0, 1], [1, 0], [0, 1]], dtype=bool),
            'q1': np.array([[0], [0], [1]], dtype=bool),
        },
    )
    c = cirq.ResultDict(
        params=cirq.ParamResolver({'ax': 1}),
        measurements={
            'bad': np.array([[0, 1], [1, 0], [0, 1]], dtype=bool),
            'q1': np.array([[0], [0], [1]], dtype=bool),
        },
    )
    d = cirq.ResultDict(
        params=cirq.ParamResolver({'ax': 1}),
        measurements={
            'q0': np.array([[0, 1], [1, 0], [0, 1]], dtype=bool),
            'q1': np.array([[0, 1], [0, 1], [1, 1]], dtype=bool),
        },
    )
    e = cirq.ResultDict(
        params=cirq.ParamResolver({'ax': 1}),
        records={
            'q0': np.array([[0, 1], [1, 0], [0, 1]], dtype=bool),
            'q1': np.array([[[0], [0]], [[0], [1]], [[1], [0]]], dtype=bool),
        },
    )

    with pytest.raises(ValueError, match='different parameters'):
        _ = a + b
    with pytest.raises(ValueError, match='different measurement shapes'):
        _ = a + c
    with pytest.raises(ValueError, match='different measurement shapes'):
        _ = a + d
    with pytest.raises(ValueError, match='different measurement shapes'):
        _ = a + e
    with pytest.raises(TypeError):
        _ = a + 'junk'


def test_qubit_keys_for_histogram():
    a, b, c = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.measure(a, b), cirq.X(c), cirq.measure(c))
    results = cirq.Simulator().run(program=circuit, repetitions=100)
    with pytest.raises(KeyError):
        _ = results.histogram(key=a)

    assert results.histogram(key=[a, b]) == collections.Counter({0: 100})
    assert results.histogram(key=c) == collections.Counter({True: 100})
    assert results.histogram(key=[c]) == collections.Counter({1: 100})


def test_text_diagram_jupyter():
    result = cirq.ResultDict(
        params=cirq.ParamResolver({}),
        measurements={
            'ab': np.array([[0, 1], [0, 1], [0, 1], [1, 0], [0, 1]], dtype=bool),
            'c': np.array([[0], [0], [1], [0], [1]], dtype=bool),
        },
    )

    # Test Jupyter console output from
    class FakePrinter:
        def __init__(self):
            self.text_pretty = ''

        def text(self, to_print):
            self.text_pretty += to_print

    p = FakePrinter()
    result._repr_pretty_(p, False)
    assert p.text_pretty == 'ab=00010, 11101\nc=00101'

    # Test cycle handling
    p = FakePrinter()
    result._repr_pretty_(p, True)
    assert p.text_pretty == 'ResultDict(...)'


@pytest.mark.parametrize('use_records', [False, True])
def test_json_bit_packing_and_dtype(use_records: bool) -> None:
    shape = (256, 3, 256) if use_records else (256, 256)

    prng = np.random.RandomState(1234)
    bits = prng.randint(2, size=shape).astype(np.uint8)
    digits = prng.randint(256, size=shape).astype(np.uint8)

    params = cirq.ParamResolver({})
    if use_records:
        bits_result = cirq.ResultDict(params=params, records={'m': bits})
        digits_result = cirq.ResultDict(params=params, records={'m': digits})
    else:
        bits_result = cirq.ResultDict(params=params, measurements={'m': bits})
        digits_result = cirq.ResultDict(params=params, measurements={'m': digits})

    bits_json = cirq.to_json(bits_result)
    digits_json = cirq.to_json(digits_result)

    loaded_bits_result = cirq.read_json(json_text=bits_json)
    loaded_digits_result = cirq.read_json(json_text=digits_json)

    if use_records:
        assert loaded_bits_result.records['m'].dtype == np.uint8
        assert loaded_digits_result.records['m'].dtype == np.uint8
    else:
        assert loaded_bits_result.measurements['m'].dtype == np.uint8
        assert loaded_digits_result.measurements['m'].dtype == np.uint8
    np.testing.assert_allclose(len(bits_json), len(digits_json) / 8, rtol=0.02)


def test_json_bit_packing_error():
    with pytest.raises(ValueError):
        _pack_digits(np.ones(10), pack_bits='hi mom')


def test_json_bit_packing_force():
    assert _pack_digits(np.ones(10, dtype=int), pack_bits='force') == _pack_digits(
        np.ones(10), pack_bits='auto'
    )

    assert _pack_digits(2 * np.ones(10, dtype=int), pack_bits='force') != _pack_digits(
        2 * np.ones(10, dtype=int), pack_bits='auto'
    )
    # These are the `np.packbits` semantics, namely calling packbits on things
    # that aren't bits first converts elements to bool, which is why these
    # two calls are equivalent.
    assert _pack_digits(2 * np.ones(10, dtype=int), pack_bits='force') == _pack_digits(
        np.ones(10), pack_bits='auto'
    )


def test_json_unpack_compat():
    """Test reading old json with serialized measurements array."""
    old_json = """
        {
            "cirq_type": "ResultDict",
            "params": {
                "cirq_type": "ParamResolver",
                "param_dict": []
            },
            "measurements": {
                "m": {
                    "packed_digits": "d32a",
                    "binary": true,
                    "dtype": "bool",
                    "shape": [
                        3,
                        5
                    ]
                }
            }
        }
    """
    result = cirq.read_json(json_text=old_json)
    assert result == cirq.ResultDict(
        params=cirq.ParamResolver({}),
        measurements={
            'm': np.array(
                [
                    [True, True, False, True, False],
                    [False, True, True, False, False],
                    [True, False, True, False, True],
                ],
                dtype=bool,
            )
        },
    )
