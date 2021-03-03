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
import pytest
import pandas as pd

import cirq
from cirq.study.result import _pack_digits


def test_result_init():
    assert cirq.Result(params=cirq.ParamResolver({}), measurements=None).repetitions == 0
    assert cirq.Result(params=cirq.ParamResolver({}), measurements={}).repetitions == 0


def test_repr():
    v = cirq.Result.from_single_parameter_set(
        params=cirq.ParamResolver({'a': 2}), measurements={'xy': np.array([[1, 0], [0, 1]])}
    )
    cirq.testing.assert_equivalent_repr(v)


def test_str():
    result = cirq.Result.from_single_parameter_set(
        params=cirq.ParamResolver({}),
        measurements={
            'ab': np.array([[0, 1], [0, 1], [0, 1], [1, 0], [0, 1]]),
            'c': np.array([[0], [0], [1], [0], [1]]),
        },
    )
    assert str(result) == 'ab=00010, 11101\nc=00101'

    result = cirq.Result.from_single_parameter_set(
        params=cirq.ParamResolver({}),
        measurements={
            'ab': np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
            'c': np.array([[0], [1], [2], [3], [4]]),
        },
    )
    assert str(result) == 'ab=13579, 2 4 6 8 10\nc=01234'


def test_df():
    result = cirq.Result.from_single_parameter_set(
        params=cirq.ParamResolver({}),
        measurements={
            'ab': np.array([[0, 1], [0, 1], [0, 1], [1, 0], [0, 1]], dtype=np.bool),
            'c': np.array([[0], [0], [1], [0], [1]], dtype=np.bool),
        },
    )
    remove_end_measurements = pd.DataFrame(data={'ab': [1, 1, 2], 'c': [0, 1, 0]}, index=[1, 2, 3])

    pd.testing.assert_frame_equal(result.data.iloc[1:-1], remove_end_measurements)

    # Frequency counting.
    df = result.data
    assert len(df[df['ab'] == 1]) == 4
    assert df.c.value_counts().to_dict() == {0: 3, 1: 2}


def test_histogram():
    result = cirq.Result.from_single_parameter_set(
        params=cirq.ParamResolver({}),
        measurements={
            'ab': np.array([[0, 1], [0, 1], [0, 1], [1, 0], [0, 1]], dtype=np.bool),
            'c': np.array([[0], [0], [1], [0], [1]], dtype=np.bool),
        },
    )

    assert result.histogram(key='ab') == collections.Counter({1: 4, 2: 1})
    assert result.histogram(key='ab', fold_func=tuple) == collections.Counter(
        {(False, True): 4, (True, False): 1}
    )
    assert result.histogram(key='ab', fold_func=lambda e: None) == collections.Counter(
        {
            None: 5,
        }
    )
    assert result.histogram(key='c') == collections.Counter({0: 3, 1: 2})


def test_multi_measurement_histogram():
    result = cirq.Result.from_single_parameter_set(
        params=cirq.ParamResolver({}),
        measurements={
            'ab': np.array([[0, 1], [0, 1], [0, 1], [1, 0], [0, 1]], dtype=np.bool),
            'c': np.array([[0], [0], [1], [0], [1]], dtype=np.bool),
        },
    )

    assert result.multi_measurement_histogram(keys=['ab']) == collections.Counter(
        {
            (1,): 4,
            (2,): 1,
        }
    )
    assert result.multi_measurement_histogram(keys=['c']) == collections.Counter(
        {
            (0,): 3,
            (1,): 2,
        }
    )
    assert result.multi_measurement_histogram(keys=['ab', 'c']) == collections.Counter(
        {
            (
                1,
                0,
            ): 2,
            (
                1,
                1,
            ): 2,
            (
                2,
                0,
            ): 1,
        }
    )

    assert result.multi_measurement_histogram(
        keys=[], fold_func=lambda e: None
    ) == collections.Counter(
        {
            None: 5,
        }
    )
    assert result.multi_measurement_histogram(
        keys=['ab'], fold_func=lambda e: None
    ) == collections.Counter(
        {
            None: 5,
        }
    )
    assert result.multi_measurement_histogram(
        keys=['ab', 'c'], fold_func=lambda e: None
    ) == collections.Counter(
        {
            None: 5,
        }
    )

    assert result.multi_measurement_histogram(
        keys=['ab', 'c'], fold_func=lambda e: tuple(tuple(f) for f in e)
    ) == collections.Counter(
        {
            ((False, True), (False,)): 2,
            ((False, True), (True,)): 2,
            ((True, False), (False,)): 1,
        }
    )


def test_trial_result_equality():
    et = cirq.testing.EqualsTester()
    et.add_equality_group(
        cirq.Result.from_single_parameter_set(
            params=cirq.ParamResolver({}), measurements={'a': np.array([[0]] * 5)}
        )
    )
    et.add_equality_group(
        cirq.Result.from_single_parameter_set(
            params=cirq.ParamResolver({}), measurements={'a': np.array([[0]] * 6)}
        )
    )
    et.add_equality_group(
        cirq.Result.from_single_parameter_set(
            params=cirq.ParamResolver({}), measurements={'a': np.array([[1]] * 5)}
        )
    )


def test_trial_result_addition_valid():
    a = cirq.Result.from_single_parameter_set(
        params=cirq.ParamResolver({'ax': 1}),
        measurements={
            'q0': np.array([[0, 1], [1, 0], [0, 1]], dtype=np.bool),
            'q1': np.array([[0], [0], [1]], dtype=np.bool),
        },
    )
    b = cirq.Result.from_single_parameter_set(
        params=cirq.ParamResolver({'ax': 1}),
        measurements={
            'q0': np.array([[0, 1]], dtype=np.bool),
            'q1': np.array([[0]], dtype=np.bool),
        },
    )

    c = a + b
    np.testing.assert_array_equal(c.measurements['q0'], np.array([[0, 1], [1, 0], [0, 1], [0, 1]]))
    np.testing.assert_array_equal(c.measurements['q1'], np.array([[0], [0], [1], [0]]))


def test_trial_result_addition_invalid():
    a = cirq.Result.from_single_parameter_set(
        params=cirq.ParamResolver({'ax': 1}),
        measurements={
            'q0': np.array([[0, 1], [1, 0], [0, 1]], dtype=np.bool),
            'q1': np.array([[0], [0], [1]], dtype=np.bool),
        },
    )
    b = cirq.Result.from_single_parameter_set(
        params=cirq.ParamResolver({'bad': 1}),
        measurements={
            'q0': np.array([[0, 1], [1, 0], [0, 1]], dtype=np.bool),
            'q1': np.array([[0], [0], [1]], dtype=np.bool),
        },
    )
    c = cirq.Result.from_single_parameter_set(
        params=cirq.ParamResolver({'ax': 1}),
        measurements={
            'bad': np.array([[0, 1], [1, 0], [0, 1]], dtype=np.bool),
            'q1': np.array([[0], [0], [1]], dtype=np.bool),
        },
    )
    d = cirq.Result.from_single_parameter_set(
        params=cirq.ParamResolver({'ax': 1}),
        measurements={
            'q0': np.array([[0, 1], [1, 0], [0, 1]], dtype=np.bool),
            'q1': np.array([[0, 1], [0, 1], [1, 1]], dtype=np.bool),
        },
    )

    with pytest.raises(ValueError, match='same parameters'):
        _ = a + b
    with pytest.raises(ValueError, match='same measurement keys'):
        _ = a + c
    with pytest.raises(ValueError):
        _ = a + d
    with pytest.raises(TypeError):
        _ = a + 'junk'


def test_qubit_keys_for_histogram():
    a, b, c = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.measure(a, b),
        cirq.X(c),
        cirq.measure(c),
    )
    results = cirq.Simulator().run(program=circuit, repetitions=100)
    with pytest.raises(KeyError):
        _ = results.histogram(key=a)

    assert results.histogram(key=[a, b]) == collections.Counter({0: 100})
    assert results.histogram(key=c) == collections.Counter({True: 100})
    assert results.histogram(key=[c]) == collections.Counter({1: 100})


def test_text_diagram_jupyter():
    result = cirq.Result.from_single_parameter_set(
        params=cirq.ParamResolver({}),
        measurements={
            'ab': np.array([[0, 1], [0, 1], [0, 1], [1, 0], [0, 1]], dtype=np.bool),
            'c': np.array([[0], [0], [1], [0], [1]], dtype=np.bool),
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
    assert p.text_pretty == 'Result(...)'


def test_json_bit_packing_and_dtype():
    prng = np.random.RandomState(1234)
    bits = prng.randint(2, size=(256, 256)).astype(np.uint8)
    digits = prng.randint(256, size=(256, 256)).astype(np.uint8)

    bits_result = cirq.Result(params=cirq.ParamResolver({}), measurements={'m': bits})
    digits_result = cirq.Result(params=cirq.ParamResolver({}), measurements={'m': digits})

    bits_json = cirq.to_json(bits_result)
    digits_json = cirq.to_json(digits_result)

    loaded_bits_result = cirq.read_json(json_text=bits_json)
    loaded_digits_result = cirq.read_json(json_text=digits_json)

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


def test_deprecation_log():
    with cirq.testing.assert_deprecated('TrialResult was used but is deprecated', deadline="v0.11"):
        cirq.TrialResult(params=cirq.ParamResolver({}), measurements={})


def test_deprecated_json():
    with cirq.testing.assert_deprecated('TrialResult was used but is deprecated', deadline="v0.11"):
        result = cirq.read_json(
            json_text="""{
          "cirq_type": "TrialResult",
          "params": {
            "cirq_type": "ParamResolver",
            "param_dict": []
          },
          "measurements": {
            "0,1": {
              "packed_digits": "fcc0",
              "binary": true,
              "dtype": "uint8",
              "shape": [
                5,
                2
              ]
            }
          }
        }
        """
        )

        assert result == cirq.Result(
            params=cirq.ParamResolver({}),
            measurements={
                '0,1': np.array([[1, 1], [1, 1], [1, 1], [0, 0], [1, 1]], dtype=np.uint8)
            },
        )
