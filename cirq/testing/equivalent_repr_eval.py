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

from typing import Any


def assert_equivalent_repr(
        value: Any,
        *,
        setup_code: str = 'import cirq') -> None:
    """Checks that eval(repr(v)) == v.

    Args:
        value: A value whose repr should be evaluatable python
            code that produces an equivalent value.
        setup_code: Code that must be executed before the repr can be evaluated.
            Ideally this should just be a series of 'import' lines.
    """

    global_vals = {}
    local_vals = {}
    exec(setup_code, global_vals, local_vals)

    try:
        eval_repr_value = eval(repr(value), global_vals, local_vals)
    except Exception as ex:
        raise AssertionError(
            'eval(repr(value)) raised an exception.\n'
            '\n'
            'setup_code={!r}\n'
            'value={!r}\n'
            'error={}'.format(ex, setup_code, value))

    assert eval_repr_value == value, (
        "The repr of a value of type {} didn't evaluate to something equal "
        "to the value.\n"
        'eval(repr(value)) != value\n'
        '\n'
        'value: {}\n'
        'repr(value): {!r}\n'
        'eval(repr(value)): {}\n'
        'repr(eval(repr(value))): {!r}\n'
        '\n'
        'type(value): {}\n'
        'type(eval(repr(value))): {!r}\n'
        '\n'
        'setup_code:\n{}\n'
    ).format(type(value),
             value,
             repr(value),
             eval_repr_value,
             repr(eval_repr_value),
             type(value),
             type(eval_repr_value),
             '    ' + setup_code.replace('\n', '\n    '))
