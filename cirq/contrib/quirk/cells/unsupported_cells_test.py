# Copyright 2018 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from cirq.contrib.quirk.cells.testing import assert_url_to_circuit_returns
from cirq.contrib.quirk.url_to_circuit import quirk_url_to_circuit


def test_non_physical_operations():
    with pytest.raises(NotImplementedError, match="Unphysical operation"):
        _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":['
                                 '["__error__"]]}')
    with pytest.raises(NotImplementedError, match="Unphysical operation"):
        _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":['
                                 '["__unstable__UniversalNot"]]}')


def test_not_implemented_gates():
    # This test mostly exists to ensure the gates are tested if added.

    for k in ["X^⌈t⌉", "X^⌈t-¼⌉", "Counting4", "Uncounting4", ">>t3", "<<t3"]:
        with pytest.raises(NotImplementedError, match="discrete parameter"):
            _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={'
                                     '"cols":[["' + k + '"]]}')

    for k in ["add3", "sub3", "c+=ab4", "c-=ab4"]:
        with pytest.raises(NotImplementedError, match="deprecated"):
            _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={'
                                     '"cols":[["' + k + '"]]}')

    for k in ["X", "Y", "Z"]:
        with pytest.raises(NotImplementedError, match="feedback"):
            _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"'
                                     'cols":[["' + k + 'DetectControlReset"]]}')

    for k in ["|0⟩⟨0|", "|1⟩⟨1|", "|+⟩⟨+|", "|-⟩⟨-|", "|X⟩⟨X|", "|/⟩⟨/|", "0"]:
        with pytest.raises(NotImplementedError, match="postselection"):
            _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={'
                                     '"cols":[["' + k + '"]]}')


def test_example_qft_circuit():
    qft_example_diagram = """
0: ───×───────────────H───S───────T───────────Z─────────────────────Z────────────────────────────────Z──────────────────────────────────────────Z────────────────────────────────────────────────────Z──────────────────────────────────────────────────────────────
      │                   │       │           │                     │                                │                                          │                                                    │
1: ───┼───×───────────────@───H───┼───S───────┼─────────T───────────┼──────────Z─────────────────────┼─────────Z────────────────────────────────┼─────────Z──────────────────────────────────────────┼─────────Z────────────────────────────────────────────────────
      │   │                       │   │       │         │           │          │                     │         │                                │         │                                          │         │
2: ───┼───┼───×───────────────────@───@───H───┼─────────┼───S───────┼──────────┼─────────T───────────┼─────────┼──────────Z─────────────────────┼─────────┼─────────Z────────────────────────────────┼─────────┼─────────Z──────────────────────────────────────────
      │   │   │                               │         │   │       │          │         │           │         │          │                     │         │         │                                │         │         │
3: ───┼───┼───┼───×───────────────────────────@^(1/8)───@───@───H───┼──────────┼─────────┼───S───────┼─────────┼──────────┼─────────T───────────┼─────────┼─────────┼──────────Z─────────────────────┼─────────┼─────────┼─────────Z────────────────────────────────
      │   │   │   │                                                 │          │         │   │       │         │          │         │           │         │         │          │                     │         │         │         │
4: ───┼───┼───┼───×─────────────────────────────────────────────────@^(1/16)───@^(1/8)───@───@───H───┼─────────┼──────────┼─────────┼───S───────┼─────────┼─────────┼──────────┼─────────T───────────┼─────────┼─────────┼─────────┼──────────Z─────────────────────
      │   │   │                                                                                      │         │          │         │   │       │         │         │          │         │           │         │         │         │          │
5: ───┼───┼───×──────────────────────────────────────────────────────────────────────────────────────@^0.031───@^(1/16)───@^(1/8)───@───@───H───┼─────────┼─────────┼──────────┼─────────┼───S───────┼─────────┼─────────┼─────────┼──────────┼─────────T───────────
      │   │                                                                                                                                     │         │         │          │         │   │       │         │         │         │          │         │
6: ───┼───×─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────@^0.016───@^0.031───@^(1/16)───@^(1/8)───@───@───H───┼─────────┼─────────┼─────────┼──────────┼─────────┼───S───────
      │                                                                                                                                                                                              │         │         │         │          │         │   │
7: ───×──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────@^0.008───@^0.016───@^0.031───@^(1/16)───@^(1/8)───@───@───H───
    """

    qft_example_json = (
        '{"cols":['
        # '["Counting8"],'
        '["Chance8"],'
        '["…","…","…","…","…","…","…","…"],'
        '["Swap",1,1,1,1,1,1,"Swap"],'
        '[1,"Swap",1,1,1,1,"Swap"],'
        '[1,1,"Swap",1,1,"Swap"],'
        '[1,1,1,"Swap","Swap"],'
        '["H"],'
        '["Z^½","•"],'
        '[1,"H"],'
        '["Z^¼","Z^½","•"],'
        '[1,1,"H"],'
        '["Z^⅛","Z^¼","Z^½","•"],'
        '[1,1,1,"H"],'
        '["Z^⅟₁₆","Z^⅛","Z^¼","Z^½","•"],'
        '[1,1,1,1,"H"],'
        '["Z^⅟₃₂","Z^⅟₁₆","Z^⅛","Z^¼","Z^½","•"],'
        '[1,1,1,1,1,"H"],'
        '["Z^⅟₆₄","Z^⅟₃₂","Z^⅟₁₆","Z^⅛","Z^¼","Z^½","•"],'
        '[1,1,1,1,1,1,"H"],'
        '["Z^⅟₁₂₈","Z^⅟₆₄","Z^⅟₃₂","Z^⅟₁₆","Z^⅛","Z^¼","Z^½","•"],'
        '[1,1,1,1,1,1,1,"H"]]}')
    qft_example_json_uri_escaped = (
        '{%22cols%22:['
        # '[%22Counting8%22],'
        '[%22Chance8%22],'
        '[%22%E2%80%A6%22,%22%E2%80%A6%22,%22%E2%80%A6%22,%22%E2%80%A6%22,'
        '%22%E2%80%A6%22,%22%E2%80%A6%22,%22%E2%80%A6%22,%22%E2%80%A6%22],'
        '[%22Swap%22,1,1,1,1,1,1,%22Swap%22],'
        '[1,%22Swap%22,1,1,1,1,%22Swap%22],'
        '[1,1,%22Swap%22,1,1,%22Swap%22],'
        '[1,1,1,%22Swap%22,%22Swap%22],'
        '[%22H%22],'
        '[%22Z^%C2%BD%22,%22%E2%80%A2%22],'
        '[1,%22H%22],'
        '[%22Z^%C2%BC%22,%22Z^%C2%BD%22,%22%E2%80%A2%22],'
        '[1,1,%22H%22],'
        '[%22Z^%E2%85%9B%22,%22Z^%C2%BC%22,%22Z^%C2%BD%22,%22%E2%80%A2%22],'
        '[1,1,1,%22H%22],'
        '[%22Z^%E2%85%9F%E2%82%81%E2%82%86%22,%22Z^%E2%85%9B%22,%22Z^%C2%BC%22,'
        '%22Z^%C2%BD%22,%22%E2%80%A2%22],'
        '[1,1,1,1,%22H%22],'
        '[%22Z^%E2%85%9F%E2%82%83%E2%82%82%22,'
        '%22Z^%E2%85%9F%E2%82%81%E2%82%86%22,%22Z^%E2%85%9B%22,%22Z^%C2%BC%22,'
        '%22Z^%C2%BD%22,%22%E2%80%A2%22],'
        '[1,1,1,1,1,%22H%22],'
        '[%22Z^%E2%85%9F%E2%82%86%E2%82%84%22,'
        '%22Z^%E2%85%9F%E2%82%83%E2%82%82%22,'
        '%22Z^%E2%85%9F%E2%82%81%E2%82%86%22,%22Z^%E2%85%9B%22,%22Z^%C2%BC%22,'
        '%22Z^%C2%BD%22,%22%E2%80%A2%22],'
        '[1,1,1,1,1,1,%22H%22],'
        '[%22Z^%E2%85%9F%E2%82%81%E2%82%82%E2%82%88%22,'
        '%22Z^%E2%85%9F%E2%82%86%E2%82%84%22,'
        '%22Z^%E2%85%9F%E2%82%83%E2%82%82%22,'
        '%22Z^%E2%85%9F%E2%82%81%E2%82%86%22,%22Z^%E2%85%9B%22,%22Z^%C2%BC%22,'
        '%22Z^%C2%BD%22,%22%E2%80%A2%22],'
        '[1,1,1,1,1,1,1,%22H%22]]}')
    assert_url_to_circuit_returns(qft_example_json, diagram=qft_example_diagram)
    assert_url_to_circuit_returns(qft_example_json_uri_escaped,
                                  diagram=qft_example_diagram)
