# Copyright 2019 The Cirq Developers
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

from cirq import quirk_url_to_circuit


def test_non_physical_operations():
    with pytest.raises(NotImplementedError, match="unphysical operation"):
        _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":['
                                 '["__error__"]]}')
    with pytest.raises(NotImplementedError, match="unphysical operation"):
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
