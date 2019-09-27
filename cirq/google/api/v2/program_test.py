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

import cirq
import cirq.google.api.v2 as v2


def test_qubit_to_proto_id():
    assert v2.qubit_to_proto_id(cirq.GridQubit(1, 2)) == '1_2'
    assert v2.qubit_to_proto_id(cirq.GridQubit(10, 2)) == '10_2'


def test_qubit_from_proto_id():
    assert v2.qubit_from_proto_id('1_2') == cirq.GridQubit(1, 2)
    assert v2.qubit_from_proto_id('10_2') == cirq.GridQubit(10, 2)
