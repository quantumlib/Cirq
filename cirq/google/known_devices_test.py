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


def test_json_dict():
    assert cirq.google.Foxtail._json_dict_() == {
        'cirq_type': '_NamedConstantXmonDevice',
        'constant': 'cirq.google.Foxtail',
        'measurement_duration': cirq.Duration(nanos=1000),
        'exp_w_duration': cirq.Duration(nanos=20),
        'exp_11_duration': cirq.Duration(nanos=50),
        'qubits': sorted(cirq.google.Foxtail.qubits)
    }

    assert cirq.google.Bristlecone._json_dict_() == {
        'cirq_type': '_NamedConstantXmonDevice',
        'constant': 'cirq.google.Bristlecone',
        'measurement_duration': cirq.Duration(nanos=1000),
        'exp_w_duration': cirq.Duration(nanos=20),
        'exp_11_duration': cirq.Duration(nanos=50),
        'qubits': sorted(cirq.google.Bristlecone.qubits)
    }
