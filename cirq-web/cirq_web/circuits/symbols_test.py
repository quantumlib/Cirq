# Copyright 2021 The Cirq Developers
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
import cirq_web

def test_basic_Operation3DSymbol():
    wire_symbols = ['X']
    location_info = [{'row': 0, 'col': 0}]
    color_info = ['black']
    moment = 1

    symbol = cirq_web.circuits.symbols.Operation3DSymbol(
        wire_symbols,
        location_info,
        color_info,
        moment
    )

    actual = symbol.to_typescript()
    expected = {
        'wire_symbols': ['X'],
        'location_info': [{'row': 0, 'col': 0}],
        'color_info': ['black'],
        'moment': 1,
    }
    assert actual == expected