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

# This is more of a placeholder for now, we can add
# official color schemes in follow-ups.
SymbolColors = {
    '?': '#d3d3d3',
    '@': 'black',
    'H': 'yellow',
    'I': 'orange',
    'X': 'black',
    'Y': 'pink',
    'Z': 'cyan',
    'S': '#90EE90',
    'T': '#CBC3E3',
}


class Operation3DSymbol:
    def __init__(self, wire_symbols, location_info, color_info, moment):
        """Gathers symbol information from an operation and builds an
        object to represent it in 3D.

        Args:
            wire_symbols: a list of symbols taken from circuit_diagram_info()
            that will be used to represent the operation in the 3D circuit.

            location_info: A list of coordinates for each wire_symbol. The
            index of the coordinate tuple in the location_info list must
            correspond with the index of the symbol in the wire_symbols list.

            color_info: a list representing the desired color of the symbol(s).
            These will also correspond to index of the symbol in the
            wire_symbols list.

            moment: the moment where the symbol should be.
        """
        self.wire_symbols = wire_symbols
        self.location_info = location_info
        self.color_info = color_info
        self.moment = moment

    def to_typescript(self):
        return {
            'wire_symbols': list(self.wire_symbols),
            'location_info': self.location_info,
            'color_info': self.color_info,
            'moment': self.moment,
        }
