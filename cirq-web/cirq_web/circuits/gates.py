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

from abc import ABC, abstractmethod


class Gate(ABC):
    @abstractmethod
    def to_typescript(self):
        raise NotImplementedError()


class SingleQubitGate(Gate):
    def __init__(self, color):
        self.color = color

    def __call__(self, text, moment, loc):
        self.text = text
        self.moment = moment
        self.row = loc[0]
        self.col = loc[1]

    def to_typescript(self):
        return {
            'type': 'SingleQubitGate',
            'text': self.text,
            'color': self.color,
            'row': self.row,
            'col': self.col,
            'moment': self.moment,
        }


class TwoQubitGate(Gate):
    def __init__(self, target_gate):
        self.target_gate = target_gate

    def __call__(self, moment, loc):
        self.moment = moment
        self.row = loc[0]
        self.col = loc[1]

    def to_typescript(self):
        return {
            'type': 'TwoQubitGate',
            'row': self.row,
            'col': self.col,
            'targetGate': self.target_gate.to_typescript(),
            'moment': self.moment,
        }


class UnknownSingleQubitGate(SingleQubitGate):
    def __init__(self):
        super()
        self.color = 'gray'


class UnknownTwoQubitGate(TwoQubitGate):
    def __init__(self):
        super()
        self.target_gate = UnknownSingleQubitGate()


Gate3DSymbols = {
    # Single Qubit Gates
    'H': SingleQubitGate('yellow'),
    'I': SingleQubitGate('orange'),
    'X': SingleQubitGate('black'),
    'Y': SingleQubitGate('pink'),
    'Z': SingleQubitGate('cyan'),
    # Two Qubit Gates
    'CNOT': TwoQubitGate(SingleQubitGate('black')),
    'CZ': TwoQubitGate(SingleQubitGate('cyan')),
    'CY': TwoQubitGate(SingleQubitGate('pink')),
}
