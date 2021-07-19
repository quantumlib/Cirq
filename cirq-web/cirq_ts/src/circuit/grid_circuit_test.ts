// Copyright 2021 The Cirq Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import {expect} from 'chai';
import {GridCircuit} from './grid_circuit';
import {GridCoord} from './components/types';

describe('GridCircuit', () => {
    describe('with a 2x2 grid and 5 moments as input', () => {
        const twoByTwoGrid : GridCoord[] = [
            {'row': 0, 'col': 0},
            {'row': 0, 'col': 1},
            {'row': 1, 'col': 0},
            {'row': 1, 'col': 1},
        ]
        const circuit = new GridCircuit(5, twoByTwoGrid);

        it('generates a correct mapping of qubit coords to GridQubit objects', () => {
            // This will be improved once the forEach method is implemented on the
            // CircuitMap object.
            expect(circuit.circuit.get([0, 0])!.constructor.name).to.equal('GridQubit');
            expect(circuit.circuit.get([0, 1])!.constructor.name).to.equal('GridQubit');
            expect(circuit.circuit.get([1, 0])!.constructor.name).to.equal('GridQubit');
            expect(circuit.circuit.get([1, 1])!.constructor.name).to.equal('GridQubit');
            console.log(circuit);
        });

        it('correctly sets the moments for the circuit', () => {
            expect(circuit.moments).to.equal(5);
        });

        it('has the same # of three.js children as the number of qubits in the circuit', () => {
            expect(circuit.children.length).to.equal(circuit.circuit.size);
        });
    });
});