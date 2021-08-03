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

import {GridCircuit} from './grid_circuit';
import {Symbol3D, SymbolInformation, Coord} from './components/types';
import {expect} from 'chai';
import {GridQubit} from './components/grid_qubit';

describe('GridCircuit', () => {
  describe('with an empty input and no moments', () => {
    const qubits: Coord[] = [];
    const moments = 0;

    const circuit = new GridCircuit(moments, qubits);

    it('does not create any GridQubit objects', () => {
      expect(circuit.children.length).to.equal(0);
    });

    it('correctly sets the number of moments in the circuit', () => {
      expect(circuit.moments).to.equal(0);
    });
  });

  describe('with a 2x2 grid and 5 moments as input', () => {
    const qubits: Coord[] = [
      {row: 0, col: 0},
      {row: 0, col: 1},
      {row: 1, col: 0},
      {row: 1, col: 1},
    ];
    const moments = 5;

    const circuit = new GridCircuit(moments, qubits);

    it('creates 4 GridQubit objects that are children of the Group', () => {
      expect(circuit.children.length).to.equal(4);
      for (const child of circuit.children) {
        expect(child instanceof GridQubit).to.equal(true);
      }
    });

    it('correctly sets the number of moments in the circuit', () => {
      expect(circuit.moments).to.equal(5);
    });
  });

  describe('when calling addSymbolsFromList() with simple cases', () => {
    const qubits: Coord[] = [{row: 0, col: 0}];
    const moments = 2;
    const symbols: SymbolInformation[] = [
      {
        wire_symbols: ['X'],
        location_info: [{row: 0, col: 0}],
        color_info: ['black'],
        moment: 1,
      },
      {
        wire_symbols: ['Y'],
        location_info: [{row: 1, col: 1}],
        color_info: ['green'],
        moment: 1,
      },
      {
        wire_symbols: ['Z'],
        location_info: [{row: 0, col: 0}],
        color_info: ['green'],
        moment: 3,
      },
      {
        wire_symbols: ['Z'],
        location_info: [{row: 0, col: 0}],
        color_info: ['green'],
        moment: -1,
      },
    ];
    const circuit = new GridCircuit(moments, qubits);

    it('correctly adds one symbol at a specific moment', () => {
      circuit.addSymbolsFromList([symbols[0]]);

      // Since we only have one qubit, we know
      // it has to be the first element in the list
      const qubit = circuit.children[0];
      const symbol = qubit.children.find(
        child => child.constructor.name === 'Symbol3D'
      ) as Symbol3D;
      expect(symbol.moment).to.equal(1);
    });

    it('throws an error if given a valid symbol at the wrong row/col pair', () => {
      expect(() => circuit.addSymbolsFromList([symbols[1]])).to.throw(
        'Cannot add symbol to qubit 1,1. Qubit does not exist in circuit.'
      );
    });

    it('throws an error if given a valid symbol at the wrong moments', () => {
      expect(() => circuit.addSymbolsFromList([symbols[2]])).to.throw(
        `The SymbolInformation object ${symbols[2]} has an invalid moment 3`
      );

      expect(() => circuit.addSymbolsFromList([symbols[3]])).to.throw(
        `The SymbolInformation object ${symbols[3]} has an invalid moment -1`
      );
    });
  });
});
