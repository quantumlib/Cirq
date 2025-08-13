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
import {GeneralOperation} from './components/general_operation';
import {Symbol3D, SymbolInformation} from './components/types';
import {expect} from 'chai';

describe('GridCircuit', () => {
  describe('with an empty input and no moments', () => {
    const symbols: SymbolInformation[] = [];
    const moments = 0;

    const circuit = new GridCircuit(moments, symbols);

    it('does not create any GridQubit objects', () => {
      expect(circuit.children.length).to.equal(0);
    });
  });

  describe('when building circuits from simple cases', () => {
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
        location_info: [{row: 1, col: 1}],
        color_info: ['green'],
        moment: 2,
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
      {
        wire_symbols: ['X'],
        location_info: [{row: 0, col: 1}],
        color_info: ['black'],
        moment: 1,
      },
    ];

    it('correctly adds one symbol at a specific moment', () => {
      const circuit = new GridCircuit(moments, [symbols[0]]);

      // Since we only have one qubit, we know
      // it has to be the first element in the list
      const qubit = circuit.children[0];
      const symbol = qubit.children.find(
        child => child.constructor.name === 'Symbol3D',
      ) as Symbol3D;
      expect(symbol.moment).to.equal(1);
    });

    it('correctly handles gates overlapping on the same qubit', () => {
      const circuit = new GridCircuit(moments, [symbols[1], symbols[2]]);

      const qubit = circuit.children[0];
      const symbol = qubit.children.filter(
        child => child.constructor.name === 'Symbol3D',
      ) as Symbol3D[];
      expect(symbol[0].moment).to.equal(1);
      expect(symbol[1].moment).to.equal(2);
    });

    it('throws an error if given a valid symbol at the wrong moments', () => {
      expect(() => new GridCircuit(moments, [symbols[3]])).to.throw(
        `The SymbolInformation object ${symbols[3]} has an invalid moment 3`,
      );

      expect(() => new GridCircuit(moments, [symbols[4]])).to.throw(
        `The SymbolInformation object ${symbols[4]} has an invalid moment -1`,
      );
    });

    it('adds the correct number of GridQubit objects with overlapping rows', () => {
      const circuit = new GridCircuit(moments, [symbols[0], symbols[1], symbols[5]]);

      const qubits = circuit.children;
      expect(qubits.length).to.equal(3);
    });
  });

  describe('when a circuit contains symbols with no location info', () => {
    const moments = 2;
    const symbols: SymbolInformation[] = [
      {
        wire_symbols: ['X'],
        location_info: [{row: 0, col: 0}],
        color_info: ['black'],
        moment: 1,
      },
      {
        wire_symbols: ['H'],
        location_info: [],
        color_info: ['red'],
        moment: 0,
      },
      {
        wire_symbols: ['Y'],
        location_info: [],
        color_info: ['blue'],
        moment: 0,
      },
      {
        wire_symbols: ['Z'],
        location_info: [],
        color_info: ['green'],
        moment: 1,
      },
    ];

    const circuit = new GridCircuit(moments, symbols);
    it('creates the correct number of GridQubit children', () => {
      const qubits = circuit.children.filter(child => child.constructor.name === 'GridQubit');
      expect(qubits.length).to.equal(1);
    });

    it('creates GeneralOperation objects for each foreign symbol', () => {
      const generalOps = circuit.children.filter(
        child => child.constructor.name === 'GeneralOperation',
      );
      expect(generalOps.length).to.equal(3);
    });

    it('places foreign symbols on a new row below the qubit', () => {
      const generalOps = circuit.children.filter(
        child => child.constructor.name === 'GeneralOperation',
      ) as GeneralOperation[];

      // There is 1 qubit at row 0. Padding factor defaults to 1.
      // The new row should be max_row + padding + 1 = 0 + 1 + 1 = 2
      const expectedRow = 2;
      for (const op of generalOps) {
        expect(op.row).to.equal(expectedRow);
      }
    });

    it('places symbols from the same moment in adjacent columns', () => {
      const generalOps = circuit.children.filter(
        child => child.constructor.name === 'GeneralOperation',
      ) as GeneralOperation[];

      const opsInMoment0 = generalOps.filter(op => {
        const symbol = op.children[0] as Symbol3D;
        return symbol.moment === 0;
      });
      // Sort by column to have a deterministic test
      opsInMoment0.sort((a, b) => a.col - b.col);

      expect(opsInMoment0.length).to.equal(2);
      expect(opsInMoment0[0].col).to.equal(0);
      expect(opsInMoment0[1].col).to.equal(1);
    });

    it('resets column placement for each new moment', () => {
      const generalOps = circuit.children.filter(
        child => child.constructor.name === 'GeneralOperation',
      ) as GeneralOperation[];

      const opsInMoment1 = generalOps.filter(op => {
        const symbol = op.children[0] as Symbol3D;
        return symbol.moment === 1;
      });

      expect(opsInMoment1.length).to.equal(1);
      expect(opsInMoment1[0].col).to.equal(0);
    });
  });

  describe('when a circuit contains only symbols with no location info', () => {
    const moments = 1;
    const symbols: SymbolInformation[] = [
      {
        wire_symbols: ['H'],
        location_info: [],
        color_info: ['red'],
        moment: 0,
      },
    ];

    const circuit = new GridCircuit(moments, symbols);

    it('creates only GeneralOperation children', () => {
      const generalOps = circuit.children.filter(
        child => child.constructor.name === 'GeneralOperation',
      );
      const qubits = circuit.children.filter(child => child.constructor.name === 'GridQubit');
      expect(generalOps.length).to.equal(1);
      expect(qubits.length).to.equal(0);
    });

    it('places the foreign symbol at row 0', () => {
      const generalOp = circuit.children[0] as GeneralOperation;
      expect(generalOp.row).to.equal(0);
    });
  });
});
