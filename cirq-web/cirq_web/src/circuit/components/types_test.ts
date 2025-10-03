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

import {expect} from 'vitest';
import {Symbol3D, SymbolInformation} from './types';
import {BoxGate3DSymbol, Control3DSymbol, X3DSymbol} from './meshes';

describe('Symbol3D', () => {
  describe('with valid one qubit SymbolInformation objects', () => {
    const symbols: SymbolInformation[] = [
      {
        wire_symbols: ['X'],
        location_info: [{row: 0, col: 0}],
        color_info: ['black'],
        moment: 1,
      },
      {
        wire_symbols: ['@'],
        location_info: [{row: 1, col: 2}],
        color_info: ['black'],
        moment: 3,
      },
      {
        wire_symbols: ['Y'],
        location_info: [{row: 3, col: 2}],
        color_info: ['purple'],
        moment: 7,
      },
    ];

    it('handles the X gate special case', () => {
      const symbolObj = new Symbol3D(symbols[0]);
      const xSymbol = symbolObj.children[0];
      expect(xSymbol instanceof X3DSymbol).toBe(true);
    });

    it('handles the control symbol special case', () => {
      const symbolObj = new Symbol3D(symbols[1]);
      const ctrlSymbol = symbolObj.children[0];
      expect(ctrlSymbol instanceof Control3DSymbol).toBe(true);
    });

    it('handles an arbitrary symbol with a box', () => {
      const symbolObj = new Symbol3D(symbols[2]);
      const boxSymbol = symbolObj.children[0];
      expect(boxSymbol instanceof BoxGate3DSymbol).toBe(true);
    });

    it('builds every symbol in the correct location on the scene', () => {
      const expectedRows = [0, 1, 3];
      const expectedCols = [0, 2, 2];
      const expectedMoments = [1, 3, 7];
      symbols.forEach((value, index) => {
        const symbol = new Symbol3D(value).children[0];
        expect(symbol.position.x).toBe(expectedRows[index]);
        expect(symbol.position.y).toBe(expectedMoments[index]);
        expect(symbol.position.z).toBe(expectedCols[index]);
      });
    });
  });

  describe('with valid multi-qubit SymbolInformation objects', () => {
    const symbols: SymbolInformation[] = [
      {
        wire_symbols: ['@', 'X'],
        location_info: [
          {row: 0, col: 0},
          {row: 0, col: 1},
        ],
        color_info: ['black', 'black'],
        moment: 1,
      },
      {
        wire_symbols: ['×', '×'],
        location_info: [
          {row: 0, col: 0},
          {row: 0, col: 1},
        ],
        color_info: ['black', 'black'],
        moment: 3,
      },
      {
        wire_symbols: ['iSWAP', 'iSWAP'],
        location_info: [
          {row: 3, col: 2},
          {row: 3, col: 3},
        ],
        color_info: ['#d3d3d3', '#d3d3d3'],
        moment: 7,
      },
      {
        wire_symbols: ['@', '@', 'X'],
        location_info: [
          {row: 0, col: 0},
          {row: 0, col: 1},
          {row: 0, col: 2},
        ],
        color_info: ['black', 'black', 'black'],
        moment: 1,
      },
    ];

    it('handles a control symbol and an X gate (CNOT)', () => {
      const symbolObj = new Symbol3D(symbols[0]);

      const ctrlSymbol = symbolObj.children.find(
        child => child.constructor.name === 'Control3DSymbol',
      );
      expect(ctrlSymbol).not.toBe(undefined);
      expect(ctrlSymbol?.position.x).toBe(0);
      expect(ctrlSymbol?.position.z).toBe(0);
      expect(ctrlSymbol?.position.y).toBe(1);

      const xSymbol = symbolObj.children.find(child => child.constructor.name === 'X3DSymbol');
      expect(xSymbol).not.toBe(undefined);
      expect(xSymbol?.position.x).toBe(0);
      expect(xSymbol?.position.z).toBe(1);
      expect(xSymbol?.position.y).toBe(1);

      const connectionLine = symbolObj.children.find(
        child => child.constructor.name === 'ConnectionLine',
      );

      expect(connectionLine).not.toBe(undefined);
    });

    it('handles a two Swap3DSymbol objects correctly (SWAP)', () => {
      const symbolObj = new Symbol3D(symbols[1]);

      const boxSymbols = symbolObj.children.filter(
        child => child.constructor.name === 'Swap3DSymbol',
      );
      expect(boxSymbols.length).toBe(2);

      const expectedRows = [0, 0];
      const expectedCols = [0, 1];
      boxSymbols.forEach((value, index) => {
        expect(value.position.x).toBe(expectedRows[index]);
        expect(value.position.z).toBe(expectedCols[index]);
        expect(value.position.y).toBe(3);
      });

      const connectionLine = symbolObj.children.find(
        child => child.constructor.name === 'ConnectionLine',
      );
      expect(connectionLine).not.toBe(undefined);
    });

    it('handles a two BoxGate3DSymbol objects correctly (iSWAP)', () => {
      const symbolObj = new Symbol3D(symbols[2]);

      const boxSymbols = symbolObj.children.filter(
        child => child.constructor.name === 'BoxGate3DSymbol',
      );
      expect(boxSymbols.length).toBe(2);

      const expectedRows = [3, 3];
      const expectedCols = [2, 3];
      boxSymbols.forEach((value, index) => {
        expect(value.position.x).to.equal(expectedRows[index]);
        expect(value.position.z).to.equal(expectedCols[index]);
        expect(value.position.y).to.equal(7);
      });

      const connectionLine = symbolObj.children.find(
        child => child.constructor.name === 'ConnectionLine',
      );
      expect(connectionLine).to.not.equal(undefined);
    });

    it('handles two control symbols and an X gate correctly (Toffoli)', () => {
      const symbolObj = new Symbol3D(symbols[3]);

      const boxSymbols = symbolObj.children.filter(
        child => child.constructor.name === 'Control3DSymbol',
      );
      expect(boxSymbols.length).to.equal(2);

      const xSymbol = symbolObj.children.find(child => child.constructor.name === 'X3DSymbol');
      expect(xSymbol).not.toBe(undefined);

      const expectedRows = [0, 0, 0];
      const expectedCols = [0, 1, 2];
      boxSymbols.forEach((value, index) => {
        expect(value.position.x).toBe(expectedRows[index]);
        expect(value.position.z).toBe(expectedCols[index]);
        expect(value.position.y).toBe(1);
      });

      const connectionLines = symbolObj.children.filter(
        child => child.constructor.name === 'ConnectionLine',
      );
      expect(connectionLines.length).toBe(2);
    });
  });
});
