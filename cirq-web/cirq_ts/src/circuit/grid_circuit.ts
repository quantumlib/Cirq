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
import {Group} from 'three';
import {GridQubit} from './components/grid_qubit';
import {Symbol3D, SymbolInformation, Coord} from './components/types';

/**
 * Class that gathers serialized circuit information
 * from a Python cirq.Circuit and reconstructs it to be
 * displayed using three.js
 */
export class GridCircuit extends Group {
  readonly moments: number;
  private circuit: Map<string, GridQubit>;
  private padding_factor: number;
  /**
   * Class constructor
   * @param moments The number of moments of the circuit. This
   * determines the length of all the qubit lines in the diagram.
   * @param qubits A list of GridCoord objects representing the locations of the
   * qubits in the circuit.
   */
  constructor(moments: number, qubits: Coord[], padding_factor = 1) {
    super();
    this.moments = moments;
    this.padding_factor = padding_factor;

    this.circuit = new Map();

    for (const coord of qubits) {
      this.addQubit(coord.row, coord.col);
    }
  }

  /**
   * Adds symbols to the circuit map and renders them in the
   * 3D scene.
   * @param list A list of SymbolInformation objects that give instructions
   * on how to render the operations in 3D.
   */
  addSymbolsFromList(list: SymbolInformation[]) {
    for (const symbol of list) {
      this.addSymbol(symbol);
    }
  }

  private addSymbol(symbol_info: SymbolInformation) {
    //.get gives a reference to an object, so we're good to
    // just modify
    const key = [
      symbol_info.location_info[0].row,
      symbol_info.location_info[0].col,
    ].join(',');
    const symbol = new Symbol3D(symbol_info, this.padding_factor);

    // In production these issues will never come up, since we will always be given
    // a valid grid circuit as input. For development purposes, however,
    // these checks will be useful.
    if (symbol_info.moment < 0 || symbol_info.moment > this.moments) {
      throw new Error(
        `The SymbolInformation object ${symbol_info} has an invalid moment ${symbol_info.moment}`
      );
    }
    const qubit = this.circuit.get(key);
    if (qubit) {
      qubit.addSymbol(symbol);
    } else {
      throw new Error(
        `Cannot add symbol to qubit ${key}. Qubit does not exist in circuit.`
      );
    }
  }

  private addQubit(x: number, y: number) {
    const qubit = new GridQubit(x, y, this.moments, this.padding_factor);
    const key = [x, y].join(',');
    this.circuit.set(key, qubit);
    this.add(qubit);
  }
}
