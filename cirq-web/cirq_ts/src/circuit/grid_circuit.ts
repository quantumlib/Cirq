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
import {Symbol3D, SymbolInformation} from './components/types';

/**
 * Class that gathers serialized circuit information
 * from a Python cirq.Circuit and reconstructs it to be
 * displayed using three.js
 */
export class GridCircuit extends Group {
  // The keys of this map are serialized Coord arrays [row, col],
  // representing the row and column where each GridQubit object
  // is located.
  private circuit: Map<string, GridQubit>;
  private padding_factor: number;
  /**
   * Class constructor
   * @param initial_num_moments The number of moments of the circuit. This
   * determines the length of all the qubit lines in the diagram.
   * @param qubits A list of GridCoord objects representing the locations of the
   * qubits in the circuit.
   * @param padding_factor A number scaling the distance between meshes.
   */
  constructor(
    initial_num_moments: number,
    symbol_list: SymbolInformation[],
    padding_factor = 1
  ) {
    super();
    this.padding_factor = padding_factor;
    this.circuit = new Map();

    for (const symbol of symbol_list) {
      // Being accurate is more important than speed here, so
      // traversing through each object isn't a big deal.
      // However, this logic can be changed if needed to avoid redundancy.
      for (const coordinate of symbol.location_info) {
        // If the key already exists in the map, don't repeat.
        if (this.circuit.has([coordinate.row, coordinate.col].join(','))) {
          continue;
        }
        this.addQubit(coordinate.row, coordinate.col, initial_num_moments);
      }
      this.addSymbol(symbol, initial_num_moments);
    }
  }

  private addSymbol(
    symbol_info: SymbolInformation,
    initial_num_moments: number
  ) {
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
    if (symbol_info.moment < 0 || symbol_info.moment > initial_num_moments) {
      throw new Error(
        `The SymbolInformation object ${symbol_info} has an invalid moment ${symbol_info.moment}`
      );
    }

    const qubit = this.circuit.get(key)!;
    qubit.addSymbol(symbol);
  }

  private addQubit(x: number, y: number, initial_num_moments: number) {
    const qubit = new GridQubit(x, y, initial_num_moments, this.padding_factor);
    const key = [x, y].join(',');
    this.circuit.set(key, qubit);
    this.add(qubit);
  }
}
