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
  // A nested map where the outer map correlates
  // rows to <column, GridQubit> pairs.
  private qubit_map: Map<number, Map<number, GridQubit>>;
  private padding_factor: number;
  /**
   * Class constructor
   * @param initial_num_moments The number of moments of the circuit. This
   * determines the length of all the qubit lines in the diagram.
   * @param symbols A list of SymbolInformation objects with info about operations
   * in the circuit.
   * @param padding_factor A number scaling the distance between meshes.
   */
  constructor(
    initial_num_moments: number,
    symbols: SymbolInformation[],
    padding_factor = 1
  ) {
    super();
    this.padding_factor = padding_factor;
    this.qubit_map = new Map();

    for (const symbol of symbols) {
      // Being accurate is more important than speed here, so
      // traversing through each object isn't a big deal.
      // However, this logic can be changed if needed to avoid redundancy.
      for (const coordinate of symbol.location_info) {
        // If the key already exists in the map, don't repeat.
        if (this.hasQubit(coordinate.row, coordinate.col)) {
          continue;
        }
        this.addQubit(coordinate.row, coordinate.col, initial_num_moments);
      }
      this.addSymbol(symbol, initial_num_moments);
    }
  }

  private addSymbol(
    symbolInfo: SymbolInformation,
    initial_num_moments: number
  ) {
    const symbol = new Symbol3D(symbolInfo, this.padding_factor);

    // In production these issues will never come up, since we will always be given
    // a valid grid circuit as input. For development purposes, however,
    // these checks will be useful.
    if (symbolInfo.moment < 0 || symbolInfo.moment > initial_num_moments) {
      throw new Error(
        `The SymbolInformation object ${symbolInfo} has an invalid moment ${symbolInfo.moment}`
      );
    }

    const qubit = this.getQubit(
      symbolInfo.location_info[0].row,
      symbolInfo.location_info[0].col
    )!;
    qubit.addSymbol(symbol);
  }

  private addQubit(row: number, col: number, initial_num_moments: number) {
    const qubit = new GridQubit(
      row,
      col,
      initial_num_moments,
      this.padding_factor
    );
    this.setQubit(row, col, qubit);
    this.add(qubit);
  }

  private getQubit(row: number, col: number) {
    // We will never hit an undefined value, since we're
    // adding qubits based off the provided information to the user.
    const innerMap = this.qubit_map.get(row);
    return innerMap!.get(col);
  }

  private setQubit(row: number, col: number, qubit: GridQubit) {
    const innerMap = this.qubit_map.get(row);
    if (innerMap) {
      innerMap.set(col, qubit);
    } else {
      this.qubit_map.set(row, new Map().set(col, qubit));
    }
  }

  private hasQubit(row: number, col: number) {
    const innerMap = this.qubit_map.get(row);
    if (innerMap) {
      return innerMap.has(col);
    }
    return false;
  }
}
