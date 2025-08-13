// Copyright 2025 The Cirq Developers
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
import {Symbol3D} from './types';

/**
 * Class that represents a GeneralOperation.
 * A GeneralOperation consists of a sprite
 * with location information, and three.js mesh objects
 * representing the gates, if applicable.
 */
export class GeneralOperation extends Group {
  readonly row: number;
  readonly col: number;

  /**
   * Class constructor.
   * @param row The row of the GridQubit
   * @param col The column of the GridQubit
   */
  constructor(row: number, col: number) {
    super();

    this.row = row;
    this.col = col;
  }

  /**
   * Adds a designated symbol to the qubit. Location information,
   * including the moment at which it occurs, is included in the
   * symbol itself.
   * @param symbol A Symbol3D object to add to the qubit.
   */
  addSymbol(symbol: Symbol3D) {
    this.add(symbol);
  }
}
