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

import {Group, Vector3} from 'three';
import {
  QubitLine,
  QubitLabel,
} from './meshes';
import {
  Symbol3D
} from './types';

/**
 * Class that represents a GridQubit.
 * A GridQubit consists of a three.js line, a sprite
 * with location information, and three.js mesh objects
 * representing the gates, if applicable.
 */
export class GridQubit extends Group {
  readonly row: number;
  readonly col: number;
  readonly moments: number;

  /**
   * Class constructor.
   * @param row The row of the GridQubit
   * @param col The column of the GridQubit
   * @param moments The number of moments of the entire circuit. This
   * determines the length of the three.js line representing the GridQubit
   */
  constructor(row: number, col: number, moments: number) {
    super();

    this.row = row;
    this.col = col;
    this.moments = moments;
    this.add(this.createLine(moments));
    this.add(this.addLocationLabel());
  }

  addSymbol(symbol: Symbol3D) {
    this.add(symbol);
  }


  private createLine(moments: number) : QubitLine {
    const coords = [
      new Vector3(this.row, 0, this.col),
      new Vector3(this.row, this.moments, this.col),
    ];
    return new QubitLine(coords[0], coords[1]);
  }

  private addLocationLabel() : QubitLabel {
    const sprite = new QubitLabel(`(${this.row}, ${this.col})`);
    sprite.position.copy(new Vector3(this.row, -0.6, this.col));
    return sprite;
  }
}
