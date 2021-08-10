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
  BoxGate3DSymbol,
  ConnectionLine,
  Control3DSymbol,
  Swap3DSymbol,
  X3DSymbol,
} from './meshes';

export interface SymbolInformation {
  readonly wire_symbols: string[];
  readonly location_info: Coord[];
  readonly color_info: string[];
  readonly moment: number;
}

export interface Coord {
  readonly row: number;
  readonly col: number;
}

/**
 * Builds a 3D symbol representing a Cirq `Operation`.
 */
export class Symbol3D extends Group {
  readonly moment: number;
  private padding_factor: number;

  /**
   * Class constructor.
   * @param symbol_info A typed object with information instructing
   * the class on how to build the mesh.
   * @param padding_factor A number scaling the distance between meshes.
   */
  constructor(symbol_info: SymbolInformation, padding_factor = 1) {
    super();
    this.padding_factor = padding_factor;
    this.moment = symbol_info.moment;
    this.buildAndAddMeshesToGroup(symbol_info);

    // If this is a multi-qubit operation, we automatically
    // add lines connecting the symbols.
    this.addConnectionLines(symbol_info);
  }

  private buildAndAddMeshesToGroup(symbol_info: SymbolInformation) {
    const locationInfo = symbol_info.location_info;

    symbol_info.wire_symbols.forEach((symbol, index) => {
      let mesh;
      switch (symbol) {
        case 'X':
          mesh = new X3DSymbol(symbol_info.color_info[index]);
          break;
        case '@':
          mesh = new Control3DSymbol();
          break;
        case '×':
          mesh = new Swap3DSymbol();
          break;
        default:
          mesh = new BoxGate3DSymbol(symbol, symbol_info.color_info[index]);
      }

      mesh.position.set(
        locationInfo[index].row * this.padding_factor,
        symbol_info.moment * this.padding_factor,
        locationInfo[index].col * this.padding_factor
      );
      this.add(mesh);
    });
  }

  private addConnectionLines(symbol_info: SymbolInformation) {
    const locationInfo = symbol_info.location_info;

    if (locationInfo.length > 1) {
      let i = 0;
      while (i < locationInfo.length - 1) {
        const coords = [
          new Vector3(
            locationInfo[i].row * this.padding_factor,
            symbol_info.moment * this.padding_factor,
            locationInfo[i].col * this.padding_factor
          ),
          new Vector3(
            locationInfo[i + 1].row * this.padding_factor,
            symbol_info.moment * this.padding_factor,
            locationInfo[i + 1].col * this.padding_factor
          ),
        ];
        this.add(new ConnectionLine(coords[0], coords[1]));
        i++;
      }
    }
  }
}
