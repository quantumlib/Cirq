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
 * Builds a 3D symbol from the given information.
 */
export class Symbol3D extends Group {
  private SCALING_FACTOR = 1;
  readonly moment: number;

  constructor(symbol_info: SymbolInformation) {
    super();
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
        default:
          mesh = new BoxGate3DSymbol(symbol, symbol_info.color_info[index]);
      }

      mesh.position.set(
        locationInfo[index].row * this.SCALING_FACTOR,
        symbol_info.moment,
        locationInfo[index].col * this.SCALING_FACTOR
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
            locationInfo[i].row * this.SCALING_FACTOR,
            symbol_info.moment,
            locationInfo[i].col * this.SCALING_FACTOR
          ),
          new Vector3(
            locationInfo[i + 1].row * this.SCALING_FACTOR,
            symbol_info.moment,
            locationInfo[i + 1].col * this.SCALING_FACTOR
          ),
        ];
        this.add(new ConnectionLine(coords[0], coords[1]));
        i++;
      }
    }
  }
}
