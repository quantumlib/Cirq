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
import { BoxGate3DSymbol, ConnectionLine, Control3DSymbol, X3DSymbol } from "./meshes";

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
  constructor(symbol_info: SymbolInformation) {
    super();

    let mesh;
    symbol_info.wire_symbols.forEach((symbol, index) => {
      if (symbol == 'X') {
        mesh = new X3DSymbol(symbol_info.color_info[index]);
      } else if (symbol == '@') {
        mesh = new Control3DSymbol()
      } else {
        mesh = new BoxGate3DSymbol(symbol, symbol_info.color_info[index]); 
      }

      mesh.position.set(
        symbol_info.location_info[index].row,
        symbol_info.moment,
        symbol_info.location_info[index].col,
        )
      this.add(mesh);
    });

    // Add lines by default if you have multiple symbols
    if (symbol_info.location_info.length > 1){
      let i = 0;
      while (i < symbol_info.location_info.length - 1) {
        const coords = [
          new Vector3(
            symbol_info.location_info[i].row,
            symbol_info.moment,
            symbol_info.location_info[i].col,
            ),
          new Vector3(
            symbol_info.location_info[i+1].row,
            symbol_info.moment,
            symbol_info.location_info[i+1].col,
          )
        ];
        this.add(new ConnectionLine(coords[0], coords[1]));
        i++;
      }


    }
  }

}