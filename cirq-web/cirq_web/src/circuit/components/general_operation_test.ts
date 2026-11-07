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

import {expect} from 'chai';
import {Symbol3D, SymbolInformation} from './types';
import {GeneralOperation} from './general_operation';

describe('GeneralOperation', () => {
  it('initializes and adds a symbol correctly', () => {
    const row = 1;
    const col = 2;
    const genOp = new GeneralOperation(row, col);

    const symbolInfo: SymbolInformation = {
      wire_symbols: ['X'],
      color_info: ['black'],
      moment: 1,
      location_info: [{row, col}],
    };
    const symbol = new Symbol3D(symbolInfo, 1);
    genOp.addSymbol(symbol);

    const symbol3D = genOp.children[0] as Symbol3D;
    expect(symbol3D.children[0].constructor.name).to.equal('X3DSymbol');
  });
});
