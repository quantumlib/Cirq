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

import {expect} from 'chai';
import {Line} from 'three';
import {GridQubit} from './grid_qubit';
import {QubitLabel} from './meshes';
import {Symbol3D, SymbolInformation} from './types';

describe('GridQubit with 5 default moments', () => {
  const DEFAULT_ROW = 0;
  const DEFAULT_COL = 0;
  const DEFAULT_MOMENTS = 5;
  const gridQubit = new GridQubit(DEFAULT_ROW, DEFAULT_COL, DEFAULT_MOMENTS);
  const children = gridQubit.children;

  it('is a three.js line object with length 5', () => {
    const line = children.find(child => child.type === 'Line') as Line;
    line.computeLineDistances();
    const distancePoints = line.geometry.attributes.lineDistance.array;

    expect(distancePoints[0]).to.equal(0);
    expect(distancePoints[1]).to.equal(DEFAULT_MOMENTS);
  });

  it('has a three.js sprite label', () => {
    const sprite = children.find(
      child => child.type === 'Sprite'
    ) as QubitLabel;
    expect(sprite.text).to.equal(`(${DEFAULT_ROW}, ${DEFAULT_COL})`);
  });

  it('handles adding a basic Symbol3D object', () => {
    const symbolInfo: SymbolInformation = {
      wire_symbols: ['X'],
      location_info: [{row: 0, col: 0}],
      color_info: ['black'],
      moment: 1,
    };

    const symbol = new Symbol3D(symbolInfo);
    gridQubit.addSymbol(symbol);

    const symbol3D = children.find(
      child => child.constructor.name === 'Symbol3D'
    )!;
    expect(symbol3D.children[0].constructor.name).to.equal('X3DSymbol');
  });
});
