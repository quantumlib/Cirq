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
import {GridQubit} from './grid_qubit';

describe('GridQubit', () => {
  const DEFAULT_ROW = 0;
  const DEFAULT_COL = 0;
  const DEFAULT_MOMENTS = 5;
  const gridQubit = new GridQubit(DEFAULT_ROW, DEFAULT_COL, DEFAULT_MOMENTS);
  const children = gridQubit.children;

  it('is a three.js line object', () => {
    const line = children.find(child => child.type === 'Line');
    expect(line).to.not.equal(undefined);
  });

  it('has a three.js sprite label', () => {
    const line = children.find(child => child.type === 'Sprite');
    expect(line).to.not.equal(undefined);
  });
});
