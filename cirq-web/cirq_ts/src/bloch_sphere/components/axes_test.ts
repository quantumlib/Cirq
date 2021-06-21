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

import {assert, expect} from 'chai';
import {generateAxis} from './axes';

describe('Axes methods', () => {
  // Sanity check
  it('returns a type Group from Axes', () => {
    const axes = generateAxis(5);
    expect(axes.type).to.equal('Group')
    expect(axes.constructor.name).to.equal('Axes')
  });

  it('returns 3 Line objects', () => {
    const axes = generateAxis(5);
    const children = axes.children;
    for(const child of children){
      expect(child.type).to.equal('Line');
    }
    expect(children.length).to.equal(3);
  });

  it('has configurable axis colors', () => {
    const axes = generateAxis(5, '#fff', '#fff', '#fff');
    const children = axes.children;
    assert(false, "Not implemented");
  });
});
