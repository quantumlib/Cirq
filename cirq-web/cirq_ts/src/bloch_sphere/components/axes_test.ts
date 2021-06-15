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
  it('returns an object', () => {
    const axes = generateAxis(5);
    assert.typeOf(axes, 'object');
  });

  it('returns a mapping of Line objects to axis labels', () => {
    const axes = generateAxis(5);
    expect(axes.x.type).to.equal('Line');
    expect(axes.y.type).to.equal('Line');
    expect(axes.z.type).to.equal('Line');
  });

  it('has configurable axis colors', () => {
    // No way access color attribute in three.js, looking into it
    const axes = generateAxis(5, '#fff', '#fff', '#fff');
    expect(axes.x.type).to.equal('Line');
    expect(axes.y.type).to.equal('Line');
    expect(axes.z.type).to.equal('Line');
  });
});
