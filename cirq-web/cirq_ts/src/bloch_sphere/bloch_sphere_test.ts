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
import {BlochSphere} from './bloch_sphere';

describe('The BlochSphere class', () => {
  // Sanity check
  it('has a default radius of 5', () => {
    const bloch_sphere = new BlochSphere();
    const radius = bloch_sphere.getRadius();
    expect(radius).to.equal(5);
  });

  it('has a configurable radius', () => {
    const bloch_sphere = new BlochSphere(3);
    const radius = bloch_sphere.getRadius();
    expect(radius).to.equal(3);
  });

  it('returns a Group object on getBlochSphere()', () => {
    const bloch_sphere = new BlochSphere(5);
    const sphere = bloch_sphere.getBlochSphere();
    expect(sphere.type).to.equal('Group');
  });
});
