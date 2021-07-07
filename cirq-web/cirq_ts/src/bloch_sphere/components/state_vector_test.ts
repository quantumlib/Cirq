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
import {ArrowHelper, Vector3} from 'three';
import {StateVector} from './state_vector';

describe('StateVector', () => {
  describe('by default', () => {
    const vector = new StateVector(1, 1, 2, 5);
    it('start at the correct point given arbitrary vector coordinates', () => {
      const nestedVector = vector.children[0] as ArrowHelper;
      expect(nestedVector.position).to.eql(new Vector3(0, 0, 0));
    });
  });
});
