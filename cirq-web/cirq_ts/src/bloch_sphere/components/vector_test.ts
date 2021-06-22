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
import {Vector} from './vector';

describe('Vector methods', () => {
  describe('defaults', () => {
    // Example test case
    const vector = new Vector(undefined);
    it('Empty vector initializes to the correct values', () => {
      const nestedVector = vector.children[0] as ArrowHelper;
      expect(nestedVector.position).to.eql(new Vector3(0, 0, 0));
    });
  });

  describe('configurables', () => {
    // Example test case
    const vector = new Vector('{"x": 1,"y": 1, "z": 2, "length": 5}');
    it('Mocked vector initializes to the correct values', () => {
      const nestedVector = vector.children[0] as ArrowHelper;
      expect(nestedVector.position).to.eql(new Vector3(0, 0, 0));
    });
  });
});
