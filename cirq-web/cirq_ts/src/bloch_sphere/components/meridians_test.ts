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
import {
  createHorizontalChordMeridians,
  createVerticalMeridians,
  createHorizontalCircleMeridians,
} from './meridians';

describe('Meridians', () => {
  const DEFAULT_RADIUS = 5;
  const DEFAULT_H_MERIDIANS = 7;
  const DEFAULT_V_MERIDIANS = 4;

  describe('defaults', () => {
    it('createHorizontalChordMeridians() returns type Group from Meridians', () => {
      const meridians = createHorizontalChordMeridians(DEFAULT_RADIUS, DEFAULT_H_MERIDIANS);
      expect(meridians.type).to.equal('Group')
      expect(meridians.constructor.name).to.equal('Meridians')
    });

    it('createHorizontalCircleMeridians() returns type Group from Meridians', () => {
      const meridians = createHorizontalCircleMeridians(DEFAULT_RADIUS, DEFAULT_H_MERIDIANS);
      expect(meridians.type).to.equal('Group')
      expect(meridians.constructor.name).to.equal('Meridians')
    });

    it('createVerticalMeridians() returns type Group from Meridians', () => {
      const meridians = createVerticalMeridians(DEFAULT_RADIUS, DEFAULT_V_MERIDIANS);
      expect(meridians.type).to.equal('Group')
      expect(meridians.constructor.name).to.equal('Meridians')
    });
  });

  describe('configurables', () => {
    it('change the number of horizontal meridians', () => {
      const meridians = createHorizontalChordMeridians(DEFAULT_RADIUS, 51);
      expect(meridians.children.length).to.equal(51);
    });
    it('change the number of vertical meridians', () => {
      const meridians = createVerticalMeridians(DEFAULT_RADIUS, 16);
      expect(meridians.children.length).to.equal(16);
    });
  }); 
});