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
import {Axes} from './axes';
import {Line, LineDashedMaterial, Color} from 'three';

describe('Axes', () => {
  describe('by default', () => {
    const axes = new Axes(5);
    const children = axes.children as Line[];

    it('returns 3 Line objects', () => {
      expect(children.length).to.equal(3);
    });

    it('returns the correct default colors for each line', () => {
      const defaultColors = ['#1f51ff', '#ff3131', '#39ff14'];

      children.forEach((el, index) => {
        const material = el.material as LineDashedMaterial;
        expect(material.color).to.eql(new Color(defaultColors[index]));
      });
    });

    it('returns all lines with a constant linewidth (1.5)', () => {
      children.forEach(el => {
        const material = el.material as LineDashedMaterial;
        expect(material.linewidth).to.equal(1.5);
      });
    });
  });
});
