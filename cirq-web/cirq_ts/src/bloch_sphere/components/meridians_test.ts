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
import {Meridians} from './meridians';
import {Orientation} from './enums';
import {Vector3} from 'three';

describe('Meridians', () => {
  const DEFAULT_RADIUS = 5;
  const DEFAULT_H_MERIDIANS = 7;
  const DEFAULT_V_MERIDIANS = 4;

  describe('by default', () => {
    it(`generates the correct number of lines given 
    the default number (7)`, () => {
      const meridians = new Meridians(
        DEFAULT_RADIUS,
        DEFAULT_H_MERIDIANS,
        Orientation.HORIZONTAL
      );
      expect(meridians.children.length).to.equal(DEFAULT_H_MERIDIANS);
    });

    it(`generates the correct number of lines given 
    the default number (4)`, () => {
      const meridians = new Meridians(
        DEFAULT_RADIUS,
        DEFAULT_V_MERIDIANS,
        Orientation.VERTICAL
      );
      expect(meridians.children.length).to.equal(DEFAULT_V_MERIDIANS);
    });

    it(`generates lines at the correct positions 
    with defaults`, () => {
      const meridians = new Meridians(
        DEFAULT_RADIUS,
        DEFAULT_H_MERIDIANS,
        Orientation.HORIZONTAL
      );

      const positions = [
        new Vector3(0, 0, 0),
        new Vector3(0, 4.5, 0),
        new Vector3(0, -4.5, 0),
        new Vector3(0, 3, 0),
        new Vector3(0, -3, 0),
        new Vector3(0, 1.5, 0),
        new Vector3(0, -1.5, 0),
      ];

      meridians.children.forEach((el, index) => {
        expect(el.position).to.eql(positions[index]);
      });
    });
  });

  describe('allows configuration that', () => {
    it('generates the correct number of horizontal meridians given valid numbers', () => {
      // Note that due to asethetic choices, if the number of circles
      // provided is even, createHorizontalChordMeridians() will automatically
      // adjust to building an odd number of meridians.
      const lineValues = [4, 0, 17, 299, 4.123, 1];
      const expectedLineNumbers = [5, 0, 17, 299, 5, 1];
      lineValues.forEach((el, index) => {
        const meridians = new Meridians(
          DEFAULT_RADIUS,
          el,
          Orientation.HORIZONTAL
        );
        expect(meridians.children.length).to.equal(expectedLineNumbers[index]);
      });
    });

    it('generates the correct number of vertical meridians given valid numbers', () => {
      const lineValues = [4, 0, 17, 299, 4.123];
      const expectedLineNumbers = [4, 0, 18, 299, 4];
      lineValues.forEach((el, index) => {
        const meridians = new Meridians(
          DEFAULT_RADIUS,
          el,
          Orientation.VERTICAL
        );
        expect(meridians.children.length).to.equal(expectedLineNumbers[index]);
      });
    });

    it('changes the number of horizontal chord meridians correctly with scale', () => {
      const meridians = new Meridians(
        DEFAULT_RADIUS,
        9,
        Orientation.HORIZONTAL
      );

      const positions = [
        new Vector3(0, 0, 0),
        new Vector3(0, 4.5, 0),
        new Vector3(0, -4.5, 0),
        new Vector3(0, 3.375, 0),
        new Vector3(0, -3.375, 0),
        new Vector3(0, 2.25, 0),
        new Vector3(0, -2.25, 0),
        new Vector3(0, 1.125, 0),
        new Vector3(0, -1.125, 0),
      ];

      meridians.children.forEach((el, index) => {
        expect(el.position).to.eql(positions[index]);
      });
    });

    describe('throws the correct errors if the user', () => {
      it('gives invalid horizontal meridian inputs (-1, 301)', () => {
        const lineValues = [-1, 301];
        const expectedErrorMessage = [
          'A negative number of meridians are not supported',
          'Over 300 meridians are not supported',
        ];
        lineValues.forEach((el, index) => {
          expect(() => {
            new Meridians(DEFAULT_RADIUS, el, Orientation.HORIZONTAL);
          }).to.throw(expectedErrorMessage[index]);
        });
      });

      it('gives invalid vertical meridian inputs (-1, 301)', () => {
        const lineValues = [-1, 301];
        const expectedErrorMessage = [
          'A negative number of meridians are not supported',
          'Over 300 meridians are not supported',
        ];
        lineValues.forEach((el, index) => {
          expect(() => {
            new Meridians(DEFAULT_RADIUS, el, Orientation.VERTICAL);
          }).to.throw(expectedErrorMessage[index]);
        });
      });

      it('throws an error correctly if given a bad orientation enum in the constructor', () => {
        expect(() => {
          new Meridians(DEFAULT_RADIUS, DEFAULT_V_MERIDIANS, undefined!);
        }).to.throw('Invalid orientation input in Meridians constructor');
      });
    });
  });
});
