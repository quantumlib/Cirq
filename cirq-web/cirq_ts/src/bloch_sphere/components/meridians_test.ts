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

  describe('defaults', () => {
    it('createHorizontalChordMeridians() generates the correct number of lines given the default number (7)', () => {
      const meridians = new Meridians(
        DEFAULT_RADIUS,
        DEFAULT_H_MERIDIANS,
        Orientation.HORIZONTAL_CHORD
      );
      expect(meridians.children.length).to.equal(7);
    });

    it('createHorizontalCircleMeridians() generates the correct number of lines given the default number (7)', () => {
      const meridians = new Meridians(
        DEFAULT_RADIUS,
        DEFAULT_H_MERIDIANS,
        Orientation.HORIZONTAL
      );
      expect(meridians.children.length).to.equal(7);
    });

    it('createVerticalMeridians() generates the correct number of lines given the default number (4)', () => {
      const meridians = new Meridians(
        DEFAULT_RADIUS,
        DEFAULT_V_MERIDIANS,
        Orientation.VERTICAL
      );
      expect(meridians.children.length).to.equal(4);
    });

    it('createHorizontalChordMeridians() generates lines at the correct positions with defaults', () => {
      const meridians = new Meridians(
        DEFAULT_RADIUS,
        DEFAULT_H_MERIDIANS,
        Orientation.HORIZONTAL_CHORD
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

  describe('configurables', () => {
    it('createHorizontalChordMeridians() generates the correct number of lines given valid numbers', () => {
      const lineValues = [4, 0, 17, 299, 4.123];
      const expectedLineNumbers = [5, 0, 17, 299, 5];
      lineValues.forEach((el, index) => {
        const meridians = new Meridians(
          DEFAULT_RADIUS,
          el,
          Orientation.HORIZONTAL_CHORD
        );
        expect(meridians.children.length).to.equal(expectedLineNumbers[index]);
      });
    });

    it('createHorizontalCircleMeridians() generates the correct number of lines given valid numbers', () => {
      const lineValues = [4, 0, 17, 299, 4.123];
      const expectedLineNumbers = [4, 0, 18, 299, 4];
      lineValues.forEach((el, index) => {
        const meridians = new Meridians(
          DEFAULT_RADIUS,
          el,
          Orientation.HORIZONTAL
        );
        expect(meridians.children.length).to.equal(expectedLineNumbers[index]);
      });
    });

    it('createVerticalMeridians() generates the correct number of lines given valid numbers', () => {
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

    it('createHorizontalChordMeridians() defaults if given invalid inputs (-1, 301)', () => {
      const lineValues = [-1, 301];
      const expectedLineNumbers = [7, 7];
      lineValues.forEach((el, index) => {
        const meridians = new Meridians(
          DEFAULT_RADIUS,
          el,
          Orientation.HORIZONTAL_CHORD
        );
        expect(meridians.children.length).to.equal(expectedLineNumbers[index]);
      });
    });

    it('createHorizontalCircleMeridians() defaults if given invalid inputs (-1, 301)', () => {
      const lineValues = [-1, 301];
      const expectedLineNumbers = [4, 4];
      lineValues.forEach((el, index) => {
        const meridians = new Meridians(
          DEFAULT_RADIUS,
          el,
          Orientation.HORIZONTAL
        );
        expect(meridians.children.length).to.equal(expectedLineNumbers[index]);
      });
    });

    it('createVerticalMeridians() defaults if given invalid inputs (-1, 301)', () => {
      const lineValues = [-1, 301];
      const expectedLineNumbers = [4, 4];
      lineValues.forEach((el, index) => {
        const meridians = new Meridians(
          DEFAULT_RADIUS,
          el,
          Orientation.VERTICAL
        );
        expect(meridians.children.length).to.equal(expectedLineNumbers[index]);
      });
    });

    it('changing the number of horizontal chord meridians changes the positions correctly with scale', () => {
      const meridians = new Meridians(
        DEFAULT_RADIUS,
        9,
        Orientation.HORIZONTAL_CHORD
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

    it('giving bad orientation enum in the constructor will return no circles', () => {
      const meridians = new Meridians(
        DEFAULT_RADIUS,
        DEFAULT_V_MERIDIANS,
        undefined!
      );

      expect(meridians.children.length).to.equal(0);
    });
  });
});
