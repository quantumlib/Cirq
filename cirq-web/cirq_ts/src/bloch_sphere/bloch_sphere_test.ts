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
import {Scene, Vector3} from 'three';
import {JSDOM} from 'jsdom';
import {Orientation} from './components/enums';
import {Sphere} from './components/sphere';
import {Meridians} from './components/meridians';
import {Axes} from './components/axes';
import {Labels} from './components/text';
import {StateVector} from './components/state_vector';

/**
 * Using JSDOM to create a global document which the canvas elements
 * generated in loadAndDisplayText can be created on.
 */
const {window} = new JSDOM('<!doctype html><html><body></body></html>');
global.document = window.document;

describe('BlochSphere (with empty constructor)', () => {
  const scene = new Scene();
  const bloch_sphere = new BlochSphere();
  scene.add(bloch_sphere);

  // Extract all of the Bloch sphere default info first,
  // and then test that it's what we want.
  const children = bloch_sphere.children;

  const sphere = children.find(
    child => child.constructor.name === 'Sphere'
  ) as Sphere;

  const meridians = children.filter(
    child => child.constructor.name === 'Meridians'
  ) as Meridians[];

  const horizontalMeridians = meridians.find(
    child => child.orientation === Orientation.HORIZONTAL
  ) as Meridians;

  const verticalMeridians = meridians.find(
    child => child.orientation === Orientation.VERTICAL
  ) as Meridians;

  const axes = children.find(
    child => child.constructor.name === 'Axes'
  ) as Axes;

  const labels = children.find(
    child => child.constructor.name === 'Labels'
  ) as Labels;

  it('adds a single BlochSphere of type Group', () => {
    const children = scene.children;
    expect(children.length).to.equal(1);
    expect(children[0].type).to.equal('Group');
    // Sanity check to make sure it works as BlochSphere
    expect(children[0] as BlochSphere).to.be.instanceOf(BlochSphere);
  });

  describe('child group (Sphere, Meridians, etc.)', () => {
    it('Sphere contains the correct number of components', () => {
      const sphereExists = children.find(
        child => child.constructor.name === 'Sphere'
      );
      expect(sphereExists).to.not.equal(undefined);
    });

    it('Meridians appear in 2 sets', () => {
      const numOfMeridians = children.filter(
        child => child.constructor.name === 'Meridians'
      ).length;
      expect(numOfMeridians).to.equal(2);
    });

    it('Axes exist', () => {
      const axesExists = children.some(
        child => child.constructor.name === 'Axes'
      );
      expect(axesExists).to.equal(true);
    });

    it('Labels exist', () => {
      const labelsExists = children.some(
        child => child.constructor.name === 'Labels'
      );
      expect(labelsExists).to.equal(true);
    });
  });

  describe('child group Sphere', () => {
    it('has a radius of 5 by default', () => {
      expect(sphere.radius).to.equal(5);
    });
  });

  describe('child groups Meridians', () => {
    describe('by default', () => {
      it('contains 7 horizontal chord meridians', () => {
        expect(horizontalMeridians.numCircles).to.equal(7);
      });

      it('contains 4 vertical meridians', () => {
        expect(verticalMeridians.numCircles).to.equal(4);
      });

      it('has horizontal chord meridians that share the same radius of the sphere', () => {
        expect(horizontalMeridians.radius).to.equal(sphere.radius);
      });

      it('has vertical meridians that share the same radius of the sphere', () => {
        expect(verticalMeridians.radius).to.equal(sphere.radius);
      });
    });
  });

  describe('child group Axes', () => {
    it('contains the 3 axes by default', () => {
      const numberOfAxisLines = axes.children.length;
      expect(numberOfAxisLines).to.equal(3);
    });

    it('has axes lines with half-length equal to the radius of the sphere', () => {
      expect(axes.halfLength).to.equal(sphere.radius);
    });

    it('does not have any 2 lines with the same color', () => {
      const colors = [axes.xAxisColor, axes.yAxisColor, axes.zAxisColor];
      expect(new Set(colors).size).to.equal(colors.length);
    });
  });

  describe('child group Labels', () => {
    it('has labels that are fit to the radius of the sphere with the correct spacing (0.5)', () => {
      const expectedBlochSphereLabels = {
        '|+⟩': new Vector3(sphere.radius + 0.5, 0, 0),
        '|-⟩': new Vector3(-(sphere.radius + 0.5), 0, 0),
        '|i⟩': new Vector3(0, 0, -(sphere.radius + 0.5)),
        '|-i⟩': new Vector3(0, 0, sphere.radius + 0.5),
        '|0⟩': new Vector3(0, sphere.radius + 0.5, 0),
        '|1⟩': new Vector3(0, -(sphere.radius + 0.5), 0),
      };
      expect(labels.labels).to.eql(expectedBlochSphereLabels);
    });
  });
});

describe('BlochSphere (with valid custom constructor values)', () => {
  const bloch_sphere = new BlochSphere(3, 9, 6);
  const children = bloch_sphere.children;

  const sphere = children.find(
    child => child.constructor.name === 'Sphere'
  ) as Sphere;

  const meridians = children.filter(
    child => child.constructor.name === 'Meridians'
  ) as Meridians[];

  const horizontalMeridians = meridians.find(
    child => child.orientation === Orientation.HORIZONTAL
  ) as Meridians;

  const verticalMeridians = meridians.find(
    child => child.orientation === Orientation.VERTICAL
  ) as Meridians;

  describe('has a child group Sphere', () => {
    it('with a configured radius of 3', () => {
      expect(sphere.radius).to.equal(3);
    });
  });

  describe('has a child group Meridians', () => {
    describe('with a horizontal set of meridians that', () => {
      it('accepts the configured number of meridians', () => {
        expect(horizontalMeridians.numCircles).to.equal(9);
      });
    });

    describe('with a vertical set of meridians that', () => {
      it('accepts the configured number of meridians', () => {
        expect(verticalMeridians.numCircles).to.equal(6);
      });
    });
  });

  describe('handles vectors correctly by', () => {
    it('adding three unique vectors without failing', () => {
      // Adding multiple vectors is supported in the prototype,
      // but the front end isn't yet formatted to accommodate this.
      // The next PR will contain this additional support.

      bloch_sphere.addVector(1, 0, 0);
      bloch_sphere.addVector(1, 1, 2);
      bloch_sphere.addVector(3, 4, 5);

      const vectors = children.filter(
        child => child.constructor.name === 'Vector'
      ) as StateVector[];

      const expectedVectorX = [1, 1, 3];
      const expectedVectorY = [0, 1, 4];
      const expectedVectorZ = [0, 2, 5];

      vectors.forEach((vector, index) => {
        expect(vector.x).to.equal(expectedVectorX[index]);
        expect(vector.y).to.equal(expectedVectorY[index]);
        expect(vector.z).to.equal(expectedVectorZ[index]);
        expect(vector.blochSphereRadius).to.equal(sphere.radius);
      });
    });
  });
});

describe('BlochSphere (with invalid custom constructor values)', () => {
  it('fails correctly if given an invalid radius', () => {
    const inputs = [0, -1];
    const errorMessage =
      'The radius of a Sphere must be greater than or equal to 1';

    inputs.forEach(input => {
      expect(() => new BlochSphere(input)).to.throw(errorMessage);
    });
  });

  it('fails correctly if given invalid horizontal meridian input', () => {
    const inputs = [-2, 301];
    const errorMessages = [
      'A negative number of meridians are not supported',
      'Over 300 meridians are not supported',
    ];

    inputs.forEach((input, i) => {
      expect(() => new BlochSphere(5, input)).to.throw(errorMessages[i]);
    });
  });

  it('fails correctly if given invalid vertical meridian input', () => {
    const inputs = [-2, 301];
    const errorMessages = [
      'A negative number of meridians are not supported',
      'Over 300 meridians are not supported',
    ];

    inputs.forEach((input, i) => {
      expect(() => new BlochSphere(5, 0, input)).to.throw(errorMessages[i]);
    });
  });
});
