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
import {Vector} from './components/vector';

/**
 * Using JSDOM to create a global document which the canvas elements
 * generated in loadAndDisplayText can be created on.
 */
const {window} = new JSDOM('<!doctype html><html><body></body></html>');
global.document = window.document;

describe('BlochSphere defaults', () => {
  const scene = new Scene();
  const bloch_sphere = new BlochSphere();
  bloch_sphere.addToScene(scene);

  it('adds a single object to a scene of type Group, specifically BlochSphere', () => {
    const children = scene.children;
    expect(children.length).to.equal(1);
    expect(children[0].type).to.equal('Group');
    // Sanity check to make sure it works as BlochSphere
    expect(children[0] as BlochSphere).to.be.instanceOf(BlochSphere);
  });

  describe('Child groups (Sphere, Meridians, etc.) contain the correct num of components', () => {
    const children = bloch_sphere.children;
    it('has a sphere', () => {
      const sphereExists = children.find(
        child => child.constructor.name === 'Sphere'
      );
      expect(sphereExists).to.not.equal(undefined);
    });

    it('has 2 sets of meridians', () => {
      const numOfMeridians = children.filter(
        child => child.constructor.name === 'Meridians'
      ).length;
      expect(numOfMeridians).to.equal(2);
    });

    it('has axes', () => {
      const axesExists = children.some(
        child => child.constructor.name === 'Axes'
      );
      expect(axesExists).to.equal(true);
    });

    it('has labels', () => {
      const labelsExists = children.some(
        child => child.constructor.name === 'Labels'
      );
      expect(labelsExists).to.equal(true);
    });
  });

  describe('Sphere', () => {
    it('has a radius of 5 by default', () => {
      const children = bloch_sphere.children;
      const sphere = children.find(
        child => child.constructor.name === 'Sphere'
      ) as Sphere;
      expect(sphere.radius).to.equal(5);
    });
  });

  describe('Meridians', () => {
    const children = bloch_sphere.children;
    const meridians = children.filter(
      child => child.constructor.name === 'Meridians'
    ) as Meridians[];

    const horizontalMeridians = meridians.find(
      child => child.orientation === Orientation.HORIZONTAL
    ) as Meridians;

    const verticalMeridians = meridians.find(
      child => child.orientation === Orientation.VERTICAL
    ) as Meridians;

    it('contains 7 horizontal chord meridians by default', () => {
      expect(horizontalMeridians.numCircles).to.equal(7);
    });

    it('contains 4 vertical meridians by default', () => {
      expect(verticalMeridians.numCircles).to.equal(4);
    });

    it('horizontal chord meridians are gray by default', () => {
      expect(horizontalMeridians.color).to.equal('gray');
    });

    it('vertical meridians are gray by default', () => {
      expect(verticalMeridians.color).to.equal('gray');
    });

    it('horizontal chord meridians share the same radius (5) of the sphere by default', () => {
      expect(horizontalMeridians.radius).to.equal(5);
    });

    it('vertical meridians share the same radius (5) of the sphere by default', () => {
      expect(verticalMeridians.radius).to.equal(5);
    });
  });

  describe('Axes', () => {
    const children = bloch_sphere.children;
    const axes = children.find(
      child => child.constructor.name === 'Axes'
    ) as Axes;

    it('contains the 3 axes by default', () => {
      const numberOfAxisLines = axes.children.length;
      expect(numberOfAxisLines).to.equal(3);
    });

    it('The axes fit a circle of radius 5 by default', () => {
      expect(axes.radius).to.equal(5);
    });

    it('The axes are the right colors by default', () => {
      expect(axes.xAxisColor).to.equal('#1f51ff');
      expect(axes.yAxisColor).to.equal('#ff3131');
      expect(axes.zAxisColor).to.equal('#39ff14');
    });
  });

  describe('Labels', () => {
    const children = bloch_sphere.children;
    const labels = children.find(
      child => child.constructor.name === 'Labels'
    ) as Labels;

    it('The default labels are fit towards the default radius of the sphere (5) with the default spacing (0.5)', () => {
      const expectedBlochSphereLabels = {
        '|+⟩': new Vector3(5.5, 0, 0),
        '|-⟩': new Vector3(-5.5, 0, 0),
        '|i⟩': new Vector3(0, 0, -5.5),
        '|-i⟩': new Vector3(0, 0, 5.5),
        '|0⟩': new Vector3(0, 5.5, 0),
        '|1⟩': new Vector3(0, -5.5, 0),
      };
      expect(labels.labels).to.eql(expectedBlochSphereLabels);
    });
  });
});

describe('BlochSphere configurables', () => {
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

  describe('the radius is configurable', () => {
    it('accepts a radius of 3', () => {
      expect(sphere.radius).to.equal(3);
    });

    it('throws an error correctly with an invalid radius', () => {
      expect(() => new BlochSphere(-100)).to.throw(
        'Invalid radius provided in the BlochSphere constructor'
      );
    });
  });

  describe('the meridians are configurable', () => {
    it('accepts horizontal chord meridians of 9', () => {
      expect(horizontalMeridians.numCircles).to.equal(9);
    });

    it('accepts vertical meridians of 6', () => {
      expect(verticalMeridians.numCircles).to.equal(6);
    });
  });

  describe('vectors can be added', () => {
    // Sanitize checks not needed, since only way vectors may be added
    // is via valid state vectors in Cirq.
    bloch_sphere.addVector('{"x": 1,"y": 1, "z": 2, "length": 5}');
    const vector = children.find(
      child => child.constructor.name === 'Vector'
    ) as Vector;
    expect(vector.x).to.equal(1);
    expect(vector.y).to.equal(1);
    expect(vector.z).to.equal(2);
    expect(vector.length).to.equal(5);
  });
});
