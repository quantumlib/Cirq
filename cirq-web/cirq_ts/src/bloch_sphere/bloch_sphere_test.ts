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
import {Object3D, Scene} from 'three';
import {JSDOM} from 'jsdom';
import {Orientation} from './components/enums';
import {Sphere} from './components/sphere';
import {Meridians} from './components/meridians';
import {Axes} from './components/axes';

// class EnumerableScene extends Scene {
//   public addedObjects : Object3D[] = [];

//   // Understand what ES6 does, what feature enables it, etc.
//   constructor(){
//     super();
//   }

//   add(...object: Object3D[]): this {
//     this.addedObjects.push(...object);
//     return this;
//   }
// }

/**
 * Using JSDOM to create a global document which the canvas elements
 * generated in loadAndDisplayText can be created on.
 */
const {window} = new JSDOM('<!doctype html><html><body></body></html>');
global.document = window.document;

describe('The BlochSphere class by default', () => {
  const scene = new Scene();
  const bloch_sphere = new BlochSphere();
  bloch_sphere.addToScene(scene);

  it('has a radius of 5', () => {
    // Go to the sphere component of the visuzliation
    // and make sure that its radius is 5;
    const children = bloch_sphere.children;
    const sphere = children.find(
      child => child.constructor.name === 'Sphere'
    ) as Sphere;
    expect(sphere.radius).to.equal(5);
  });

  it('adds a group object to a scene', () => {
    const children = scene.children;
    expect(children.length).to.equal(1);
    expect(children[0].type).to.equal('Group');
  });

  describe('The groups contain the correct num of components', () => {
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

  describe('The meridian group', () => {
    const children = bloch_sphere.children;
    const meridians = children.filter(
      child => child.constructor.name === 'Meridians'
    ) as Meridians[];

    it('contains 7 horizontal chord meridians by default', () => {
      const horizontalMeridians = meridians.find(
        child => child.orientation === Orientation.HORIZONTAL_CHORD
      ) as Meridians;
      expect(horizontalMeridians.numCircles).to.equal(7);
    });

    it('contains 4 vert meridians by default', () => {
      const verticalMeridians = meridians.find(
        child => child.orientation === Orientation.VERTICAL
      ) as Meridians;
      expect(verticalMeridians.numCircles).to.equal(4);
    });
  });

  describe('The axes group', () => {
    const children = bloch_sphere.children;
    const axes = children.find(child => child.constructor.name === 'Axes')!;

    it('contains the 3 axes by default', () => {
      const numberOfLines = axes.children.length;
      expect(numberOfLines).to.equal(3);
    });

    it('The axes fit a circle of radius 5 by default', () => {
      const tcAxes = axes as Axes;
      expect(tcAxes.radius).to.equal(5);
    });

    it('The axes are the right colors by default', () => {
      const tcAxes = axes as Axes;
      expect(tcAxes.xAxisColor).to.equal('#1f51ff');
      expect(tcAxes.yAxisColor).to.equal('#ff3131');
      expect(tcAxes.zAxisColor).to.equal('#39ff14');
    });
  });

  describe('The label group', () => {
    const children = bloch_sphere.children;
  });
});

describe('The BlochSphere class is configurable', () => {
  describe('the radius is configurable', () => {
    it('accepts a radius of 3', () => {
      // Go to the sphere component of the visuzliation
      // and make sure that its radius is 5;
      const bloch_sphere = new BlochSphere(3);
      const children = bloch_sphere.children;
      const sphere = children.find(
        child => child.constructor.name === 'Sphere'
      ) as Sphere;
      expect(sphere.radius).to.equal(3);
    });
  });
});
