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

import {createSphere} from './components/sphere';
import {generateAxis} from './components/axes';
import {
  createHorizontalChordMeridians,
  createVerticalMeridians,
} from './components/meridians';
import {generateLabels} from './components/text';
import {generateVector} from './components/vector';

import {BlochSphereScene} from './components/scene';
import {Group, Scene, Vector3} from 'three';

/**
 * Class bringinging together the individual components like the
 * Sphere, Axes, Meridians, and Text into the overall visualization
 * of the Bloch sphere.
 */

export class BlochSphere extends Group {
  private radius: number;
  private group: Group;

  // Pull logic of where labels are into here, and not 
  // into the bloch sphere

  // Class that contains the default config paramaters, which
  // are overridable, and then sets up all the components
  // with the right configuration

  constructor(radius = 5) {
    super();

    this.radius = radius;
    this.userData.radius = radius;

    this.group = new Group();
    this.addSphere();
    this.addHorizontalMeridians();
    this.addVerticalMeridians();
    this.addAxes();
    this.addLabels();

    return this;
  }

  /**
   * Adds the Bloch sphere to the designated Three.js Scene.
   * @param scene A Three.js scene object
   */
  addToScene(scene: Scene | BlochSphereScene) {
    scene.add(this);
  }

  public addVector(vectorData?: string) {
    const vector = generateVector(vectorData);
    this.add(vector);
  }

  private addSphere() {
    const sphere = createSphere(this.radius);
    this.add(sphere);
  }

  private addAxes() {
    const axes = generateAxis(this.radius);
    this.add(axes);
  }

  private addHorizontalMeridians() {
    const meridians = createHorizontalChordMeridians(this.radius, 7);
    this.add(meridians);
  }

  private addVerticalMeridians() {
    const meridians = createVerticalMeridians(this.radius, 4);
    this.add(meridians);
  }

  private addLabels() {
    // Location of labels go's here
    // label = new Label(text, direction)\
    const spacing = 0.5;
    const labels = {
      '|+⟩' : new Vector3(this.radius + spacing, 0, 0),
      '|-⟩' : new Vector3(-this.radius - spacing, 0, 0),
      '|i⟩' : new Vector3(0, 0, -this.radius - spacing),
      '|-i⟩' : new Vector3(0, 0, this.radius + spacing),
      '|0⟩' : new Vector3(0, this.radius + spacing, 0),
      '|1⟩' : new Vector3(0, -this.radius - spacing, 0),
    }

    this.add(generateLabels(labels));
  }
}
