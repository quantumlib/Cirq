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

import {Orientation} from './components/enums';
import {Sphere} from './components/sphere';
import {Axes} from './components/axes';
import {Meridians} from './components/meridians';
import {Labels} from './components/text';
import {StateVector} from './components/state_vector';

import {Group, Vector3} from 'three';

/**
 * Class bringinging together the individual components like the
 * Sphere, Axes, Meridians, and Text into the overall visualization
 * of the Bloch sphere.
 */
export class BlochSphere extends Group {
  private radius: number;
  private hMeridians: number;
  private vMeridians: number;

  /**
   * Class constructor.
   * @param radius The radius of the Bloch sphere
   * @param hMeridians The number of horizontal meridians desired for the Bloch sphere.
   * @param vMeridians The number of vertical meridians desired for the Bloch sphere.
   * @returns An instance of the class which can be easily added to a scene.
   */
  constructor(radius = 5, hMeridians = 7, vMeridians = 4) {
    super();

    this.radius = radius;
    this.hMeridians = hMeridians;
    this.vMeridians = vMeridians;

    this.addSphere();
    this.addHorizontalMeridians();
    this.addVerticalMeridians();
    this.addAxes();
    this.addLabels();

    return this;
  }

  /**
   * Adds a vector to the Bloch sphere based on a set
   * of given x, y, and z coordinates of the vector's endpoint
   * in order to be parsed successfully.
   * @param x the x coordinate of the vector's endpoint
   * @param y the y coordinate of the vector's endpoint
   * @param z the z coordinate of the vector's endpoint
   *
   */
  public addVector(x: number, y: number, z: number) {
    const vector = new StateVector(x, y, z, this.radius);
    this.add(vector);
  }

  private addSphere() {
    const sphere = new Sphere(this.radius);
    this.add(sphere);
  }

  private addAxes() {
    const axes = new Axes(this.radius);
    this.add(axes);
  }

  private addHorizontalMeridians() {
    const meridians = new Meridians(
      this.radius,
      this.hMeridians,
      Orientation.HORIZONTAL
    );
    this.add(meridians);
  }

  private addVerticalMeridians() {
    const meridians = new Meridians(
      this.radius,
      this.vMeridians,
      Orientation.VERTICAL
    );
    this.add(meridians);
  }

  private addLabels() {
    const spacing = 0.5;
    const labelData = {
      '|+⟩': new Vector3(this.radius + spacing, 0, 0),
      '|-⟩': new Vector3(-this.radius - spacing, 0, 0),
      '|i⟩': new Vector3(0, 0, -this.radius - spacing),
      '|-i⟩': new Vector3(0, 0, this.radius + spacing),
      '|0⟩': new Vector3(0, this.radius + spacing, 0),
      '|1⟩': new Vector3(0, -this.radius - spacing, 0),
    };

    const labels = new Labels(labelData);
    this.add(labels);
  }
}
