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

import {ArrowHelper, Vector3, Group, LineBasicMaterial} from 'three';

/**
 * Generates a vector to add to the Bloch sphere.
 * The length and direction of the vector are configurable.
 */
export class Vector extends Group {
  readonly scaling_factor: number;
  readonly x: number;
  readonly y: number;
  readonly z: number;

  /**
   * Class constructor.
   * @param x the x coordinate of the vector tip
   * @param y the y coordinate of the vector tip
   * @param z the z coordinate of the vector tip
   * @returns An instance of the class containing the generated vector. This can be
   * added to the Bloch sphere instance as well as the scene.
   */
  constructor(x: number, y: number, z: number, scaling_factor: number) {
    super();
    this.x = x;
    this.y = y;
    this.z = z;
    this.scaling_factor = scaling_factor;

    this.generateVector(this.x, this.y, this.z, this.scaling_factor);
    return this;
  }

  /**
   * Generates a vector starting at (0, 0) and ending at the coordinates
   * of the given parameters, and adds to group.
   * Utilizes three.js ArrowHelper function generate the vector.
   * @param x The x coordinate of the vector tip.
   * @param y The y coordinate of the vector tip.
   * @param z The z coordinate of the vector tip.
   * @param scaling_factor The quantity that will be multiplied by
   * the vector length to fit to the sphere. This will be the sphere's radius.
   */
  private generateVector(
    x: number,
    y: number,
    z: number,
    scaling_factor: number
  ) {
    const directionVector = new Vector3(x, y, z);

    // Apply a -90 degree correction rotation across the x axis
    // to match coords of Cirq with coords of three.js scene.
    // This is necessary to make sure the vector points to the correct state
    const axis = new Vector3(1, 0, 0);
    const angle = -Math.PI / 2;
    directionVector.applyAxisAngle(axis, angle);

    // Set base properties of the vector
    const origin = new Vector3(0, 0, 0);
    const hex = '#800080';
    const headWidth = 1;

    // Calculate the distance, and alter it to be proportional
    // to the length
    const newLength = origin.distanceTo(directionVector) * scaling_factor;

    // Create the arrow representation of the vector and add it to the group
    const arrowHelper = new ArrowHelper(
      directionVector,
      origin,
      newLength,
      hex,
      undefined,
      headWidth
    );

    const arrowLine = arrowHelper.line.material as LineBasicMaterial;
    arrowLine.linewidth = 20;

    this.add(arrowHelper);
  }
}
