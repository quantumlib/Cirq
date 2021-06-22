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
 * Generates a vector to add to the Bloch sphere. The length and direction of the vector are configurable.
 */
export class Vector extends Group {
  readonly length: number;
  readonly x: number;
  readonly y: number;
  readonly z: number;

  /**
   * Class constructor.
   * @param vectorData A JSON string containing information used to build the vector.
   * @returns An instance of the class containing the generated vector. This can be
   * added to the Bloch sphere instance as well as the scene.
   */
  constructor(vectorData?: string) {
    super();

    if (vectorData) {
      const parsedObj = JSON.parse(vectorData);
      this.x = parsedObj.x;
      this.y = parsedObj.y;
      this.z = parsedObj.z;
      this.length = parsedObj.length;
    } else {
      this.x = 0;
      this.y = 0;
      this.z = 0;
      this.length = 5;
    }

    this.generateVector(this.x, this.y, this.z, this.length);
    return this;
  }

  /**
   * Generates a vector starting at (0, 0) and ending at the coordinates
   * of the given parameters, and adds to group.
   * Utilizes three.js ArrowHelper function generate the vector.
   * @param x The x coordinate of the vector tip.
   * @param y The y coordinate of the vector tip.
   * @param z The z coordinate of the vector tip.
   * @param length The length of the vector.
   */
  private generateVector(x: number, y: number, z: number, length: number) {
    const directionVector = new Vector3(x, y, z);

    // Apply a -90 degree correction rotation across the x axis
    // to match coords of Cirq with coords of three.js scene.
    // This is necessary to make sure the vector points to the correct state
    const axis = new Vector3(1, 0, 0);
    const angle = -Math.PI / 2;
    directionVector.applyAxisAngle(axis, angle);

    // Needed so that ArrowHelper can generate the length easily
    directionVector.normalize();

    // Set base properties of the vector
    const origin = new Vector3(0, 0, 0);
    const hex = '#800080';
    const headWidth = 1;

    // Create the arrow representation of the vector and add it to the group
    const arrowHelper = new ArrowHelper(
      directionVector,
      origin,
      length,
      hex,
      undefined,
      headWidth
    );

    const arrowLine = arrowHelper.line.material as LineBasicMaterial;
    arrowLine.linewidth = 20;

    this.add(arrowHelper);
  }
}
