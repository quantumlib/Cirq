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

import {ArrowHelper, Vector3, Group} from 'three';

/**
 * Adds a state vector to the bloch sphere.
 * @param vectorData information representing the location of the vector.
 * @returns an ArrowHelper object to be rendered by the scene.
 */

class Vectors extends Group {
  constructor(){
    super();
  }
}

interface Vector {
  x: number;
  y: number;
  z: number;
  length: number;
}


export function generateVector(inputData?: string): Vectors {
  let vectorData: Vector;
  if (inputData) {
    vectorData = JSON.parse(inputData);
  } else {
    vectorData = {x: 0, y: 0, z: 0, length: 5,};
  }

  const directionVector = new Vector3(vectorData.x, vectorData.y, vectorData.z);

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
  const length = vectorData.length;
  const hex = '#800080';
  const headWidth = 1;

  // Create the arrow representation of the vector and add it to the scene
  const arrowHelper = new ArrowHelper(
    directionVector,
    origin,
    length,
    hex,
    undefined,
    headWidth
  );

  const vectors = new Vectors();
  vectors.add(arrowHelper);

  return vectors;
}
