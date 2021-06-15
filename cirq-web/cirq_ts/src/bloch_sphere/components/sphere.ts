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

import {SphereGeometry, MeshNormalMaterial, Mesh} from 'three';

/**
 * Generates a sphere Mesh object, which serves as the foundation
 * of the bloch sphere visualization.
 * @param radius The desired radius of the overall bloch sphere.
 * @returns a sphere Mesh object to be rendered in the scene.
 */
export function createSphere(radius: number) {
  const geometry = new SphereGeometry(radius, 32, 32);
  const properties = {
    opacity: 0.4,
    transparent: true,
  };

  const material = new MeshNormalMaterial(properties);

  const sphere = new Mesh(geometry, material);

  // Smooth out the shape
  sphere.geometry.computeVertexNormals();

  return sphere;
}
