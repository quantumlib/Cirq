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

import {SphereGeometry, MeshNormalMaterial, Mesh, Group} from 'three';

/**
 * Generates the sphere shape of the Bloch sphere. The radius is configurable.
 */
export class Sphere extends Group {
  readonly radius: number;

  /**
   * Class constructor
   * @param radius the desired radius of the sphere
   * @returns An instance of the class containing the generated sphere. This can be
   * added to the Bloch sphere instance as well as the scene.
   */
  constructor(radius: number) {
    super();

    if (radius < 1) {
      throw new Error(
        'The radius of a Sphere must be greater than or equal to 1'
      );
    } else {
      this.radius = radius;
    }

    this.createSphere(this.radius);
    return this;
  }

  /**
   * Generates a sphere Mesh object, which serves as the foundation
   * of the Bloch sphere visualization, adding the mesh object
   * to the group.
   * @param radius The desired radius of the overall bloch sphere.
   */
  private createSphere(radius: number) {
    const geometry = new SphereGeometry(radius, 32, 32);
    const properties = {
      opacity: 0.6,
      transparent: true,
    };

    const material = new MeshNormalMaterial(properties);

    const mesh = new Mesh(geometry, material);

    // Smooth out the shape
    mesh.geometry.computeVertexNormals();

    this.add(mesh);
  }
}
