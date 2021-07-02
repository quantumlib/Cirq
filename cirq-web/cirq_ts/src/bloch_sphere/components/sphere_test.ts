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
import {Sphere} from './sphere';
import {Mesh, MeshNormalMaterial, SphereGeometry} from 'three';

function getSphereInfo(sphere: Sphere) {
  const mesh = sphere.children[0] as Mesh;
  const material = mesh.material as MeshNormalMaterial;
  const geometry = mesh.geometry as SphereGeometry;
  const params = geometry.parameters;

  return {
    mesh: mesh,
    material: material,
    geometry: geometry,
    params: params,
  };
}

describe('Sphere', () => {
  const DEFAULT_RADIUS = 5;
  const sphere = new Sphere(DEFAULT_RADIUS);

  describe('by default', () => {
    const sphereInfo = getSphereInfo(sphere);

    it('has a default radius of 5', () => {
      expect(sphereInfo.params.radius).to.equal(5);
    });

    it('uses 32 width and height segments for the sphere shape', () => {
      expect(sphereInfo.params.widthSegments).to.equal(32);
      expect(sphereInfo.params.widthSegments).to.equal(32);
    });

    it('returns a transparent sphere', () => {
      expect(sphereInfo.material.opacity).to.equal(0.6);
      expect(sphereInfo.material.transparent).to.equal(true);
    });
  });

  describe('allows configuration that', () => {
    it('accepts valid radius inputs (1, 14.2, 100)', () => {
      const radiusInputs = [1, 14.2, 100];
      const expectedRadius = [1, 14.2, 100];

      radiusInputs.forEach((el, index) => {
        const sphere = new Sphere(el);
        const sphereInfo = getSphereInfo(sphere);

        expect(sphereInfo.params.radius).to.equal(expectedRadius[index]);
      });
    });

    describe('throws the correct errors if the user', () => {
      it('gives invalid radius inputs (-1, 0)', () => {
        const radiusInputs = [-1, 0];
        const expectedErrorMessage =
          'The radius of a Sphere must be greater than or equal to 1';
        radiusInputs.forEach(el => {
          expect(() => new Sphere(el)).to.throw(expectedErrorMessage);
        });
      });
    });
  });
});
