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

import {BlochSphereScene} from './components/scene';
import {BlochSphere} from './bloch_sphere';

/**
 * Adds a Bloch sphere element with relevant, configurable data for the
 * sphere shape of the Bloch sphere and the state vector displayed with it.
 * These elements are added to a Scene object, which is added to the DOM
 * tree in the BlochSphereScene class.
 * @param containerId A string containing the container (div, span, etc.) id that
 * will contain the visualization output.
 * @param radius The radius of the bloch sphere
 * @param hMeridians The designated number of horizontal meridians of the Bloch sphere
 * @param vMeridians The designated number of vertical meridians in the Bloch sphere
 */
export function renderBlochSphere(
  containerId: string,
  radius = 5,
  hMeridians = 7,
  vMeridians = 4
) {
  const scene = new BlochSphereScene();
  scene.addSceneToHTMLContainer(containerId);

  const blochSphere = new BlochSphere(radius, hMeridians, vMeridians);
  scene.add(blochSphere);

  return blochSphere;
}
