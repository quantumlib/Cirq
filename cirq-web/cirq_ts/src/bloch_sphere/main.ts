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
 * @param blochSphereConfigJSON A JSON string containing information that configures the
 * bloch sphere.
 * @param containerId A string containing the container (div, span, etc.) id that will contain the visualization
 * output.
 */
export function renderBlochSphere(
  blochSphereConfigJSON: string,
  containerId?: string
) {
  const DEFAULT_RADIUS = 5;
  const DEFAULT_H_MERIDIANS = 7;
  const DEFAULT_V_MERIDIANS = 4;

  const sphereJSONObject = JSON.parse(blochSphereConfigJSON);

  const scene = new BlochSphereScene();
  scene.addSceneToHTMLContainer(containerId || 'container');

  const blochSphere = new BlochSphere(
    sphereJSONObject.radius || DEFAULT_RADIUS,
    sphereJSONObject.hMeridians || DEFAULT_H_MERIDIANS,
    sphereJSONObject.vMeridians || DEFAULT_V_MERIDIANS
  );
  scene.add(blochSphere);

  return blochSphere;
}
