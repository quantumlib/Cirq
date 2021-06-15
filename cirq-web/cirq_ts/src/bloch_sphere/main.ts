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
import {createVector} from './components/vector';
/**
 * Adds a Bloch sphere element with relevant, configurable data for the
 * sphere shape of the Bloch sphere and the state vector displayed with it.
 * These elements are added to a Scene object, which is added to the DOM
 * tree in the BlochSphereScene class.
 * @param circleData A JSON string containing information that configures the
 * bloch sphere.
 * @param vectorData A JSON string containing information that configures the
 * state vector.
 * @param divId A string containing the div id that will contain the visualization
 * output.
 */
export function showSphere(
  circleData: string,
  vectorData?: string,
  divId?: string
) {
  const inputData = JSON.parse(circleData);
  const scene = new BlochSphereScene(divId);

  const bloch_sphere = new BlochSphere(inputData.radius).getBlochSphere();
  scene.add(bloch_sphere);

  const vector = createVector(vectorData || undefined);
  scene.add(vector);
}
