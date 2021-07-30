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

import {Scene, PerspectiveCamera, WebGLRenderer, OrthographicCamera} from 'three';
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls';
import {GridCircuit} from './grid_circuit';
import {Coord} from './components/types';

/**
 * Creates a three.js scene object, adds it to the container element
 * at the designated id, and adds the orbit control functionality.
 * @param numQubits The number of qubits in the circuit
 * @param sceneId The container id that will host the three.js scene object
 * @returns The three.js scene object
 */
function createAndRenderScene(numQubits: number, sceneId: string): Scene {
  const WIDTH = 1000;
  const HEIGHT = 700;
  const NUM_QUBITS = numQubits;

  const scene = new Scene();
  const camera = new PerspectiveCamera(75, WIDTH / HEIGHT, 0.1, 1000);
  //const camera = new OrthographicCamera(WIDTH / -2 , WIDTH / 2, HEIGHT / 2, HEIGHT / -2, 1, 1000);

  const renderer = new WebGLRenderer({alpha: true});
  const controls = new OrbitControls(camera, renderer.domElement);

  renderer.setSize(WIDTH, HEIGHT);
  const el = document.getElementById(sceneId)!;

  el.appendChild(renderer.domElement);

  // TODO: create a better way to start the
  // camera in the center of the scene.
  camera.position.x = 0;
  camera.position.z = 0;
  camera.position.y = 2.5;

  controls.target.set(0, 2.5, 0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.screenSpacePanning = false;
  controls.minDistance = 5;
  controls.maxDistance = 200;
  controls.maxPolarAngle = Math.PI;

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  return scene;
}

/**
 * Creates and returns an empty GridCircuit object with qubits at the
 * designated coordinates. The returned GridCircuit object can then take
 * input to add gates to the circuit.
 * @param qubits A list of GridCoord objects representing the location of
 * each qubit.
 * @param numMoments The number of total moments in the circuit
 * @param sceneId The container id with the three.js scene that will render
 * the three.js components
 * @returns A GridCircuit object
 */
export function createGridCircuit(
  qubits: Coord[],
  numMoments: number,
  sceneId: string
): GridCircuit {
  const scene = createAndRenderScene(qubits.length, sceneId);

  const circuit = new GridCircuit(numMoments, qubits);
  scene.add(circuit);

  return circuit;
}
