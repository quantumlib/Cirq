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

import {
  Scene,
  PerspectiveCamera,
  WebGLRenderer,
  OrthographicCamera,
  Vector3,
  Box3,
} from 'three';
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls';
import {GridCircuit} from './grid_circuit';
import {SymbolInformation} from './components/types';

class CircuitScene extends Scene {
  private WIDTH = 1000;
  private HEIGHT = 700;

  public camera: PerspectiveCamera | OrthographicCamera;
  public renderer: WebGLRenderer;
  public controls: OrbitControls;

  /**
   * Support for both the Perspective and Orthographic
   * cameras offered by three.js
   */
  private perspectiveCamera: PerspectiveCamera;
  private orthographicCamera: OrthographicCamera;
  private perspectiveControls: OrbitControls;
  private orthographicControls: OrbitControls;

  /**
   * Class constructor.
   * Creates a three.js scene object, adds it to the container element
   * at the designated id, and adds the orbit control functionality.
   * @param sceneId The container id that will host the three.js scene object
   * @returns The three.js scene object
   */
  constructor(sceneId: string) {
    super();

    this.renderer = new WebGLRenderer({alpha: true, antialias: true});
    this.renderer.setSize(this.WIDTH, this.HEIGHT);

    this.perspectiveCamera = new PerspectiveCamera(
      75,
      this.WIDTH / this.HEIGHT,
      0.1,
      1000
    );

    this.orthographicCamera = new OrthographicCamera(
      this.WIDTH / this.HEIGHT / -2,
      this.WIDTH / this.HEIGHT / 2,
      this.HEIGHT / this.WIDTH / 2,
      this.HEIGHT / this.WIDTH / -2,
      0.1,
      100
    );
    this.orthographicCamera.zoom = 0.1;
    // The default camera is the Perspective camera
    this.camera = this.perspectiveCamera;

    this.perspectiveControls = new OrbitControls(
      this.perspectiveCamera,
      this.renderer.domElement
    );
    this.orthographicControls = new OrbitControls(
      this.orthographicCamera,
      this.renderer.domElement
    );
    // The default controls are the Orbit controls for the Perspective camera
    this.controls = this.perspectiveControls;

    const el = document.getElementById(sceneId)!;
    el.appendChild(this.renderer.domElement);

    this.animate();
  }

  /**
   * Orients the camera and controls based off of the generated circuit,
   * setting its position, zoom, and orientation.
   * @param circuit The circuit that the camera is oriented off of.
   */
  setCameraAndControls(circuit: GridCircuit) {
    // Set the camera positions
    this.fitPerspectiveCamera(circuit);
    this.fitOrthographicCamera(circuit);

    // Tells the camera which way is up.
    this.camera.up.set(0, 1, 0);

    // The camera will always look at circuit to start
    this.camera.lookAt(circuit.position);

    // Tells the control anchor to be set to the center
    // of the circuit.
    const center = new Box3().setFromObject(circuit).getCenter(new Vector3());
    this.controls.target.set(center.x, center.y, center.z);
  }

  /**
   * Toggles between a Perspective and Orthographic camera, resetting
   * the camera to the default orientation each time.
   * @param circuit The circuit that the camera is oriented off of.
   */
  toggleCamera(circuit: GridCircuit) {
    if (this.camera instanceof PerspectiveCamera) {
      this.camera = this.orthographicCamera;
      this.controls = this.orthographicControls;
    } else {
      this.camera = this.perspectiveCamera;
      this.controls = this.perspectiveControls;
    }

    this.setCameraAndControls(circuit);
  }

  private animate() {
    requestAnimationFrame(this.animate.bind(this));
    this.controls.update();
    this.renderer.render(this, this.camera);
  }

  private fitPerspectiveCamera(circuit: GridCircuit) {
    const boundingBox = new Box3();
    boundingBox.setFromObject(circuit);

    const size = boundingBox.getSize(new Vector3());

    const max = Math.max(size.x, size.y, size.z);
    const fov = this.perspectiveCamera.fov * (Math.PI / 180);

    const z = Math.abs(max / Math.sin(fov / 2));

    this.perspectiveCamera.position.x = 0;
    this.perspectiveCamera.position.y = 2.5;
    this.perspectiveCamera.position.z = z;
  }

  private fitOrthographicCamera(circuit: GridCircuit) {
    // TODO: Issue #4395. For more precision, calculate the bounding box of the
    // circuit and set the orthographic camera to the four corners
    // plus an offset.
    this.orthographicCamera.position.x = 1;
    this.orthographicCamera.position.y = 2;
    this.orthographicCamera.zoom = 0.07;
    this.orthographicCamera.updateProjectionMatrix();

    this.orthographicCamera.lookAt(circuit.position);
  }
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
 * @param padding_factor A number that represents how much the visualization
 * should be scaled on the x,z coordinate plane. Default is 1.
 * @returns A GridCircuit object
 */
export function createGridCircuit(
  symbol_info: SymbolInformation[],
  numMoments: number,
  sceneId: string,
  padding_factor = 1
): {circuit: GridCircuit; scene: CircuitScene} {
  const scene = new CircuitScene(sceneId);

  const circuit = new GridCircuit(numMoments, symbol_info, padding_factor);
  scene.add(circuit);
  scene.setCameraAndControls(circuit);

  return {circuit, scene};
}
