import {Cirq3DScene} from './scene.class';
import {CirqSphere} from './sphere.class';
import {Vector3, ArrowHelper, AxesHelper} from 'three';

const scene = new Cirq3DScene();

export function showSphere(circleData: string) {
  // Unused for now
  const inputData = JSON.parse(circleData);
  console.log(`circleData: ${circleData}`);

  
  const object = new CirqSphere(inputData.radius, inputData.color).getSphere();
  scene.camera.position.z = 10;

  scene.add(object);

  // Add axes to the sphere
  const axesHelper = new AxesHelper(4);
  scene.add(axesHelper);
}

export function addVector(vectorData: string) {
  const inputData = JSON.parse(vectorData);
  console.log(`vectorData: ${vectorData}`);

  // Create and normalize the new vector
  const dir = new Vector3(inputData.x, inputData.y, inputData.z);
  dir.normalize();

  // Set base properties of the vector
  const origin = new Vector3(0, 0, 0);
  const length = inputData.v_length;
  const hex = '#800080';
  const headWidth = 1;

  // Create the arrow representation of the vector and add it to the scene
  const arrowHelper = new ArrowHelper(dir, origin, length, hex, undefined, headWidth);
  scene.add(arrowHelper);
}

