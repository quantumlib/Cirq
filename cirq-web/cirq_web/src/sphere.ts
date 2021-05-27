import {Cirq3DScene} from './scene.class';
import {CirqSphere} from './sphere.class';
import {Vector3, ArrowHelper, AxesHelper} from 'three';

export function showSphere(circleData: string) {
  // Unused for now
  const inputData = JSON.parse(circleData);
  console.log(inputData.radius);
  console.log(inputData.color);

  const scene = new Cirq3DScene();
  const object = new CirqSphere(inputData.radius, inputData.color).getSphere();
  scene.camera.position.z = 10;

  scene.add(object);

  const dir = new Vector3(0, 1, 0);
  dir.normalize();

  const origin = new Vector3(0, 0, 0);
  const length = 5;
  const hex = 0x00ff00;
  const headWidth = 1;

  const arrowHelper = new ArrowHelper(dir, origin, length, hex, undefined, headWidth);

  scene.add(arrowHelper);

  const axesHelper = new AxesHelper(3);
  scene.add(axesHelper);
}
