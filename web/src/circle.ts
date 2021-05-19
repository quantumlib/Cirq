import {Cirq3DScene} from './scene.class';
import {CirqSphere} from './sphere.class';

export function showCircle(circleData: string) {
  // Unused for now
  const inputData = JSON.parse(circleData);
  console.log(inputData.radius);
  console.log(inputData.color);

  const scene = new Cirq3DScene();
  const object = new CirqSphere().getSphere();
  scene.camera.position.z = 10;
  scene.add(object);
}
