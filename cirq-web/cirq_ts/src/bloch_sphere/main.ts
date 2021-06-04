import {BlochSphereScene} from './components/Scene.class';
import {CirqBlochSphere} from './CirqBlochSphere.class';
import {Vector} from './components/Vector.class'

export function showSphere(circleData: string, vectorData?: string) {
  const inputData = JSON.parse(circleData);

  const scene = new BlochSphereScene();

  const object = new CirqBlochSphere(inputData.radius).createSphere();
  scene.add(object);

  const vector = Vector.createVector(vectorData || undefined);
  scene.add(vector);
}
