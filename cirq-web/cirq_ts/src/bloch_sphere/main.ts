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
 */
export function showSphere(circleData: string, vectorData?: string) {
  const inputData = JSON.parse(circleData);

  const scene = new BlochSphereScene();

  const object = new BlochSphere(inputData.radius).getBlochSphere();
  scene.add(object);

  const vector = createVector(vectorData || undefined);
  scene.add(vector);
}
