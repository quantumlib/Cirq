import {ArrowHelper, Vector3} from 'three';

/**
 * Adds a state vector to the bloch sphere.
 * @param vectorData information representing the location of the vector.
 * @returns an ArrowHelper object to be rendered by the scene.
 */
interface Vector {
  x: number;
  y: number;
  z: number;
  v_length: number;
}

export function createVector(vectorData?: string): ArrowHelper {
  let inputData: Vector;
  if (vectorData) {
    inputData = JSON.parse(vectorData);
  } else {
    inputData = {
      x: 0,
      y: 0,
      z: 0,
      v_length: 5,
    };
  }

  // Create and normalize the new vector
  const directionVector = new Vector3(inputData.x, inputData.y, inputData.z);
  directionVector.normalize();

  // Set base properties of the vector
  const origin = new Vector3(0, 0, 0);
  const length = inputData.v_length;
  const hex = '#800080';
  const headWidth = 1;

  // Create the arrow representation of the vector and add it to the scene
  const arrowHelper = new ArrowHelper(
    directionVector,
    origin,
    length,
    hex,
    undefined,
    headWidth
  );

  return arrowHelper;
}
