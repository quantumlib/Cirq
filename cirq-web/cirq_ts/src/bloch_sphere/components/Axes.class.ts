import {Vector3, LineBasicMaterial, BufferGeometry, Line} from 'three';
import { addSyntheticLeadingComment } from 'typescript';

interface Axis {
  readonly lineWidth?: number = 1.5;
  points: [Vector3, Vector3],
  hexColor: string,
}

/**
 * Creates the x, y, and z axis for the Bloch sphere.
 * @param radius The overall radius of the bloch sphere.
 * @returns An object mapping the name of the axis to its corresponding
 * Line object to be rendered by the three.js scene.
 */
export function generateAxis(
  radius: number,
  xAxisColor: string = '#1f51ff',
  yAxisColor: string = '#ff3131',
  zAxisColor: string = '#39ff14',
  ) {
  const xPoints : [Vector3, Vector3] = [new Vector3(-radius, 0, 0), new Vector3(radius, 0, 0)];
  const yPoints : [Vector3, Vector3] = [new Vector3(0, 0, -radius), new Vector3(0, 0, radius)];
  const zPoints : [Vector3, Vector3] = [new Vector3(0, -radius, 0), new Vector3(0, radius, 0)];
  
  return {
    x: asLine({points: xPoints, hexColor: xAxisColor}),
    y: asLine({points: yPoints, hexColor: yAxisColor}),
    z: asLine({points: zPoints, hexColor: zAxisColor}),
  }
}

function asLine(axis: Axis) : Line {
  return new Line(
    new BufferGeometry().setFromPoints(axis.points),
    new LineBasicMaterial({color: axis.hexColor, linewidth: axis.lineWidth}));
}
