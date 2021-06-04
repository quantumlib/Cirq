import {Vector3, LineBasicMaterial, BufferGeometry, Line} from 'three';

export class Axes {
  /**
   * Creates the x, y, and z axis for the bloch_sphere.
   * @param radius The overall radius of the bloch sphere.
   * @returns An object mapping the name of the axis to its corresponding
   * Line object to be rendered by the three.js scene.
   */
  public static createAxes(radius: number) {
    const xAxis = [new Vector3(0, 0, -radius), new Vector3(0, 0, radius)];
    const yAxis = [new Vector3(-radius, 0, 0), new Vector3(radius, 0, 0)];
    const zAxis = [new Vector3(0, -radius, 0), new Vector3(0, radius, 0)];
    const colors = {
      xAxis: '#1f51ff', // neon blue
      yAxis: '#ff3131', // neon red
      zAxis: '#39ff14' // neon green
    }
    const lineWidth = 1.5;

    let geometry: BufferGeometry;
    geometry = new BufferGeometry().setFromPoints(xAxis);
    const blueLine = new Line(
      geometry,
      new LineBasicMaterial({color: colors.xAxis, linewidth: lineWidth})
    );

    geometry = new BufferGeometry().setFromPoints(yAxis);
    const redLine = new Line(
      geometry,
      new LineBasicMaterial({color: colors.yAxis, linewidth: lineWidth})
    );

    geometry = new BufferGeometry().setFromPoints(zAxis);
    const greenLine = new Line(
      geometry,
      new LineBasicMaterial({color: colors.zAxis, linewidth: lineWidth})
    );

    return {
      x: blueLine,
      y: redLine,
      z: greenLine,
    };
  }
}
