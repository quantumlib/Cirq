import {Vector3, LineBasicMaterial, BufferGeometry, Line} from 'three';

export class Axes {
  /**
   * Creates the x, y, and z axis for the Bloch sphere.
   * @param radius The overall radius of the bloch sphere.
   * @returns An object mapping the name of the axis to its corresponding
   * Line object to be rendered by the three.js scene.
   */
  public static createAxes(radius: number) {
    const xAxis = [new Vector3(-radius, 0, 0), new Vector3(radius, 0, 0)];
    const yAxis = [new Vector3(0, 0, -radius), new Vector3(0, 0, radius)];
    const zAxis = [new Vector3(0, -radius, 0), new Vector3(0, radius, 0)];
    const colors = {
      xAxis: '#1f51ff', // neon blue
      yAxis: '#ff3131', // neon red
      zAxis: '#39ff14', // neon green
    };
    const lineWidth = 1.5;
    const blueLine = Axes._createLine(xAxis, colors.xAxis, lineWidth);
    const redLine = Axes._createLine(yAxis, colors.yAxis, lineWidth);
    const greenLine = Axes._createLine(zAxis, colors.zAxis, lineWidth);
    return {
      x: blueLine,
      y: redLine,
      z: greenLine,
    };
  }

  private static _createLine(endpoints: Array<Vector3>, color: string, lineWidth: number) {
    let geometry : BufferGeometry = new BufferGeometry().setFromPoints(endpoints);
    return new Line(
      geometry,
      new LineBasicMaterial({color: color, linewidth: lineWidth})
    );
  }
}
