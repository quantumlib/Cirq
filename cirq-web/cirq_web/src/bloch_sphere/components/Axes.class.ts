import {Vector3, LineBasicMaterial, BufferGeometry, Line} from 'three';

export class Axes {
  public static createAxes(radius: number) {
    const xAxis = [new Vector3(0, 0, -radius), new Vector3(0, 0, radius)];

    const yAxis = [new Vector3(-radius, 0, 0), new Vector3(radius, 0, 0)];

    const zAxis = [new Vector3(0, -radius, 0), new Vector3(0, radius, 0)];

    const geometry = new BufferGeometry().setFromPoints(xAxis);
    const blueLine = new Line(
      geometry,
      new LineBasicMaterial({color: '#1f51ff', linewidth: 1.5})
    );

    const geometry2 = new BufferGeometry().setFromPoints(yAxis);
    const redLine = new Line(
      geometry2,
      new LineBasicMaterial({color: '#ff3131', linewidth: 1.5})
    );

    const geometry3 = new BufferGeometry().setFromPoints(zAxis);
    const greenLine = new Line(
      geometry3,
      new LineBasicMaterial({color: '#39ff14', linewidth: 1.5})
    );

    return {
      x: blueLine,
      y: redLine,
      z: greenLine,
    };
  }
}
