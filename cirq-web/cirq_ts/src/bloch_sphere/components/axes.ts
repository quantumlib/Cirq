// Copyright 2021 The Cirq Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import {Vector3, LineBasicMaterial, BufferGeometry, Line, Group} from 'three';

interface Axis {
  points: [Vector3, Vector3];
  hexColor: string;
  readonly lineWidth: number;
}

export class Axes extends Group {
  readonly radius: number;
  readonly xAxisColor: string;
  readonly yAxisColor: string;
  readonly zAxisColor: string;

  constructor(
    radius: number,
    xAxisColor = '#1f51ff',
    yAxisColor = '#ff3131',
    zAxisColor = '#39ff14'
  ) {
    super();
    this.radius = radius;
    this.xAxisColor = xAxisColor;
    this.yAxisColor = yAxisColor;
    this.zAxisColor = zAxisColor;

    this.generateAxes(
      this.radius,
      this.xAxisColor,
      this.yAxisColor,
      this.zAxisColor
    );
    return this;
  }

  /**
   * Creates the x, y, and z axis for the Bloch sphere, adding
   * them to the group.
   * @param radius The overall radius of the bloch sphere.
   */
  private generateAxes(
    radius: number,
    xAxisColor: string,
    yAxisColor: string,
    zAxisColor: string
  ) {
    const LINE_WIDTH = 1.5;
    const xPoints: [Vector3, Vector3] = [
      new Vector3(-radius, 0, 0),
      new Vector3(radius, 0, 0),
    ];
    const yPoints: [Vector3, Vector3] = [
      new Vector3(0, 0, -radius),
      new Vector3(0, 0, radius),
    ];
    const zPoints: [Vector3, Vector3] = [
      new Vector3(0, -radius, 0),
      new Vector3(0, radius, 0),
    ];

    const axesMap = {
      x: this.asLine({
        points: xPoints,
        hexColor: xAxisColor,
        lineWidth: LINE_WIDTH,
      }),
      y: this.asLine({
        points: yPoints,
        hexColor: yAxisColor,
        lineWidth: LINE_WIDTH,
      }),
      z: this.asLine({
        points: zPoints,
        hexColor: zAxisColor,
        lineWidth: LINE_WIDTH,
      }),
    };

    this.add(axesMap.x);
    this.add(axesMap.y);
    this.add(axesMap.z);
  }

  private asLine(axis: Axis): Line {
    return new Line(
      new BufferGeometry().setFromPoints(axis.points),
      new LineBasicMaterial({color: axis.hexColor, linewidth: axis.lineWidth})
    );
  }
}
