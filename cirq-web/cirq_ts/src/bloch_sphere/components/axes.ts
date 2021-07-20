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

import {Vector3, LineDashedMaterial, BufferGeometry, Line, Group} from 'three';

interface Axis {
  points: [Vector3, Vector3];
  hexColor: string;
  readonly lineWidth: number;
}

/**
 * Generates the axes for the Bloch sphere. The halfLength
 * (length of one half of the axes line) and color of each axis are configurable.
 */
export class Axes extends Group {
  readonly halfLength: number;
  readonly xAxisColor: string = '#1f51ff';
  readonly yAxisColor: string = '#ff3131';
  readonly zAxisColor: string = '#39ff14';

  /**
   * Class constructor.
   * @param halfLength The halfLength of the axes line. This should be the same as the
   * radius of the sphere.
   * @returns An instance of the class containing the generated axes. This can be easily
   * added to the Bloch sphere instance, or the scene itself.
   */
  constructor(halfLength: number) {
    super();
    this.halfLength = halfLength;
    this.generateAxes();
    return this;
  }

  /**
   * Creates the x, y, and z axis for the Bloch sphere, adding
   * them to the group.
   */
  private generateAxes() {
    const LINE_WIDTH = 1.5;
    const xPoints: [Vector3, Vector3] = [
      new Vector3(-this.halfLength, 0, 0),
      new Vector3(this.halfLength, 0, 0),
    ];
    const yPoints: [Vector3, Vector3] = [
      new Vector3(0, 0, -this.halfLength),
      new Vector3(0, 0, this.halfLength),
    ];
    const zPoints: [Vector3, Vector3] = [
      new Vector3(0, -this.halfLength, 0),
      new Vector3(0, this.halfLength, 0),
    ];

    const axesMap = {
      x: this.asLine({
        points: xPoints,
        hexColor: this.xAxisColor,
        lineWidth: LINE_WIDTH,
      }),
      y: this.asLine({
        points: yPoints,
        hexColor: this.yAxisColor,
        lineWidth: LINE_WIDTH,
      }),
      z: this.asLine({
        points: zPoints,
        hexColor: this.zAxisColor,
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
      new LineDashedMaterial({
        color: axis.hexColor,
        linewidth: axis.lineWidth,
        scale: 1,
        dashSize: 0.1,
        gapSize: 0.1,
      })
    ).computeLineDistances();
  }
}
