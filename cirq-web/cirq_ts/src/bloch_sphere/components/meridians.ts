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

import {
  EllipseCurve,
  BufferGeometry,
  LineBasicMaterial,
  Line,
  Group,
} from 'three';
import {Orientation} from './enums';

interface CurveData {
  anchorX: number;
  anchorY: number;
  radius: number;
  startAngle: number;
  endAngle: number;
  isClockwise: boolean;
  rotation: number;
}

/**
 * Generates the meridians for the Bloch sphere. The radius,
 * number of meridian circles, and the orientation of the circles are configurable.
 */
export class Meridians extends Group {
  readonly radius: number;
  readonly numCircles: number;
  readonly orientation: Orientation;
  readonly color: string = 'gray';

  /**
   * Class constructor.
   * @param radius The radius of the Bloch Sphere meridians. This should be equal
   * to the radius of the sphere instance.
   * @param numCircles The number of circles desired for the given set of meridians.
   * Note that certain types of meridians have certain guidelines to which numbers are possible
   * @param orientation The orientation of the meridians (HORIZONTAL_CHORD, HORIZONTAL, VERTICAL)
   * @returns An instance of the class including generated meridians. This can be
   * added to the Bloch sphere instance as well as the scene.
   */
  constructor(radius: number, numCircles: number, orientation: Orientation) {
    super();
    this.radius = radius;
    this.orientation = orientation;

    switch (orientation) {
      case Orientation.HORIZONTAL: {
        this.numCircles = this.sanitizeCircleInput(numCircles);
        this.createHorizontalChordMeridians(this.radius, this.numCircles);
        return this;
      }
      case Orientation.VERTICAL: {
        this.numCircles = this.sanitizeCircleInput(numCircles);
        this.createVerticalMeridians(this.radius, this.numCircles);
        return this;
      }
      default:
        throw new Error('Invalid orientation input in Meridians constructor');
    }
  }

  /**
   * Creates the special horizontal meridian lines of the Bloch
   * sphere, each with a different radius and location, adding
   * them to the group afterwards.
   * @param radius The radius of the overall Bloch sphere
   * @param numCircles The number of circles displayed. The number must be odd,
   * if an even number is provided. it will round up to the next highest odd number.
   * If 0 < numCircles < 3, 3 meridians will be displayed.
   */
  private createHorizontalChordMeridians(radius: number, numCircles: number) {
    if (numCircles === 0) {
      return;
    }

    let nonEquatorCircles: number;
    numCircles % 2 !== 0
      ? (nonEquatorCircles = numCircles - 1)
      : (nonEquatorCircles = numCircles);
    const circlesPerHalf = nonEquatorCircles / 2;

    // Creates chords proportionally to radius 5 circle.
    const initialFactor = (0.5 * radius) / 5;

    // Start building the chords from the top down.
    // If the user only wants one chord, skip the loop.
    let topmostChordPos: number;
    numCircles === 1
      ? (topmostChordPos = 0)
      : (topmostChordPos = radius - initialFactor);

    const chordYPositions = [0]; // equator
    for (
      let i = topmostChordPos;
      i > 0;
      i -= topmostChordPos / circlesPerHalf
    ) {
      chordYPositions.push(i);
      chordYPositions.push(-i);
    }

    // Calculate the lengths of the chords of the circle, and then add them
    for (const position of chordYPositions) {
      const hyp2 = Math.pow(radius, 2);
      const distance2 = Math.pow(position, 2);
      const newRadius = Math.sqrt(hyp2 - distance2); // radius^2 - b^2 = a^2

      const curveData = this.curveDataWithRadius(newRadius);
      const curve = this.createMeridianCurve(curveData);
      const meridianLine = this.createMeridianLine(
        curve,
        Math.PI / 2,
        Orientation.HORIZONTAL,
        position
      );
      this.add(meridianLine);
    }
  }

  /**
   * Creates equally spaced and sized vertical meridian lines which rotate
   * by varying degrees across the same axis, adding them to the group.
   * @param radius The radius of the overall bloch sphere
   * @param numCircles The number of circles to add. This number must be even,
   * if an odd number is provided, one more circle will be generated to ensure it is even.
   */
  private createVerticalMeridians(radius: number, numCircles: number) {
    if (numCircles === 0) {
      return;
    }

    const curveData = {
      anchorX: 0,
      anchorY: 0,
      radius,
      startAngle: 0,
      endAngle: 2 * Math.PI,
      isClockwise: false,
      rotation: 0,
    };

    for (let i = 0; i < Math.PI; i += Math.PI / numCircles) {
      const curve = this.createMeridianCurve(curveData);
      const meridianLine = this.createMeridianLine(
        curve,
        i,
        Orientation.VERTICAL
      );
      this.add(meridianLine);
    }
  }

  /**
   * Helper function that generates the actual Line object which will be
   * rendered by the three.js scene.
   * @param curve An EllipseCurve object that provides location/size info
   * @param rotationAngle The desired angle of rotation in radians
   * @param orientation The orientation of the meridian (horizontal_chord, horizontal or vertical)
   * @param yPosition (Optional) Allows the yPosition of the line to be updated to
   * the provided value
   * @returns A Line object that can be rendered by a three.js scene.
   */
  private createMeridianLine(
    curve: EllipseCurve,
    rotationAngle: number,
    orientation: Orientation,
    yPosition?: number
  ): Line {
    const points = curve.getSpacedPoints(128);
    const meridianGeom = new BufferGeometry().setFromPoints(points);

    switch (orientation) {
      case Orientation.HORIZONTAL: {
        meridianGeom.rotateX(rotationAngle);
        break;
      }
      case Orientation.VERTICAL: {
        meridianGeom.rotateY(rotationAngle);
        break;
      }
      // No default case is needed, since an invalid orientation input will never make it
      // through the constructor.
    }

    const meridianLine = new Line(
      meridianGeom,
      new LineBasicMaterial({color: 'gray'})
    );
    if (yPosition) {
      meridianLine.position.y = yPosition;
    }
    return meridianLine;
  }

  /**
   * Helper function that generates a necessary EllipseCurve
   * given the required information.
   * @param curveData An object that contains info about the curve
   * @returns An EllipseCurve object based off the curve information.
   */
  private createMeridianCurve(curveData: CurveData): EllipseCurve {
    return new EllipseCurve(
      curveData.anchorX,
      curveData.anchorY,
      curveData.radius,
      curveData.radius,
      curveData.startAngle,
      curveData.endAngle,
      curveData.isClockwise,
      curveData.rotation
    );
  }

  private curveDataWithRadius(radius: number): CurveData {
    return {
      anchorX: 0,
      anchorY: 0,
      radius,
      startAngle: 0,
      endAngle: 2 * Math.PI,
      isClockwise: false,
      rotation: 0,
    };
  }

  private sanitizeCircleInput(input: number) {
    if (input < 0) {
      throw new Error('A negative number of meridians are not supported');
    } else if (input > 300) {
      throw new Error('Over 300 meridians are not supported');
    }

    return Math.floor(input);
  }
}
