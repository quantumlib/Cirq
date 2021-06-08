import {EllipseCurve, BufferGeometry, LineBasicMaterial, Line} from 'three';

export class Meridians {
  private static _curveData = {
    anchorX: 0,
    anchor: 0,
    radius: 0, // hacky
    startAngle: 0,
    endAngle: 2 * Math.PI,
    isClockwise: false,
    rotation: 0,
  };

  /**
   * Creates the special horizontal meridian lines of the Bloch
   * sphere, each with a different radius and location.
   * @param radius The radius of the overall Bloch sphere
   * @returns A list of circles (reprsented as Line objs) to draw on the scene
   */
  public static createHorizontalChordMeridians(radius: number): Line[] {
    // Creates chords proportionally to radius 5 circle.
    const initialFactor = (0.5 * radius) / 5;

    const chordYPositions = [];
    const topmostChordPos = radius - initialFactor;
    chordYPositions.push(0); // equator
    for (let i = topmostChordPos; i > 0; i -= topmostChordPos / 3) {
      chordYPositions.push(i);
      chordYPositions.push(-i);
    }

    // Calculate the lengths of the chords of the circle, and then add them
    const meridians = [];
    for (const position of chordYPositions) {
      const hyp2 = Math.pow(radius, 2);
      const distance2 = Math.pow(position, 2);
      const newRadius = Math.sqrt(hyp2 - distance2); //radius^2 - b^2 = a^2

      Meridians._curveData.radius = newRadius;
      const curve = Meridians._createMeridianCurve(this._curveData);
      const meridianLine = Meridians._createMeridianLine(
        curve,
        Math.PI / 2,
        false,
        position
      );
      meridians.push(meridianLine);
    }
    return meridians;
  }

  /**
   * Creates equally sized horizontal meridian lines which rotate
   * by varying degrees across the same axis.
   * @param radius The radius of the overall bloch sphere
   * @returns A list of circles (represented as Line objs) to draw on the scene
   */
  public static createHorizontalCircleMeridians(radius: number): Line[] {
    Meridians._curveData.radius = radius;
    const meridians = [];
    for (let i = 0; i < Math.PI; i += Math.PI / 4) {
      const curve = Meridians._createMeridianCurve(Meridians._curveData);
      const meridianLine = Meridians._createMeridianLine(curve, i);
      meridians.push(meridianLine);
    }
    return meridians;
  }

  /** Creates equally sized vertical meridian lines which rotate
   * by varying degrees across the same axis
   * @param radius The radius of the overall bloch sphere
   * @returns A list of circles (represented as Line objs) to draw on the scene
   */
  public static createVerticalMeridians(radius: number): Line[] {
    const curveData = {
      anchorX: 0,
      anchor: 0,
      radius: radius,
      startAngle: 0,
      endAngle: 2 * Math.PI,
      isClockwise: false,
      rotation: 0,
    };

    const meridians = [];
    for (let i = 0; i < Math.PI; i += Math.PI / 4) {
      const curve = Meridians._createMeridianCurve(curveData);
      const meridianLine = Meridians._createMeridianLine(curve, i, true);
      meridians.push(meridianLine);
    }
    return meridians;
  }

  /**
   * Helper function that generates a necessary EllipseCurve
   * given the required information.
   * @param curveData An object that contains info about the curve
   * @returns An EllipseCurve object based off the curve information.
   */
  private static _createMeridianCurve(curveData: any): EllipseCurve {
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

  /**
   * Helper function that generates the actual Line object which will be
   * rendered by the three.js scene.
   * @param curve An EllipseCurve object that provides location/size info
   * @param rotationFactor The desired angle of rotation in radians
   * @param vertical (Optional) boolean that tells whether or not we're generating a horizontal
   * or vertical line.
   * @param yPosition (Optional) Allows the yPosition of the line to be updated to
   * the provided value
   * @returns A Line object that can be rendered by a three.js scene.
   */
  private static _createMeridianLine(
    curve: EllipseCurve,
    rotationFactor: number,
    vertical?: boolean,
    yPosition?: number
  ): Line {
    const points = curve.getSpacedPoints(128); // Performance impact?
    const meridianGeom = new BufferGeometry().setFromPoints(points);

    vertical
      ? meridianGeom.rotateY(rotationFactor)
      : meridianGeom.rotateX(rotationFactor);

    const meridianLine = new Line(
      meridianGeom,
      new LineBasicMaterial({color: 'gray'})
    );
    if (yPosition) {
      meridianLine.position.y = yPosition;
    }
    return meridianLine;
  }
}
