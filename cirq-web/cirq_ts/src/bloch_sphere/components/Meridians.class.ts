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

  public static _createHorizontalCircleMeridians(radius: number): Line[] {
    Meridians._curveData.radius = radius;
    const meridians = [];
    for (let i = 0; i < Math.PI; i += Math.PI / 4) {
      const curve = Meridians._createMeridianCurve(Meridians._curveData);
      const meridianLine = Meridians._createMeridianLine(curve, i);
      meridians.push(meridianLine);
    }
    return meridians;
  }

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
