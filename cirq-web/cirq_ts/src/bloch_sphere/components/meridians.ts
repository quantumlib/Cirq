import {EllipseCurve, BufferGeometry, LineBasicMaterial, Line} from 'three';

enum Orientation {
  HORIZONTAL,
  VERTICAL,
}

interface CurveData {
  anchorX: number;
  anchorY: number;
  radius: number;
  startAngle: number;
  endAngle: number;
  isClockwise: boolean;
  rotation: number;
}

function curveDataWithRadius(radius: number): CurveData {
  return {
    anchorX: 0,
    anchorY: 0,
    radius: radius,
    startAngle: 0,
    endAngle: 2 * Math.PI,
    isClockwise: false,
    rotation: 0,
  };
}

/**
 * Creates the special horizontal meridian lines of the Bloch
 * sphere, each with a different radius and location.
 * @param radius The radius of the overall Bloch sphere
 * @returns A list of circles (represented as Line objects) to draw on the scene
 */
export function createHorizontalChordMeridians(radius: number): Line[] {
  // Creates chords proportionally to radius 5 circle.
  const initialFactor = (0.5 * radius) / 5;

  const chordYPositions = [0]; // equator
  const topmostChordPos = radius - initialFactor;
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

    const curveData = curveDataWithRadius(newRadius);
    const curve = createMeridianCurve(curveData);
    const meridianLine = createMeridianLine(
      curve,
      Math.PI / 2,
      Orientation.HORIZONTAL,
      position
    );
    meridians.push(meridianLine);
  }
  return meridians;
}

/**
 * Creates equally sized horizontal meridian lines which rotate
 * by varying degrees across the same axis.
 * @param radius The radius of the overall Bloch sphere
 * @returns A list of circles (represented as Line objs) to draw on the scene
 */
export function createHorizontalCircleMeridians(radius: number): Line[] {
  const curveData = curveDataWithRadius(radius);
  const curve = createMeridianCurve(curveData);
  const meridians = [];
  for (let i = 0; i < Math.PI; i += Math.PI / 4) {
    const meridianLine = createMeridianLine(curve, i, Orientation.HORIZONTAL);
    meridians.push(meridianLine);
  }
  return meridians;
}

/**
 * Creates equally sized vertical meridian lines which rotate
 * by varying degrees across the same axis
 * @param radius The radius of the overall bloch sphere
 * @returns A list of circles (represented as Line objs) to draw on the scene
 */
export function createVerticalMeridians(radius: number): Line[] {
  const curveData = {
    anchorX: 0,
    anchorY: 0,
    radius: radius,
    startAngle: 0,
    endAngle: 2 * Math.PI,
    isClockwise: false,
    rotation: 0,
  };

  const meridians = [];
  for (let i = 0; i < Math.PI; i += Math.PI / 4) {
    const curve = createMeridianCurve(curveData);
    const meridianLine = createMeridianLine(curve, i, Orientation.VERTICAL);
    meridians.push(meridianLine);
  }
  return meridians;
}

/**
 * Helper function that generates the actual Line object which will be
 * rendered by the three.js scene.
 * @param curve An EllipseCurve object that provides location/size info
 * @param rotationAngle The desired angle of rotation in radians
 * @param orientation The orientation of the meridian (horizontal or vertical)
 * @param yPosition (Optional) Allows the yPosition of the line to be updated to
 * the provided value
 * @returns A Line object that can be rendered by a three.js scene.
 */
function createMeridianLine(
  curve: EllipseCurve,
  rotationAngle: number,
  orientation: Orientation,
  yPosition?: number
): Line {
  const points = curve.getSpacedPoints(128);
  const meridianGeom = new BufferGeometry().setFromPoints(points);

  orientation === Orientation.VERTICAL
    ? meridianGeom.rotateY(rotationAngle)
    : meridianGeom.rotateX(rotationAngle);

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
function createMeridianCurve(curveData: CurveData): EllipseCurve {
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
