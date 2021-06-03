import {Sphere} from './components/sphere.class';
import {
  Group,
  Vector3,
  LineBasicMaterial,
  Line,
  BufferGeometry,
  FontLoader,
  TextGeometry,
  Mesh,
  MeshBasicMaterial,
  EllipseCurve,
} from 'three';

export class CirqBlochSphere {
  RADIUS: number;
  private _group: Group;
  private _curveData = {
    anchorX: 0,
    anchor: 0,
    radius: 0, // hacky
    startAngle: 0,
    endAngle: 2 * Math.PI,
    isClockwise: false,
    rotation: 0,
  };

  constructor(radius: number) {
    this.RADIUS = radius;
    this._group = new Group();
    this._curveData.radius = radius;
    console.log(typeof this._curveData);

    this._init();
  }

  public returnSphere() {
    return this._group;
  }

  private _init() {
    this._add3dSphere();
    this._addHorizontalChordMeridians();
    this._addVerticalMeridians();
    this._addAxes();
    this._loadAndDisplayText();
  }

  private _add3dSphere() {
    const sphere = new Sphere(this.RADIUS).returnSphere();
    this._group.add(sphere);
  }

  private _addHorizontalChordMeridians() {
    // Creates chords proportionally to radius 5 circle.
    const initialFactor = (0.5 * this.RADIUS) / 5;

    const chordYPositions = [];
    const topmostChordPos = this.RADIUS - initialFactor;
    chordYPositions.push(0); // equator
    for (let i = topmostChordPos; i > 0; i -= topmostChordPos / 3) {
      chordYPositions.push(i);
      chordYPositions.push(-i);
    }

    // Calculate the lengths of the chords of the circle, and then draw them
    for (const position of chordYPositions) {
      const hyp2 = Math.pow(this.RADIUS, 2);
      const distance2 = Math.pow(position, 2);
      const newRadius = Math.sqrt(hyp2 - distance2); //radius^2 - b^2 = a^2

      this._curveData.radius = newRadius;
      const curve = this._createMeridianCurve(this._curveData);
      const meridianLine = this._createMeridianLine(
        curve,
        Math.PI / 2,
        false,
        position
      );
      this._group.add(meridianLine);
    }
  }

  private _addHorizontalCircleMeridians() {
    for (let i = 0; i < Math.PI; i += Math.PI / 4) {
      const curve = this._createMeridianCurve(this._curveData);
      const meridianLine = this._createMeridianLine(curve, i);
      this._group.add(meridianLine);
    }
  }

  private _addVerticalMeridians() {
    const curveData = {
      anchorX: 0,
      anchor: 0,
      radius: this.RADIUS,
      startAngle: 0,
      endAngle: 2 * Math.PI,
      isClockwise: false,
      rotation: 0,
    };

    for (let i = 0; i < Math.PI; i += Math.PI / 4) {
      const curve = this._createMeridianCurve(curveData);
      const meridianLine = this._createMeridianLine(curve, i, true);
      this._group.add(meridianLine);
    }
  }

  private _addAxes() {
    const xAxis = [
      new Vector3(0, 0, -this.RADIUS),
      new Vector3(0, 0, this.RADIUS),
    ];

    const yAxis = [
      new Vector3(-this.RADIUS, 0, 0),
      new Vector3(this.RADIUS, 0, 0),
    ];

    const zAxis = [
      new Vector3(0, -this.RADIUS, 0),
      new Vector3(0, this.RADIUS, 0),
    ];

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

    this._group.add(blueLine);
    this._group.add(redLine);
    this._group.add(greenLine);
  }

  private _createMeridianLine(
    curve: EllipseCurve,
    rotationFactor: number,
    vertical?: boolean,
    yPosition?: number
  ) {
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

  private _createMeridianCurve(curveData: any) {
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

  private _loadAndDisplayText() {
    const textLoader = new FontLoader();
    const fontLink = 'fonts/helvetiker_regular.typeface.json';
    // loading like this may present a problem when bundling
    textLoader.load(fontLink, font => {
      // ES6 arrow notation automatically binds this
      const labelSize = 0.5;
      const labelHeight = 0.1;

      const labels: Record<string, any> = {
        // explicitly typing so we can access later
        '|+>': new Vector3(0, 0, 5),
        '|->': new Vector3(0, 0, -5 - labelHeight),
        'i|+>': new Vector3(5, 0, -0.1), // z proportional to the height
        'i|->': new Vector3(-5 - labelSize, 0, -0.1),
        '|0>': new Vector3(0, 5, 0),
        '|1>': new Vector3(0, -5 - labelSize, 0),
      };

      const materials = [
        new MeshBasicMaterial({color: 0xff0000}), // front
        new MeshBasicMaterial({color: 0xffffff}), // side
      ];

      for (const label in labels) {
        const labelGeo = new TextGeometry(label, {
          font: font,
          size: labelSize,
          height: labelHeight,
        });

        const textMesh = new Mesh(labelGeo, materials);
        textMesh.position.copy(labels[label]);
        this._group.add(textMesh);
      }
    });
  }
}
