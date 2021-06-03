import {Group} from 'three';
export declare class CirqBlochSphere {
  RADIUS: number;
  private _group;
  private _curveData;
  constructor(radius: number);
  returnSphere(): Group;
  private _init;
  private _add3dSphere;
  private _addHorizontalChordMeridians;
  private _addHorizontalCircleMeridians;
  private _addVerticalMeridians;
  private _addAxes;
  private _createMeridianLine;
  private _createMeridianCurve;
  private _loadAndDisplayText;
}
