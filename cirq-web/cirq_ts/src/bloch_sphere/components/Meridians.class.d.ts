import { Line } from 'three';
export declare class Meridians {
    private static _curveData;
    static createHorizontalChordMeridians(radius: number): Line[];
    static _createHorizontalCircleMeridians(radius: number): Line[];
    static createVerticalMeridians(radius: number): Line[];
    private static _createMeridianCurve;
    private static _createMeridianLine;
}
