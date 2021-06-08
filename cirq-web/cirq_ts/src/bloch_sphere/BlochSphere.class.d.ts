import { Group } from 'three';
export declare class CirqBlochSphere {
    RADIUS: number;
    private _group;
    constructor(radius: number);
    createSphere(): Group;
    private _init;
    private _add3dSphere;
    private _addAxes;
    private _addHorizontalMeridians;
    private _addVerticalMeridians;
    private _loadAndDisplayText;
}
