import { Mesh } from 'three';
export declare class CirqSphere {
    sphere: Mesh;
    constructor(radius?: number, color?: number);
    getSphere(): Mesh<import("three").BufferGeometry, import("three").Material | import("three").Material[]>;
}
