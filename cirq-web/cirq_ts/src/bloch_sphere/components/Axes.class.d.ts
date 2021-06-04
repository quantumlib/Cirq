import { LineBasicMaterial, BufferGeometry, Line } from 'three';
export declare class Axes {
    static createAxes(radius: number): {
        x: Line<BufferGeometry, LineBasicMaterial>;
        y: Line<BufferGeometry, LineBasicMaterial>;
        z: Line<BufferGeometry, LineBasicMaterial>;
    };
}
