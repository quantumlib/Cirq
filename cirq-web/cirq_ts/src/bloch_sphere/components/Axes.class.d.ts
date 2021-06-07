import { LineBasicMaterial, BufferGeometry, Line } from 'three';
export declare class Axes {
    /**
     * Creates the x, y, and z axis for the bloch_sphere.
     * @param radius The overall radius of the bloch sphere.
     * @returns An object mapping the name of the axis to its corresponding
     * Line object to be rendered by the three.js scene.
     */
    static createAxes(radius: number): {
        x: Line<BufferGeometry, LineBasicMaterial>;
        y: Line<BufferGeometry, LineBasicMaterial>;
        z: Line<BufferGeometry, LineBasicMaterial>;
    };
}
