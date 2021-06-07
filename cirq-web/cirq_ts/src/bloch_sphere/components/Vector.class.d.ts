import { ArrowHelper } from 'three';
export declare class Vector {
    /**
     * Adds a state vector to the bloch sphere.
     * @param vectorData information representing the location of the vector
     * @returns an ArrowHelper object to be rendered by the scene.
     */
    static createVector(vectorData?: string): ArrowHelper;
}
