import { SphereGeometry, MeshNormalMaterial, Mesh } from 'three';
export declare class Sphere {
    /**
     * Generates a sphere Mesh object, which serves as the foundation
     * of the bloch sphere visualization.
     * @param radius The desired radius of the overall bloch sphere.
     * @returns a sphere Mesh object to be rendered in the scene.
     */
    static createSphere(radius: number): Mesh<SphereGeometry, MeshNormalMaterial>;
}
