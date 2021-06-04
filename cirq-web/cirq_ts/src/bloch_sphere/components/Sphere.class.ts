import {SphereGeometry, MeshNormalMaterial, Mesh} from 'three';

export class Sphere {
  /**
   * Generates a sphere Mesh object, which serves as the foundation
   * of the bloch sphere visualization.
   * @param radius The desired radius of the overall bloch sphere.
   * @returns a sphere Mesh object to be rendered in the scene.
   */
  public static createSphere(radius: number) {
    const geometry = new SphereGeometry(radius, 32, 32);
    const properties = {
      opacity: 0.4,
      transparent: true,
    };

    const material = new MeshNormalMaterial(properties);

    const sphere = new Mesh(geometry, material);

    // Smooth out the shape
    sphere.geometry.computeVertexNormals();

    return sphere;
  }
}
