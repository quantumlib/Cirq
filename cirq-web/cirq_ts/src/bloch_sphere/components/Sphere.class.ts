import {SphereGeometry, MeshNormalMaterial, Mesh} from 'three';

export class Sphere {
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
