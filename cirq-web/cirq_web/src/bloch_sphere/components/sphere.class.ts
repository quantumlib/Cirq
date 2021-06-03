import {SphereGeometry, MeshNormalMaterial, Mesh} from 'three';

export class Sphere {
  sphere: Mesh;

  constructor(radius: number) {
    const geometry = new SphereGeometry(radius, 32, 32);
    const properties = {
      opacity: 0.4,
      transparent: true,
    };

    const material = new MeshNormalMaterial(properties);

    this.sphere = new Mesh(geometry, material);

    // Smooth out the shape
    this.sphere.geometry.computeVertexNormals();

    this.returnSphere();
  }

  returnSphere() {
    return this.sphere;
  }
}
