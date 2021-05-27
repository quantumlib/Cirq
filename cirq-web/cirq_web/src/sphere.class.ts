import {SphereGeometry, MeshBasicMaterial, MeshNormalMaterial,  Mesh} from 'three';

export class CirqSphere {
  sphere: Mesh;

  constructor(radius = 1, color = 0xff0000) {
    const geometry = new SphereGeometry(radius, 8, 8);
    const properties = {
      color: color,
      opacity: 0.5,
      transparent: true,
    }
    const material = new MeshBasicMaterial(properties);
    //const material = new MeshNormalMaterial(properties);

    this.sphere = new Mesh(geometry, material);

    // Smooth out the shape
    this.sphere.geometry.computeVertexNormals();

  }

  public getSphere() {
    return this.sphere;
  }
}
