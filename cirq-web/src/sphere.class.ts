import {SphereGeometry, MeshBasicMaterial, Mesh} from 'three';

export class CirqSphere {
  sphere: Mesh;

  constructor(radius = 1) {
    const geometry = new SphereGeometry(radius);
    const material = new MeshBasicMaterial({color: 0xff0000});
    this.sphere = new Mesh(geometry, material);
  }

  public getSphere() {
    return this.sphere;
  }
}
