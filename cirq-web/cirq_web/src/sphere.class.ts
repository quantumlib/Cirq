import {SphereGeometry, MeshBasicMaterial, MeshNormalMaterial,  Mesh} from 'three';

export class CirqSphere {
  sphere: Mesh;

  constructor(radius = 1, color = 0xff0000) {
    const geometry = new SphereGeometry(radius);
    const material = new MeshBasicMaterial({color: color});
    this.sphere = new Mesh(geometry, material);
  }

  public getSphere() {
    return this.sphere;
  }
}
