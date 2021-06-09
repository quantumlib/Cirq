import {SphereGeometry, MeshNormalMaterial, Mesh} from 'three';

/**
 * Generates a sphere Mesh object, which serves as the foundation
 * of the bloch sphere visualization.
 * @param radius The desired radius of the overall bloch sphere.
 * @returns a sphere Mesh object to be rendered in the scene.
 */
export function createSphere(radius: number) {
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
