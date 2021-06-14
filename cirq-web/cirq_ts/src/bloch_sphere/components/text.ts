import {
  FontLoader,
  Vector3,
  MeshBasicMaterial,
  TextGeometry,
  Mesh,
} from 'three';

import font_json from '../assets/fonts/helvetiker_regular.typeface.json';

const materials = [
  new MeshBasicMaterial({color: 0xff0000}), // front
  new MeshBasicMaterial({color: 0xffffff}), // side
];
/**
 * Displays the state labels onto the bloch sphere.
 * @returns A list of text Mesh objects to be rendered by the scene
 */
export function loadAndDisplayText(): Mesh[] {
  const textLoader = new FontLoader();
  const resultLabels: Mesh[] = [];

  const font = textLoader.parse(font_json);

  const labelSize = 0.5;
  const labelHeight = 0.1;
  const labels: Map<string, Vector3> = new Map();
  // State labels are tentative
  labels.set('|+>', new Vector3(5, 0, -0.1)); // z proportional to the height
  labels.set('|->', new Vector3(-5 - labelSize, 0, -0.1));
  labels.set('|-i>', new Vector3(0, 0, 5));
  labels.set('|i>', new Vector3(0, 0, -5 - labelHeight));
  labels.set('|0>', new Vector3(0, 5, 0));
  labels.set('|1>', new Vector3(0, -5 - labelSize, 0));

  for (const [text, vector] of labels) {
    const labelGeo = new TextGeometry(text, {
      font: font,
      size: labelSize,
      height: labelHeight,
    });

    const textMesh = new Mesh(labelGeo, materials);
    textMesh.position.copy(vector);
    textMesh.rotateY(Math.PI / 2);
    resultLabels.push(textMesh);
  }

  return resultLabels;
}
