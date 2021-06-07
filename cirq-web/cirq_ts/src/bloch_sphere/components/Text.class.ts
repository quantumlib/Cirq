import {
  FontLoader,
  Vector3,
  MeshBasicMaterial,
  TextGeometry,
  Mesh,
} from 'three';

import font_json from '../fonts/helvetiker_regular.typeface.json';

export class Text {
  /**
   * Displays the state labels onto the bloch sphere.
   * @returns A list of text Mesh objects to be rendered by the scene
   */
  public static loadAndDisplayText(): Mesh[] {
    const textLoader = new FontLoader();
    const resultLabels: Mesh[] = [];

    // Get the font
    const font = textLoader.parse(font_json);

    const labelSize = 0.5;
    const labelHeight = 0.1;

    const labels: Record<string, Vector3> = {
      // explicitly typing so we can access later
      '|+>': new Vector3(5, 0, -0.1), // z proportional to the height
      '|->': new Vector3(-5 - labelSize, 0, -0.1),
      'i|->': new Vector3(0, 0, 5),
      'i|+>': new Vector3(0, 0, -5 - labelHeight),
      '|0>': new Vector3(0, 5, 0),
      '|1>': new Vector3(0, -5 - labelSize, 0),
    };

    const materials = [
      new MeshBasicMaterial({color: 0xff0000}), // front
      new MeshBasicMaterial({color: 0xffffff}), // side
    ];

    for (const label in labels) {
      const labelGeo = new TextGeometry(label, {
        font: font,
        size: labelSize,
        height: labelHeight,
      });

      const textMesh = new Mesh(labelGeo, materials);
      textMesh.position.copy(labels[label]);
      textMesh.rotateY(Math.PI / 2);
      resultLabels.push(textMesh);
    }

    return resultLabels;
  }
}
