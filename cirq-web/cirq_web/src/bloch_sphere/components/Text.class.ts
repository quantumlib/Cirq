import {
  FontLoader,
  Vector3,
  MeshBasicMaterial,
  TextGeometry,
  Mesh,
} from 'three';

export class Text {
  public static loadAndDisplayText(): Mesh[] {
    const textLoader = new FontLoader();
    const resultLabels: Mesh[] = [];
    // Did this because of bundling
    textLoader.load('./fonts/helvetiker_regular.typeface.json', font => {
      // ES6 arrow notation automatically binds this
      const labelSize = 0.5;
      const labelHeight = 0.1;

      const labels: Record<string, Vector3> = {
        // explicitly typing so we can access later
        '|+>': new Vector3(0, 0, 5),
        '|->': new Vector3(0, 0, -5 - labelHeight),
        'i|+>': new Vector3(5, 0, -0.1), // z proportional to the height
        'i|->': new Vector3(-5 - labelSize, 0, -0.1),
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
        resultLabels.push(textMesh);
      }
    });
    return resultLabels;
  }
}
