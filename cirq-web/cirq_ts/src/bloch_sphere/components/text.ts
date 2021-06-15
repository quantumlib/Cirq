import {Vector3, Sprite, Texture, SpriteMaterial} from 'three';

/**
 * Displays the state labels onto the bloch sphere.
 * @returns A list of text Mesh objects to be rendered by the scene
 */
export function loadAndDisplayText(): Sprite[] {
  const resultLabels: Sprite[] = [];

  const labelSize = 0.5;
  const labelHeight = 0.1;
  const labels: Map<string, Vector3> = new Map();
  // State labels are tentative
  labels.set('|+⟩', new Vector3(5.5, 0, -0.1)); // z proportional to the height
  labels.set('|-⟩', new Vector3(-5.5 - labelSize, 0, -0.1));
  labels.set('|-i⟩', new Vector3(0, 0, 5.5));
  labels.set('|i⟩', new Vector3(0, 0, -5.5 - labelHeight));
  labels.set('|0⟩', new Vector3(0, 5.5, 0));
  labels.set('|1⟩', new Vector3(0, -5.5 - labelSize, 0));

  for (const [text, vector] of labels) {
    const sprite = createSprite(text, vector);
    resultLabels.push(sprite);
  }

  return resultLabels;
}

function createSprite(text: string, location: Vector3): Sprite {
  const size = 256;

  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;

  const context = canvas.getContext('2d')!;
  context.fillStyle = '#000000';
  context.textAlign = 'center';
  context.font = '120px Arial';
  context.fillText(text, size / 2, size / 2);

  const map = new Texture(canvas);
  map.needsUpdate = true;

  const material = new SpriteMaterial({
    map: map,
    transparent: true,
  });

  const sprite = new Sprite(material);
  sprite.position.copy(location);

  return sprite;
}
