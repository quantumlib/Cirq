// Copyright 2021 The Cirq Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import {Vector3, Sprite, Texture, SpriteMaterial, Group} from 'three';

/**
 * Generates and groups together many Labels. This should be used when generating
 * labels for dedicated visualizations.
 */
export class Labels extends Group {
  readonly labels: Object;

  /**
   * Class constructor.
   * @param labels An object mapping the desired text to be rendered with its location on the scene
   * @returns an instance of the class containing all of the generated labels. All labels can
   * be added to the Bloch sphere instance as well as the scene.
   */
  constructor(labels: Object) {
    super();
    this.labels = labels;

    this.generateLabels(this.labels);
    return this;
  }

  private generateLabels(labels: Object) {
    for (const [text, location] of Object.entries(labels)) {
      this.add(new Label(text, location));
    }
  }
}

/**
 * Creates a Sprite rendering of text which can be added to the visualization.
 * For dedicated visualizations, all Labels should be grouped together and created
 * by using the Labels constructor.
 */
export class Label extends Sprite {
  readonly text: string;

  /**
   * Class constructor
   * @param text The text of the label
   * @param positionVector The position of the vector as a Vector3 object
   * @returns a Label object with specified text and position.
   */
  constructor(text: string, positionVector: Vector3) {
    const material = createSpriteMaterial(text);
    super(material);
    this.text = text;
    this.position.copy(positionVector);
    return this;
  }
}

/**
 * Turns text into a Sprite.
 * @param text The text you want to turn into a Sprite.
 * @returns A SpriteMaterial that can be rendered by three.js
 */
function createSpriteMaterial(text: string) {
  const CANVAS_SIZE = 256;

  const canvas = document.createElement('canvas');
  canvas.width = CANVAS_SIZE;
  canvas.height = CANVAS_SIZE;
  // Allows us to keep track of what we're adding to the
  // canvas.
  canvas.textContent = text;

  const context = canvas.getContext('2d')!;
  context.fillStyle = '#000000';
  context.textAlign = 'center';
  context.font = '120px Arial';
  context.fillText(text, CANVAS_SIZE / 2, CANVAS_SIZE / 2);

  const map = new Texture(canvas);
  map.needsUpdate = true;

  return new SpriteMaterial({
    map: map,
    transparent: true, // for a transparent canvas background
  });
}
