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

export class Labels extends Group {
  readonly labels: Object;

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

class Label extends Sprite {
  constructor(text: string, positionVector: Vector3) {
    const material = createSpriteMaterial(text);
    super(material);
    this.position.copy(positionVector);
    return this;
  }
}

function createSpriteMaterial(text: string) {
  const CANVAS_SIZE = 256;

  const canvas = document.createElement('canvas');
  canvas.width = CANVAS_SIZE;
  canvas.height = CANVAS_SIZE;

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
