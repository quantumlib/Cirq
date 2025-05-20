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

import {expect} from 'chai';
import {Labels, Label} from './text';
import {JSDOM} from 'jsdom';
import {Vector3} from 'three';

/**
 * Using JSDOM to create a global document which the canvas elements
 * generated can be created on.
 */
const {window} = new JSDOM('<!doctype html><html><body></body></html>');
global.document = window.document;

describe('Labels', () => {
  const mockCanvas = document.createElement('canvas');
  mockCanvas.width = 256;
  mockCanvas.height = 256;

  const mock_label = {
    test: new Vector3(0, 0, 0),
  };
  const labels = new Labels(mock_label);

  it('successfully generates an arbitrary label at the correct position', () => {
    const label = labels.children[0] as Label;
    expect(label.position).to.eql(new Vector3(0, 0, 0));
    expect(label.text).to.equal(Object.keys(mock_label)[0]);
  });
});
