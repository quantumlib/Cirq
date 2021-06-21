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
import {Labels} from './text';
import {JSDOM} from 'jsdom';
import {Vector3} from 'three';

/**
 * Using JSDOM to create a global document which the canvas elements
 * generated in loadAndDisplayText can be created on.
 */
const {window} = new JSDOM('<!doctype html><html><body></body></html>');
global.document = window.document;

describe('Text methods', () => {
  const mock_label = {
    test: new Vector3(0, 0, 0),
  };
  const labels = new Labels(mock_label);

  it('returns a type Group as Labels', () => {
    expect(labels.type).to.equal('Group');
    expect(labels.constructor.name).to.equal('Labels');
  });

  it('has one child with one label', () => {
    const children = labels.children;
    expect(children.length).to.equal(1);
    expect(children[0].type).to.equal('Sprite');
    expect(children[0].constructor.name).to.equal('Label');
  });
});
