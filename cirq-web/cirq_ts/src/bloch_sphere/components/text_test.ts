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
import {loadAndDisplayText} from './text';
import {JSDOM} from 'jsdom';

/**
 * Using JSDOM to create a global document which the canvas elements
 * generated in loadAndDisplayText can be created on.
 */
const {window} = new JSDOM('<!doctype html><html><body></body></html>');
global.document = window.document;

describe('Text methods', () => {
  const textItems = loadAndDisplayText();

  it('returns a list of Sprite objects', () => {
    for (const text of textItems) {
      expect(text.type).to.equal('Sprite');
    }
  });

  it('returns 6 valid labels, one for each state', () => {
    expect(textItems.length).to.equal(6);
  });
});
