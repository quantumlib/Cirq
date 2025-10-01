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

import puppeteer from 'puppeteer';
import {beforeAll, describe, it, expect} from 'vitest';
import {readFileSync} from 'fs';
import pixelmatch from 'pixelmatch';
import * as PNG from 'pngjs';
import * as temp from 'temp';
import * as path from 'path';

/**
 * Generates an HTML script with the current repository bundle
 * that we will use to compare.
 */

// Due to the path, reading the file will only work by running this file in the same directory
// as the package.json file.
const bundleString = readFileSync('dist/bloch_sphere.bundle.js');
function htmlContent(clientCode: string) {
  return `
    <!doctype html>
    <meta charset="UTF-8">
    <html lang="en">
      <head>
      <title>Cirq Web Development page</title>
      </head>
      <body>
      <div id="container"></div>
      <script>${bundleString}</script>
      <script>${clientCode}</script>
      </body>
    </html>
    `;
}

/**
 * Testing to see if they look the same.
 */

// Automatically track and cleanup files on exit
temp.track();

describe('Bloch sphere', () => {
  const dirPath = temp.mkdirSync('tmp');
  const outputPath = path.join(dirPath, 'bloch_sphere');
  const newVectorOutputPath = path.join(dirPath, 'bloch_sphere_vec');

  beforeAll(async () => {
    const browser = await puppeteer.launch({args: ['--app']});
    const page = await browser.newPage();

    await page.setContent(htmlContent("renderBlochSphere('container')"));
    await page.screenshot({path: `${outputPath}.png`});
    await browser.close();
  });

  it('with no vector matches the gold PNG', () => {
    const expected = PNG.PNG.sync.read(
      readFileSync('e2e/bloch_sphere/bloch_sphere_expected.png'),
    );
    const actual = PNG.PNG.sync.read(readFileSync(`${outputPath}.png`));
    const {width, height} = expected;
    const diff = new PNG.PNG({width, height});

    const pixels = pixelmatch(expected.data, actual.data, diff.data, width, height, {
      threshold: 0.1,
    });

    expect(pixels).toBe(0);
  });

  beforeAll(async () => {
    const browser = await puppeteer.launch({args: ['--app']});
    const page = await browser.newPage();

    await page.setContent(
      htmlContent("renderBlochSphere('container').addVector(0.5, 0.5, 0.5);"),
    );
    await page.screenshot({path: `${newVectorOutputPath}.png`});
    await browser.close();
  });

  it('with custom statevector matches the gold PNG', () => {
    const expected = PNG.PNG.sync.read(
      readFileSync('e2e/bloch_sphere/bloch_sphere_expected_custom.png'),
    );
    const actual = PNG.PNG.sync.read(readFileSync(`${newVectorOutputPath}.png`));
    const {width, height} = expected;
    const diff = new PNG.PNG({width, height});

    const pixels = pixelmatch(expected.data, actual.data, diff.data, width, height, {
      threshold: 0.1,
    });

    expect(pixels).toBe(0);
  });
});
