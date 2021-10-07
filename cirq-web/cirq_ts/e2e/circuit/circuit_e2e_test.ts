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
import {expect} from 'chai';
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
const bundleString = readFileSync('dist/circuit.bundle.js');
function htmlContent(clientCode: string) {
  return `
    <!doctype html>
    <meta charset="UTF-8">
    <html lang="en">
      <head>
      <title>Cirq Web Development page</title>
      </head>
      <body>
      <div id="mycircuitdiv"></div>
      <script>${bundleString}</script>
      <script>${clientCode}</script>
      </body>
    </html>
    `;
}

// Automatically track and cleanup files on exit
temp.track();

describe('Circuit', () => {
  temp.mkdir('tmp', (err, dirPath) => {
    const outputPath = path.join(dirPath, 'circuit.png');

    before(async () => {
      const browser = await puppeteer.launch({args: ['--app']});
      const page = await browser.newPage();

      // Take a screenshot of the first image
      await page.setContent(
        htmlContent(`
      const circuit = createGridCircuit(
        [
            {
                'wire_symbols': ['Z'], 
                'location_info': [{'row': 2, 'col': 3}], 
                'color_info': ['cyan'], 
                'moment': 0
            },
            {   
                'wire_symbols': ['X'], 
                'location_info': [{'row': 2, 'col': 3}], 
                'color_info': ['black'], 
                'moment': 1
            },
            {   
                'wire_symbols': ['@', 'X'], 
                'location_info': [{'row': 3, 'col': 0}, {'row': 0, 'col': 0}], 
                'color_info': ['black', 'black'], 
                'moment': 0
            },
        ], 5, 'mycircuitdiv'
        );
      `)
      );
      await page.screenshot({path: outputPath});
      await browser.close();
    });

    it('with limited gates matches the gold copy', () => {
      const expected = PNG.PNG.sync.read(
        readFileSync('e2e/circuit/circuit_expected.png')
      );
      const actual = PNG.PNG.sync.read(readFileSync(outputPath));
      const {width, height} = expected;
      const diff = new PNG.PNG({width, height});

      const pixels = pixelmatch(
        expected.data,
        actual.data,
        diff.data,
        width,
        height,
        {threshold: 0.1}
      );

      expect(pixels).to.equal(0);
    });
  });
});
