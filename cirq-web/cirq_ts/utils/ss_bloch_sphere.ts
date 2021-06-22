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

/**
 * Usage:
 * From the cirq_ts directory, run:
 *  ts-node e2e/utils/ss_bloch_sphere.ts
 */
import puppeteer from 'puppeteer';
import {readFileSync} from 'fs';

const bundle_string = readFileSync('dist/bloch_sphere.bundle.js');
const browserContent = `
<!doctype html>
<meta charset="UTF-8">
<html lang="en">
    <head>
    <title>Cirq Web Development page</title>
    </head>
    <body>
    <div id="container"></div>
    <script>${bundle_string}</script>
    <script>
        CirqTS.blochSphere('{"radius": 5}', '{"x": 1,"y": 1, "z": 2, "length": 5}');
    </script>
    </body>
</html>
`;

(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  await page.setContent(browserContent);
  await page.screenshot({path: 'e2e/bloch_sphere/bloch_sphere_expected.png'});

  await browser.close();
})();
