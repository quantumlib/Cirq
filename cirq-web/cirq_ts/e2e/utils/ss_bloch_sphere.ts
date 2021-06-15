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
<html lang="en">
    <head>
    <title>Cirq Web Development page</title>
    </head>
    <body>
    <div id="container"></div>
    <script>${bundle_string}</script>
    <script>
        CirqTS.showSphere('{"radius": 5}');
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
