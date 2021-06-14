
import puppeteer, { Browser } from 'puppeteer';
import {expect} from 'chai';
import {readFileSync} from 'fs';
import pixelmatch from 'pixelmatch';
import * as PNG from 'pngjs';


/**
 * Generates an HTML script with the current repository bundle
 * that we will use to compare.
 */

// Due to the path, reading the file will only work by running "npm run start"
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


/**
 * Testing to see if they look the same.
 * WRITE ACTUAL TO A TEMP FOLDER
 */
describe('Check Bloch Sphere looks correct', () => {
  //Opens a headless browser with the generated HTML file and takes a screenshot.
  let browser: Browser;

  before(async () => {
    browser = await puppeteer.launch();
    const page = await browser.newPage();
    await page.setContent(browserContent);
    await page.screenshot({path: 'e2e/bloch_sphere/bloch_sphere.png'});   
    });

  after(async () => {
    await browser.close();
  })
  
  it('Bloch sphere with |0⟩ statevector is correct', () => {
    const expected = PNG.PNG.sync.read(readFileSync('e2e/bloch_sphere/bloch_sphere_expected.png'))
    const actual = PNG.PNG.sync.read(readFileSync('e2e/bloch_sphere/bloch_sphere.png'))
    const {width, height} = expected;
    const diff = new PNG.PNG({width, height});

    const pixels = pixelmatch(expected.data, actual.data, diff.data, width, height, {threshold: 0.1});
    expect(pixels).to.equal(0);
  });
});
