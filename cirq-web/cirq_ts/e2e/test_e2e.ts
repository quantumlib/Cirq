var fs = require('fs');
import * as puppeteer from 'puppeteer';
import looksSame = require('looks-same')
import {expect} from 'chai';

/**
 * Generates an HTML script with the current repository bundle
 * that we will use to compare.
 */

// Due to the path, reading the file will only work by running "npm run start"
let bundle_string = fs.readFileSync('dist/bloch_sphere.bundle.js')
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
        createSphere.showSphere('{"radius": 5}');
    </script>
    </body>
</html>
`;

/**
 * Opens a headless browser with the generated HTML file
 * and takes a screenshot.
 */
(async() => {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    await page.setContent(browserContent);
    await page.screenshot({path: 'actual/bloch_sphere_0.png'});
    await browser.close();
})();


/**
 * Testing to see if they look the same. 
 */
describe('Check Bloch Sphere looks correct', () => {
    it('BS with |0⟩ statevector is correct', () => {
        looksSame('actual/bloch_sphere_0.png', 'expected/bloch_sphere_0.png', function(error, {equal}){
            expect(equal).to.equal(true);
        })
    })
})
