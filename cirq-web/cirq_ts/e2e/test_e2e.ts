const puppeteer = require('puppeteer');
(async() => {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    await page.goto('test_file.html');
    await page.screenshot({path: 'expected.png'});

    await browser.close();
})();