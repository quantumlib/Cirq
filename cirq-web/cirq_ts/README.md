## Cirq Typescript Development

*This directory contains the code and instructions for visualization tools in a web browser or Colab/Juptyer notebooks. We do this using Typescript. While necessary, this is only half of the code needed to run the `cirq-web` project. For information on how to integrate projects here with Python and the wider Cirq package, [see the `cirq_web` directory](../cirq_web/).*

Visualizations run on [NodeJS](https://nodejs.org/en/), and we use [npm](https://www.npmjs.com/) for package management. To start developing, clone the repository and run `npm install` within this directory, or `check/npm install` from the top level directory, to install the necessary packages and begin development. You will need to install Node and npm if you haven't already.


For developing 3D visualizations, we rely on the [three.js](https://threejs.org/) framework.

For bundling the Typescript into Javascript that can be run in the browser, and for overall ease of development, we use [Webpack](https://webpack.js.org/).

As an additional note, all `npm` and `npx` commands can be ran from the top-level of Cirq like so:
```bash
# check/npm runs npm --prefix 'cirq-web/cirq_ts` and forwards arguments
check/npm [YOUR_COMMAND]

# check/npx navigates to this directory and runs from there
check/npx [YOUR_COMMAND]
```

### Visualization build structure

The reference example for the build structure of a visualization is the Bloch sphere. Reference the `src/bloch_sphere/main.ts` file and the `src/bloch_sphere/bloch_sphere.ts` file to see the code. The `src/bloch_sphere/` directory should serve as a guide for how Typescript visualizations in Cirq should be structured. Visualizations should have:
 - A "root" folder within the `src/` directory labeled according to the title of the visualization. All files and directories for a particular visualization will live here. In the case of the Bloch Sphere, this is `bloch_sphere/`.
 - A `components/` directory which contains classes representing different components of the larger visualization, following typical object oriented programming techniques. In the case of the Bloch sphere, you can see that we have different classes for `Axes`, `Meridians`, `Text` etc.
 - Any `assets/` directory with information necessary for the visualization (fonts, images, etc.). In the case of the Bloch Sphere, we can see a `fonts/` subdirectory which holds necessary font data, within the `assets/` directory, but for instances where there isn't a lot of extra information necessary subdirectories may not be needed.
 - A class within the visualization's "root" folder which brings the individual components of the visualization together. In the case of the Bloch Sphere, this is `bloch_sphere.ts`.
 - A `main.ts` consisting of functions which will be called from the bundled library. These function should handle:
    - Receiving any input data which could affect the visualization.
    - Sending final visualization output to the development environment, notebook, or HTML files. 
    - Combining aspects of the visualization that need to be added separately.

This `main.ts` file will also need to be added as an entry point in the `webpack.config.js` file in order for your visualization to be bundled accordingly.
```javascript
module.exports = {
  entry: {
    bloch_sphere: './src/bloch_sphere/main.ts',
    ...
    YOUR_VIZ_NAME: './src/YOUR_VIZ_NAME/main.ts',
  },
  ...
};
```
You can learn more about Webpack entry points here: [Webpack Entry Points](https://webpack.js.org/concepts/entry-points/)

### Creating visualization bundle files

Following this structure, you will be able to bundle your visualization by running the command `npx webpack --mode production` in this directory, or `check/ts-build` from the top-level directory.  This will build the bundled Javascript file(s) within the `dist/` directory, where you can access and reference them in HTML.

### Developing visualizations

There are two main ways to develop visualizations you are creating in Cirq. The first, and recommended way, is to spin up a Webpack development server and view your visualizations in the browser. You can also develop using Jupyter notebook if you want to easily test integration with Python code. 

#### Hot reloading development server (Recommended)

Using `webpack-dev-server`, we are able to develop and test visualizations in the browser and have changes update as we're writing the code. You can start this server by running `npm run start` in this directory, and view your work on the port specified by Webpack. This method also requires an `index.html` file placed within the `dist` folder. You can also manually determine where your index file is served from by modifying where `webpack-dev-server` searches for files in the `webpack.config.js` file:
```javascript
module.exports = {
  ...
  devServer: {
    static: path.join(__dirname, 'dist'),
    public: 'localhost:8080',
  },
  ...
};
```
Note that the bundled files that `webpack-dev-server` creates live in memory, so you won't be able to find them on the file system.
```html
<script src="/YOUR_VIZ_NAME.bundle.js"></script>
```

Note that you can also inspect the bundle Javascript from the browser by navigating to `http://localhost:8080/YOUR_VIZ_NAME.bundle.js`.

### Developing in a Jupyter Notebook

An alternative to developing using `webpack-dev-server` is to bundle the Typescript and reference the Javascript output. You can spin up a notebook server with `jupyter notebook`, and bundle (while watching for live changes and updating accordingly) with `npx webpack --mode production --watch`. These processes must run simultaneously. This is especially useful for if you want to work with integrating Python code into your visualization. There's an example notebook `example.ipynb` that provides an example on how to do this.

**NOTE:** In order to access the bundled javascript, you need to include the full path to it: `cirq_ts/dist/YOUR_VIZ_NAME.bundle.js`. If you make any changes to the directory structure, take into account that the path may change as well.

### Developing in Google Colaboratory

We currently do not support developing visualizations in Google Colaboratory notebooks. However, visualization ran from the PyPI package are able to be viewed in Colab. 

### Formatting and linting

All Typescript files need to be formatted/linted according to [Google's public Typescript style guide](https://google.github.io/styleguide/tsguide.html). We use (Google's open source tool GTS)[https://github.com/google/gts] to handle this for you. Run `npm run fix` to handle fixing changes automatically, or refer to the `package.json` file for more options.

### Testing

We expect developed visualizations to be well tested. The Cirq typescript development environment requires two types of tests for any created visualization, unit testing and visualization testing. Unit testing ensures that the Typescript you wrote compiles correctly and generates the appropriate Three.js objects without breaking the rest of your code. Visualization testing actually compares the visualizations by building the visualization, taking a PNG screenshot, and comparing it to an expected PNG. 

We use [Mocha](https://mochajs.org) and [Chai](https://www.chaijs.com/) as our main testing tools. For comparing image diffs, we use [Pixelmatch](https://github.com/mapbox/pixelmatch) and [pngjs](https://github.com/lukeapage/pngjs).

#### Unit testing

Run unit tests using `npm run test`. We expect 100% code coverage for unit tests. You can check coverage by running `npm run coverage`.

Unit tests must live adjacent to their source file with the `_test.ts` suffix. So for the file `dir/MyFile.ts`, the corresponding testfile will be `dir/MyFile_test.ts`.

#### Visualization testing

We take the following steps for visualization testing in our development environment:
  1. We generate an generic HTML file with the specified visualization's current JS bundle
  2. We run the HTML file in a headless browser (Using [Puppeteer](https://github.com/puppeteer/puppeteer))
  3. We take a screenshot of the HTML output in the "browser" (Using Puppeteer)
  4. We compare the result of the screenshot with a pre-generated PNG file.

  The screenshot of the HTML browser output must live in a temporary directory; we use the [temp](https://github.com/bruce/node-temp) package to handle that for us. Reference the test at `e2e/bloch_sphere/test_bloch_sphere.ts` to see how to easily generate the screenshot in a temporary directory.

  The pre-generated PNG file is a screenshot of the developer's choice that represents what the visualization should look like. Each visualization is required to have at least one expected PNG screenshot. For more complex visualizations, multiple screenshots may be needed.
