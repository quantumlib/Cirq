## Cirq Typescript Development

*This directory contains the code and instructions for visualization tools in a web browser or Colab/Juptyer notebooks. We do this using Typescript. While necessary, this is only half of the code needed to run the `cirq-web` project. For information on how to integrate projects here with Python and the wider Cirq package, see the `cirq_web` directory.*

Visualizations run on [NodeJS](https://nodejs.org/en/), and we use [npm](https://www.npmjs.com/) for package management. To start developing, clone the repository and run `npm install` within this directory to install the necessary packages and begin development. You will need to install Node and npm if you haven't already.

For developing 3D visualizations, we rely on the [three.js](https://threejs.org/) framework. Understanding their documentation is critical to understanding 3D visualizations in Cirq. 

For bundling the Typescript into Javascript that can be run in the browser, and for overall ease of development, we use [Webpack](https://webpack.js.org/).

### Visualization build structure

The reference example for the build structure of a visualization is the Bloch sphere. Reference the `src/bloch_sphere/main.ts` file and the `CirqBlochSphere.class.ts` file to see the code. The `src/bloch_sphere/` directory should serve as a guide for how Typescript visualizations in Cirq should be structured. Visualizations should have:
 - A "root" folder within the `src/` directory labeled according to the title of the visualization. All files and directories for a particular visualization will live here. In the case of the Bloch Sphere, this is `bloch_sphere/`.
 - A `components/` directory which contains classes representing different components of the larger visualization, following typical object oriented programming techniques. In the case of the Bloch sphere, you can see that we have different classes for `Axes`, `Meridians`, `Text` etc.
 - Any `assets/` directory with information necessary for the visualization (fonts, images, etc.). In the case of the Bloch Sphere, we can see a `fonts/` subdirectory which holds necessary font data, within the `assets/` directory, but for instances where there isn't a lot of extra information necessary subdirectories may not be needed.
 - A class within the visualization's "root" folder which brings the individual components of the visualization together. In the case of the Bloch Sphere, this is `BlochSphere.class.ts`.
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

Following this structure, you will be able to bundle your visualization by running the command `npx webpack --mode production`. This will build the bundled Javascript file(s) within the `dist/` directory, where you can access and reference them in HTML.

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
Note that the bundled files that `webpack-dev-server` creates live in memory, so you won't be able to see them. However, you can access them like so:
```html
<script src="YOUR_VIZ_NAME.bundle.js"></script>
```

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

#### Visualization testing
To summarize the visualization testing, we generate an generic HTML file with the specified visualization's current JS bundle, run it in a headless browser, take a screenshot, and then compare the result with a pre-generated PNG file.

Our visualization tests live in the `e2e/` (end-to_end) directory. Each visualiztion should have at least one expected PNG screenshot.


### CI - delete later
  Add something that makes jobs fast when there are no changes made so it's not rerunning every time.
