const path = require('path');

// devServer holds files in memory: https://webpack.js.org/guides/development/
// Need compiled files to work w/ notebook
// Nbextension needed to work with Colab?

module.exports = {
  // entry: {
  //     index: './src/index.ts',
  //     scene: './src/blank_scene.ts',
  //     circle: './src/circle.ts',
  // },
  entry: './src/circle.ts',
  devServer: {
    contentBase: './dist',
  },
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: 'ts-loader',
        exclude: /node_modules/,
      },
    ],
  },
  resolve: {
    extensions: ['.tsx', '.ts', '.js'],
  },
  output: {
    filename: 'bundle.js',
    library: 'createSphere',
    path: path.resolve(__dirname, 'dist'),
  },
  externals: {
    three: 'three',
  }
};
