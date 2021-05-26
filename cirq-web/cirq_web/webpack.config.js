const path = require('path');

// devServer holds files in memory: https://webpack.js.org/guides/development/
// Need compiled files to work w/ notebook
// Nbextension needed to work with Colab?

module.exports = {
  entry: './src/sphere.ts',
  devServer: {
    publicPath: '/',
    contentBase: './dist', // Serves static files from the /dist directory
    //TODO: investigate dev-server version based on https://github.com/webpack/webpack-dev-server/issues/2484
    injectClient: false,
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
    library: {
      name: 'createSphere',
      type: 'var',
    },
    path: path.resolve(__dirname, 'dist'),
    publicPath: './dist',
  },
};
