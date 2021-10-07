const path = require('path');

module.exports = {
  entry: {
    bloch_sphere: './src/bloch_sphere/main.ts',
    circuit: './src/circuit/main.ts',
  },
  devServer: {
    static: path.join(__dirname, 'dist'),
    public: 'localhost:8080',
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
    filename: '[name].bundle.js',
    library: {
      type: 'global',
    },
    path: path.resolve(__dirname, 'dist'),
    publicPath: '/',
  },
};
