const path = require('path');
const webpack = require('webpack');
const UglifyJsPlugin = require('uglifyjs-webpack-plugin');

module.exports = {
  entry: './src/index.js',
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'mars.js',
    libraryTarget: 'umd',
    library: 'mars',
  },
  externals: {
    '@most/core': {
      commonjs: '@most/core',
      commonjs2: '@most/core',
      amd: '@most/core',
      root: '@most/core',
    },
    '@most/scheduler': {
      commonjs: '@most/scheduler',
      commonjs2: '@most/scheduler',
      amd: '@most/scheduler',
      root: '@most/scheduler',
    },
    '@most/dom-event': {
      commonjs: '@most/dom-event',
      commonjs2: '@most/dom-event',
      amd: '@most/dom-event',
      root: '@most/dom-event',
    },
    '@most/dom-prelude': {
      commonjs: '@most/dom-prelude',
      commonjs2: '@most/dom-prelude',
      amd: '@most/dom-prelude',
      root: '@most/dom-prelude',
    },
    '@most/dom-disposable': {
      commonjs: '@most/dom-disposable',
      commonjs2: '@most/dom-disposable',
      amd: '@most/dom-disposable',
      root: '@most/dom-disposable',
    },
    colormap: {
      commonjs: 'colormap',
      commonjs2: 'colormap',
      amd: 'colormap',
      root: 'colormap',
    },
    'xebra.js': {
      commonjs: 'xebra.js',
      commonjs2: 'xebra.js',
      amd: 'xebra.js',
      root: 'xebra.js',
    },
    myo: {
      commonjs: 'myo',
      commonjs2: 'myo',
      amd: 'myo',
      root: 'myo',
    },
    vue: {
      commonjs: 'vue',
      commonjs2: 'vue',
      amd: 'vue',
      root: 'vue',
    },
    tonal: {
      commonjs: 'tonal',
      commonjs2: 'tonal',
      amd: 'tonal',
      root: 'tonal',
    },
  },
  module: {
    rules: [
      {
        test: /\.css$/,
        use: ['vue-style-loader', 'css-loader'],
      },
      {
        test: /\.js$/,
        exclude: /node_modules/,
        loader: 'babel-loader',
      },
      {
        test: /\.vue$/,
        loader: 'vue-loader',
      },
    ],
  },
  resolve: {
    alias: {
      vue$: 'vue/dist/vue.esm.js',
    },
    extensions: ['*', '.js', '.vue', '.json'],
  },
  devServer: {
    historyApiFallback: true,
    noInfo: true,
    overlay: true,
  },
  performance: {
    hints: false,
  },
  devtool: '#eval-source-map',
};

if (process.env.NODE_ENV === 'production') {
  module.exports.devtool = '#source-map';
  // http://vue-loader.vuejs.org/en/workflow/production.html
  module.exports.plugins = (module.exports.plugins || []).concat([
    new webpack.DefinePlugin({
      'process.env': {
        NODE_ENV: '"production"',
      },
    }),
    new UglifyJsPlugin({
      sourceMap: true,
    }),
    new webpack.LoaderOptionsPlugin({
      minimize: true,
    }),
  ]);
}
