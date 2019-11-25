import babel from 'rollup-plugin-babel';
import resolve from 'rollup-plugin-node-resolve';
// import { plugin as analyze } from 'rollup-plugin-analyzer';
import { terser } from 'rollup-plugin-terser';
import filesize from 'rollup-plugin-filesize';
import pkg from './package.json';

let plugins = [
  resolve(),
  babel({
    exclude: 'node_modules/**',
  }),
];
if (process.env.NODE_ENV === 'production') {
  plugins = plugins.concat([
    terser(),
    // analyze(),
    filesize(),
  ]);
}

export default {
  input: 'src/index.js',
  plugins,
  output: [
    {
      file: pkg.main,
      format: 'umd',
      name: 'xmm',
      sourcemap: true,
    },
    {
      file: pkg.module,
      format: 'es',
      sourcemap: true,
    },
  ],
};
