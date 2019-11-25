import test from 'ava';
import { readFileSync } from 'fs';
import euclidean from '../src/common/euclidean';
import TrainingSet from '../src/training_set';
import {
  trainHMM,
  HMMPredictor,
  trainMulticlassHMM,
  MulticlassHMMPredictor,
} from '../src/hmm';

test('Ergodic HMR Training and decoding output constistent results', (t) => {
  const ts = TrainingSet({ inputDimension: 2, outputDimension: 2 });
  ts.push(0, 'default');
  for (let i = 0; i < 2000; i += 1) {
    const frame = [
      Math.random() - 0.5,
      0.5 + Math.random(),
      2.5 + Math.random(),
      4.5 + Math.random(),
    ];
    ts.getPhrase(0).push(frame);
  }
  const configuration = {
    states: 3,
    gaussians: 1,
    regularization: {
      absolute: 1e-1,
      relative: 1e-3,
    },
    transitionMode: 'ergodic',
    covarianceMode: 'full',
  };
  const hmrParams = trainHMM(ts, configuration);
  const predictor = HMMPredictor(hmrParams);
  predictor.reset();
  predictor.predict([0, 1]);
  const output = predictor.results.outputValues;
  t.true(Math.abs(output[0] - 3) < 0.1);
  t.true(Math.abs(output[1] - 5) < 0.1);
});

test('Left-right HMR Training and decoding output constistent results', (t) => {
  const ts = TrainingSet({ inputDimension: 2, outputDimension: 2 });
  ts.push(0, 'default');
  for (let i = 0; i < 2000; i += 1) {
    const frame = [
      Math.random() - 0.5,
      0.5 + Math.random(),
      2.5 + Math.random(),
      4.5 + Math.random(),
    ];
    ts.getPhrase(0).push(frame);
  }
  const configuration = {
    states: 3,
    gaussians: 1,
    regularization: {
      absolute: 1e-2,
      relative: 1e-3,
    },
    transitionMode: 'leftright',
    covarianceMode: 'full',
  };
  const hmrParams = trainHMM(ts, configuration);
  const predictor = HMMPredictor(hmrParams);
  predictor.reset();
  predictor.predict([0, 1]);
  const output = predictor.results.outputValues;
  t.true(Math.abs(output[0] - 3) < 0.5);
  t.true(Math.abs(output[1] - 5) < 0.5);
});

test('Multiclass ergodic HMR with actual data', (t) => {
  const ts = TrainingSet({ inputDimension: 4, outputDimension: 4 });
  ts.push(0);
  let input = readFileSync('./test/data/hmr_input_1.txt', 'utf8')
    .split('\n')
    .filter((l) => l !== '')
    .map((line) => line.split(' ').map((x) => parseFloat(x)));
  let output = readFileSync('./test/data/hmr_output_1.txt', 'utf8')
    .split('\n')
    .filter((l) => l !== '')
    .map((line) => line.split(' ').map((x) => parseFloat(x)));
  const len = Math.min(input.length, output.length);
  input = input.slice(0, len);
  output = output.slice(0, len);
  for (let i = 0; i < len; i += 1) {
    ts.getPhrase(0).push(input[i].concat(output[i]));
  }
  const configuration = {
    states: 12,
    gaussians: 1,
    regularization: {
      absolute: 1e-3,
      relative: 1e-10,
    },
    transitionMode: 'ergodic',
    covarianceMode: 'full',
  };
  const hmrParams = trainMulticlassHMM(ts, configuration);
  const predictor = MulticlassHMMPredictor(hmrParams);
  predictor.reset();
  const prediction = [];
  let predictionError = 0;
  input.forEach((frame, i) => {
    predictor.predict(frame);
    prediction.push(predictor.results.outputValues);
    predictionError += euclidean(output[i], predictor.results.outputValues);
  });
  predictionError /= input.length;
  t.true(predictionError < 0.1);
  // writeFileSync('./test/hmr_prediction.txt', prediction.join('\n'));
});

test('Multiclass Left-Right HMR with actual data', (t) => {
  const ts = TrainingSet({ inputDimension: 4, outputDimension: 4 });
  ts.push(0);
  let input = readFileSync('./test/data/hmr_input_1.txt', 'utf8')
    .split('\n')
    .filter((l) => l !== '')
    .map((line) => line.split(' ').map((x) => parseFloat(x)));
  let output = readFileSync('./test/data/hmr_output_1.txt', 'utf8')
    .split('\n')
    .filter((l) => l !== '')
    .map((line) => line.split(' ').map((x) => parseFloat(x)));
  const len = Math.min(input.length, output.length);
  input = input.slice(0, len);
  output = output.slice(0, len);
  for (let i = 0; i < len; i += 1) {
    ts.getPhrase(0).push(input[i].concat(output[i]));
  }
  const configuration = {
    states: 12,
    gaussians: 1,
    regularization: {
      absolute: 1e-3,
      relative: 1e-10,
    },
    transitionMode: 'leftright',
    covarianceMode: 'full',
  };
  const hmrParams = trainMulticlassHMM(ts, configuration);
  const predictor = MulticlassHMMPredictor(hmrParams);
  predictor.reset();
  const prediction = [];
  let predictionError = 0;
  input.forEach((frame, i) => {
    predictor.predict(frame);
    prediction.push(predictor.results.outputValues);
    predictionError += euclidean(output[i], predictor.results.outputValues);
  });
  predictionError /= input.length;
  t.true(predictionError < 0.1);
  // writeFileSync('./test/hmr_prediction.txt', prediction.join('\n'));
});
