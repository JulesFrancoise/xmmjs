import test from 'ava';
import { readFileSync } from 'fs';
import euclidean from '../src/common/euclidean';
import TrainingSet from '../src/training_set';
import {
  trainMulticlassHMM,
  HierarchicalHMMPredictor,
} from '../src/hmm';

test('Hierarchical ergodic HMR with actual data', (t) => {
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
  const predictor = HierarchicalHMMPredictor(hmrParams);
  predictor.reset();
  const prediction = [];
  let predictionError = 0;
  input.forEach((frame, i) => {
    predictor.predict(frame);
    prediction.push(predictor.results.outputValues);
    predictionError += euclidean(output[i], predictor.results.outputValues);
  });
  predictionError /= input.length;
  // console.log('predictionError', predictionError);
  t.true(predictionError < 0.1);
  // writeFileSync('./test/hhmr_prediction.txt', prediction.join('\n'));
});

test('Hierarchical Left-Right HMR with actual data', (t) => {
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
  const predictor = HierarchicalHMMPredictor(hmrParams);
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
  // writeFileSync('./test/hhmr_prediction.txt', prediction.join('\n'));
});
