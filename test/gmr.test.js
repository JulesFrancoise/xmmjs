import test from 'ava';
import { readFileSync, writeFileSync } from 'fs';
import TrainingSet from '../src/training_set';
import { trainGMM, trainMulticlassGMM } from '../src/train';
import { GMMPredictor, multiclassGMMPredictor } from '../src/predict';

test('GMR Training and decoding output ~constistent~ results', (t) => {
  const ts = TrainingSet({ inputDimension: 2, outputDimension: 2 });
  ts.push(0, 'default');
  for (let i = 0; i < 20000; i += 1) {
    const frame = [
      Math.random() - 0.5,
      0.5 + Math.random(),
      2.5 + Math.random(),
      4.5 + Math.random(),
    ];
    ts.getPhrase(0).push(frame);
  }
  const configuration = {
    gaussians: 3,
    regularization: {
      absolute: 1e-1,
      relative: 1e-10,
    },
    covarianceMode: 'full',
  };
  const gmrParams = trainGMM(ts, configuration);
  const predictor = GMMPredictor(gmrParams);
  predictor.reset();
  predictor.predict([0, 1]);
  const output = predictor.results.outputValues;
  t.true(Math.abs(output[0] - 3) < 0.1);
  t.true(Math.abs(output[1] - 5) < 0.1);
});

test('GMR with actual data', (t) => {
  const ts = TrainingSet({ inputDimension: 1, outputDimension: 1 });
  ts.push(0);
  const input = readFileSync('./test/data/gmr_input.txt', 'utf8')
    .split('\n')
    .filter(l => l !== '')
    .map(line => line.split(' ').map(x => parseFloat(x)));
  const output = readFileSync('./test/data/gmr_output.txt', 'utf8')
    .split('\n')
    .filter(l => l !== '')
    .map(line => line.split(' ').map(x => parseFloat(x)));
  input.forEach((frame, i) => {
    ts.getPhrase(0).push(frame.concat(output[i]));
  });
  const configuration = {
    gaussians: 24,
    regularization: {
      absolute: 1e-3,
      relative: 1e-10,
    },
    covarianceMode: 'full',
  };
  const gmrParams = trainMulticlassGMM(ts, configuration);
  const predictor = multiclassGMMPredictor(gmrParams);
  predictor.reset();
  const prediction = [];
  let predictionError = 0;
  input.forEach((frame, i) => {
    predictor.predict(frame);
    prediction.push(predictor.results.outputValues[0]);
    predictionError += Math.abs(output[i] - predictor.results.outputValues[0]);
  });
  predictionError /= input.length;
  t.true(predictionError < 0.02);
  writeFileSync('./test/gmr_prediction.txt', prediction.join('\n'));
});
