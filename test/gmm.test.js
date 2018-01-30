import test from 'ava';
import { readFileSync } from 'fs';
import TrainingSet from '../src/training_set';
import { trainGMM, trainMulticlassGMM } from '../src/train';
import { GMMPredictor, multiclassGMMPredictor } from '../src/predict';

test('GMM Training and decoding output ~constistent~ results', (t) => {
  const ts = TrainingSet({ inputDimension: 3 });
  ts.push(0, 'default');
  for (let i = 0; i < 20000; i += 1) {
    const frame = Array.from(Array(3), () => 0);
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
  const gmmParams = trainGMM(ts, configuration);
  const predictor = GMMPredictor(gmmParams);
  predictor.reset();
  predictor.predict([0, 0, 0]);
  const lik1 = predictor.results.logLikelihood;
  predictor.predict([0.1, 0.1, 0.1]);
  const lik2 = predictor.results.logLikelihood;
  predictor.predict([0.5, 0.5, 0.5]);
  const lik3 = predictor.results.logLikelihood;
  predictor.predict([5, 5, 5]);
  const lik4 = predictor.results.logLikelihood;
  t.true(lik1 > lik2);
  t.true(lik2 > lik3);
  t.true(lik3 > lik4);
});

test('Multiclass GMM Training and decoding output ~constistent~ results', (t) => {
  const ts = TrainingSet({ inputDimension: 3 });
  ts.push(0, 'un');
  for (let i = 0; i < 2000; i += 1) {
    const frame = Array.from(Array(3), () => 0);
    // const frame = Array.from(Array(3), () => Math.random());
    ts.getPhrase(0).push(frame);
  }
  ts.push(1, 'deux');
  for (let i = 0; i < 2000; i += 1) {
    const frame = Array.from(Array(3), () => 1);
    // const frame = Array.from(Array(3), () => Math.random());
    ts.getPhrase(1).push(frame);
  }
  const configuration = {
    gaussians: 12,
    regularization: {
      absolute: 1e-1,
      relative: 1e-10,
    },
    covarianceMode: 'full',
  };
  const gmmParams = trainMulticlassGMM(ts, configuration);
  const predictor = multiclassGMMPredictor(gmmParams);
  predictor.reset();
  predictor.predict([0, 0, 0]);
  t.is(predictor.results.likeliest, 'un');
  predictor.predict([0.1, 0.1, 0.1]);
  t.is(predictor.results.likeliest, 'un');
  predictor.predict([0.7, 0.7, 0.7]);
  t.is(predictor.results.likeliest, 'deux');
  predictor.predict([1, 1, 1]);
  t.is(predictor.results.likeliest, 'deux');
});

test('Multiclass GMM with actual data', (t) => {
  const ts = TrainingSet({ inputDimension: 6 });
  const phrases = ['un', 'deux', 'trois'].map((label, i) => {
    ts.push(i, label);
    const input = readFileSync(`./test/data/gmm_phrase_${i + 1}.txt`, 'utf8')
      .split('\n')
      .filter(l => l !== '')
      .map(line => line.split(' ').map(x => parseFloat(x)));
    input.forEach((frame) => {
      ts.getPhrase(i).push(frame);
    });
    return input;
  });
  const configuration = {
    gaussians: 12,
    regularization: {
      absolute: 1e-1,
      relative: 1e-10,
    },
    covarianceMode: 'full',
  };
  const gmmParams = trainMulticlassGMM(ts, configuration);
  const predictor = multiclassGMMPredictor(gmmParams);
  predictor.reset();
  predictor.likelihoodWindow = phrases[0].length;
  phrases[0].forEach((frame) => {
    predictor.predict(frame);
  });
  t.is(predictor.results.likeliest, 'un');
  let r1 = predictor.results.classes.un;
  let r2 = predictor.results.classes.deux;
  let r3 = predictor.results.classes.trois;
  t.true(r1.instantLikelihood > r2.instantLikelihood);
  t.true(r1.instantLikelihood > r3.instantLikelihood);
  t.true(r1.logLikelihood > r2.logLikelihood);
  t.true(r1.logLikelihood > r3.logLikelihood);
  predictor.likelihoodWindow = phrases[1].length;
  phrases[1].forEach((frame) => {
    predictor.predict(frame);
  });
  t.is(predictor.results.likeliest, 'deux');
  r1 = predictor.results.classes.un;
  r2 = predictor.results.classes.deux;
  r3 = predictor.results.classes.trois;
  t.true(r2.instantLikelihood > r1.instantLikelihood);
  t.true(r2.instantLikelihood > r3.instantLikelihood);
  t.true(r2.logLikelihood > r1.logLikelihood);
  t.true(r2.logLikelihood > r3.logLikelihood);
});
