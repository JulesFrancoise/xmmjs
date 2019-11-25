import test from 'ava';
import { readFileSync } from 'fs';
import TrainingSet from '../src/training_set';
import {
  trainMulticlassHMM,
  HierarchicalHMMPredictor,
} from '../src/hmm';

test('Hierarchical Ergodic HMM Training and decoding output constistent results', (t) => {
  const ts = TrainingSet({ inputDimension: 3 });
  ts.push(0, 'un');
  for (let i = 0; i < 500; i += 1) {
    const frame = Array.from(Array(3), () => 0);
    // const frame = Array.from(Array(3), () => Math.random());
    ts.getPhrase(0).push(frame);
  }
  ts.push(1, 'deux');
  for (let i = 0; i < 500; i += 1) {
    const frame = Array.from(Array(3), () => 1);
    // const frame = Array.from(Array(3), () => Math.random());
    ts.getPhrase(1).push(frame);
  }
  const configuration = {
    states: 4,
    gaussians: 1,
    regularization: {
      absolute: 1e-1,
      relative: 1e-10,
    },
    transitionMode: 'ergodic',
    covarianceMode: 'full',
  };
  const hmmParams = trainMulticlassHMM(ts, configuration);
  const predictor = HierarchicalHMMPredictor(hmmParams);
  predictor.reset();
  for (let i = 0; i < 20; i += 1) {
    predictor.predict([0, 0, 0]);
  }
  t.is(predictor.results.likeliest, 'un');
  predictor.reset();
  for (let i = 0; i < 20; i += 1) {
    predictor.predict([0.1, 0.1, 0.1]);
  }
  t.is(predictor.results.likeliest, 'un');
  predictor.reset();
  for (let i = 0; i < 20; i += 1) {
    predictor.predict([0.7, 0.7, 0.7]);
  }
  t.is(predictor.results.likeliest, 'deux');
  predictor.reset();
  for (let i = 0; i < 20; i += 1) {
    predictor.predict([1, 1, 1]);
  }
  t.is(predictor.results.likeliest, 'deux');
});

test('Hierarchical Left-right HMM Training and decoding output constistent results', (t) => {
  const ts = TrainingSet({ inputDimension: 3 });
  ts.push(0, 'un');
  for (let i = 0; i < 500; i += 1) {
    const frame = Array.from(Array(3), () => 0);
    // const frame = Array.from(Array(3), () => Math.random());
    ts.getPhrase(0).push(frame);
  }
  ts.push(1, 'deux');
  for (let i = 0; i < 500; i += 1) {
    const frame = Array.from(Array(3), () => 1);
    // const frame = Array.from(Array(3), () => Math.random());
    ts.getPhrase(1).push(frame);
  }
  const configuration = {
    states: 4,
    gaussians: 1,
    regularization: {
      absolute: 1e-1,
      relative: 1e-10,
    },
    transitionMode: 'leftright',
    covarianceMode: 'full',
  };
  const hmmParams = trainMulticlassHMM(ts, configuration);
  const predictor = HierarchicalHMMPredictor(hmmParams);
  predictor.reset();
  for (let i = 0; i < 20; i += 1) {
    predictor.predict([0, 0, 0]);
  }
  t.is(predictor.results.likeliest, 'un');
  predictor.reset();
  for (let i = 0; i < 20; i += 1) {
    predictor.predict([0.1, 0.1, 0.1]);
  }
  t.is(predictor.results.likeliest, 'un');
  predictor.reset();
  for (let i = 0; i < 20; i += 1) {
    predictor.predict([0.7, 0.7, 0.7]);
  }
  t.is(predictor.results.likeliest, 'deux');
  predictor.reset();
  for (let i = 0; i < 20; i += 1) {
    predictor.predict([1, 1, 1]);
  }
  t.is(predictor.results.likeliest, 'deux');
});

test('Hierarchical Ergodic HMM with actual data', (t) => {
  const ts = TrainingSet({ inputDimension: 4 });
  const phrases = ['un', 'deux', 'trois'].map((label, i) => {
    ts.push(i, label);
    const input = readFileSync(`./test/data/hmm_gesture_${i + 1}.txt`, 'utf8')
      .split('\n')
      .filter((l) => l !== '')
      .map((line) => line.split(' ').map((x) => parseFloat(x)));
    input.forEach((frame) => {
      ts.getPhrase(i).push(frame);
    });
    return input;
  });
  const configuration = {
    states: 4,
    gaussians: 1,
    regularization: {
      absolute: 1e-1,
      relative: 1e-10,
    },
    transitionMode: 'ergodic',
    covarianceMode: 'full',
  };
  const hmmParams = trainMulticlassHMM(ts, configuration);
  const predictor = HierarchicalHMMPredictor(hmmParams);
  predictor.reset();
  predictor.setLikelihoodWindow(phrases[0].length);
  phrases[0].forEach((frame) => {
    predictor.predict(frame);
  });
  t.is(predictor.results.likeliest, 'un');
  let r1 = predictor.results.classes.un;
  let r2 = predictor.results.classes.deux;
  let r3 = predictor.results.classes.trois;
  t.true(r1.logLikelihood > r2.logLikelihood);
  t.true(r1.logLikelihood > r3.logLikelihood);
  predictor.reset();
  predictor.setLikelihoodWindow(phrases[1].length);
  phrases[1].forEach((frame) => {
    predictor.predict(frame);
  });
  t.is(predictor.results.likeliest, 'deux');
  r1 = predictor.results.classes.un;
  r2 = predictor.results.classes.deux;
  r3 = predictor.results.classes.trois;
  t.true(r2.logLikelihood > r1.logLikelihood);
  t.true(r2.logLikelihood > r3.logLikelihood);
});

test('Hierarchical Left-right HMM with actual data', (t) => {
  const ts = TrainingSet({ inputDimension: 4 });
  const phrases = ['un', 'deux', 'trois'].map((label, i) => {
    ts.push(i, label);
    const input = readFileSync(`./test/data/hmm_gesture_${i + 1}.txt`, 'utf8')
      .split('\n')
      .filter((l) => l !== '')
      .map((line) => line.split(' ').map((x) => parseFloat(x)));
    input.forEach((frame) => {
      ts.getPhrase(i).push(frame);
    });
    return input;
  });
  const configuration = {
    states: 4,
    gaussians: 1,
    regularization: {
      absolute: 1e-1,
      relative: 1e-10,
    },
    transitionMode: 'leftright',
    covarianceMode: 'full',
  };
  const hmmParams = trainMulticlassHMM(ts, configuration);
  const predictor = HierarchicalHMMPredictor(hmmParams);
  predictor.reset();
  predictor.setLikelihoodWindow(phrases[0].length);
  phrases[0].forEach((frame) => {
    predictor.predict(frame);
  });
  t.is(predictor.results.likeliest, 'un');
  let r1 = predictor.results.classes.un;
  let r2 = predictor.results.classes.deux;
  let r3 = predictor.results.classes.trois;
  t.true(r1.logLikelihood > r2.logLikelihood);
  t.true(r1.logLikelihood > r3.logLikelihood);
  predictor.reset();
  predictor.setLikelihoodWindow(phrases[1].length);
  phrases[1].forEach((frame) => {
    predictor.predict(frame);
  });
  t.is(predictor.results.likeliest, 'deux');
  r1 = predictor.results.classes.un;
  r2 = predictor.results.classes.deux;
  r3 = predictor.results.classes.trois;
  t.true(r2.logLikelihood > r1.logLikelihood);
  t.true(r2.logLikelihood > r3.logLikelihood);
  predictor.reset();
  predictor.setLikelihoodWindow(phrases[2].length);
  phrases[2].forEach((frame) => {
    predictor.predict(frame);
  });
  t.is(predictor.results.likeliest, 'trois');
  r1 = predictor.results.classes.un;
  r2 = predictor.results.classes.deux;
  r3 = predictor.results.classes.trois;
  t.true(r3.logLikelihood > r1.logLikelihood);
  t.true(r3.logLikelihood > r2.logLikelihood);
});
