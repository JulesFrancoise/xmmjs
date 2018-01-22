import test from 'ava';
import Phrase from '../src/core/training_set/phrase';

test('Create a phrase with default parameters', (t) => {
  const p = new Phrase();
  t.false(p.bimodal);
  t.is(p.dimension, 1);
  t.is(p.inputDimension, 1);
  t.is(p.outputDimension, 0);
  t.deepEqual(p.inputData, []);
  t.deepEqual(p.columnNames, ['']);
});

test('Push data to a unimodal phrase', (t) => {
  const data = Array.from(
    Array(100),
    () => [Math.random(), Math.random(), Math.random()],
  );
  const p = new Phrase({
    inputDimension: 3,
  });
  data.forEach((frame) => {
    p.push(frame);
  });
  t.deepEqual(data, p.inputData);
  data.forEach((frame, i) => {
    frame.forEach((x, d) => {
      t.is(x, p.get(i, d));
    });
  });
});

test('Push data to a bimodal phrase', (t) => {
  const data = Array.from(
    Array(2),
    () => [Math.random(), Math.random(), Math.random(), Math.random()],
  );
  const p = new Phrase({
    inputDimension: 2,
    outputDimension: 2,
    columnNames: ['x', 'y', 'a', 'b'],
  });
  t.true(p.bimodal);
  t.deepEqual(p.columnNames, ['x', 'y', 'a', 'b']);
  data.forEach((frame) => {
    p.push(frame);
  });
  t.deepEqual(data.map(x => x.slice(0, 2)), p.inputData);
  t.deepEqual(data.map(x => x.slice(2, 4)), p.outputData);
  data.forEach((frame, i) => {
    frame.forEach((x, d) => {
      t.is(x, p.get(i, d));
    });
  });
});

test('Mean, std and minmax are correctly computed', (t) => {
  const data = Array.from(
    Array(100000),
    () => [Math.random(), Math.random(), Math.random(), Math.random()],
  );
  const p = new Phrase({
    inputDimension: 2,
    outputDimension: 2,
  });
  data.forEach((frame) => {
    p.push(frame);
  });
  for (let i = 0; i < 4; i += 1) {
    t.true(Math.abs(p.mean()[i] - 0.5) < 0.01);
    t.true(Math.abs(p.standardDeviation()[i] - Math.sqrt(1 / 12)) < 0.01);
    const { min, max } = p.minmax()[i];
    t.true(Math.abs(min) < 1e-3);
    t.true(Math.abs(max - 1) < 1e-3);
  }
});
