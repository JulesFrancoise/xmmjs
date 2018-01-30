import test from 'ava';
import TrainingSet from '../src/training_set';

test('Create a training set with default parameters', (t) => {
  const ts = TrainingSet();
  t.is(ts.bimodal, false);
  t.is(ts.inputDimension, 1);
  t.is(ts.outputDimension, 0);
  t.is(ts.dimension, 1);
  t.deepEqual(ts.columnNames, ['']);
  ts.push(0, 'test');
  t.true(Object.keys(ts.phrases).includes('0'));
  const p = ts.phrases[0];
  t.is(p.bimodal, false);
  t.is(p.inputDimension, 1);
  t.is(p.outputDimension, 0);
  t.is(p.dimension, 1);
  t.is(p.length, 0);
  t.is(p.label, 'test');
  t.deepEqual(p.inputData, []);
  t.deepEqual(p.columnNames, ['']);
});

test('Create a unimodal training set', (t) => {
  const ts = TrainingSet({
    inputDimension: 2,
    outputDimension: 3,
    columnNames: ['x', 'y', 'z', 'a', 'b'],
  });
  t.true(ts.bimodal);
  t.is(ts.inputDimension, 2);
  t.is(ts.outputDimension, 3);
  t.is(ts.dimension, 5);
  t.deepEqual(ts.columnNames, ['x', 'y', 'z', 'a', 'b']);
  ts.push(12, 'test');
  t.true(Object.keys(ts.phrases).includes('12'));
  const p = ts.phrases[12];
  t.true(p.bimodal);
  t.is(p.inputDimension, 2);
  t.is(p.outputDimension, 3);
  t.is(p.dimension, 5);
  t.is(p.length, 0);
  t.is(p.label, 'test');
  t.deepEqual(p.inputData, []);
  t.deepEqual(p.outputData, []);
  t.deepEqual(p.columnNames, ['x', 'y', 'z', 'a', 'b']);
});
