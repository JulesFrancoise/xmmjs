import test from 'ava';
import TrainingSet from '../src/core/training_set';
import train from '../src/core/model/single_class_gmm_trainer';
import Decoder from '../src/core/model/single_class_gmm_decoder';

test('GMM Training and decoding output ~constistent~ results', (t) => {
  const ts = new TrainingSet({ inputDimension: 3 });
  ts.push(0, 'default');
  for (let i = 0; i < 200; i += 1) {
    const frame = Array.from(Array(3), () => 0);
    ts.getPhrase(0).push(frame);
  }
  const gmm = train(ts, {
    gaussians: 1,
    regularization: {
      absolute: 1e-1,
      relative: 1e-3,
    },
    covarianceMode: 'full',
  });
  const decoder = new Decoder(gmm);
  decoder.reset();
  decoder.predict([0, 0, 0]);
  const lik1 = decoder.results.logLikelihood;
  decoder.predict([0.1, 0.1, 0.1]);
  const lik2 = decoder.results.logLikelihood;
  decoder.predict([0.5, 0.5, 0.5]);
  const lik3 = decoder.results.logLikelihood;
  decoder.predict([5, 5, 5]);
  const lik4 = decoder.results.logLikelihood;
  t.true(lik1 > lik2);
  t.true(lik2 > lik3);
  t.true(lik3 > lik4);
});
