import { isBaseModel } from './model_base_mixin';
import CircularBuffer from '../common/circular_buffer';

const predictionBasePrototype = bimodal => ({
  likelihoodBuffer: new CircularBuffer(1),

  get likelihoodWindow() {
    return this.likelihoodBuffer.capacity;
  },

  set likelihoodWindow(lw) {
    this.likelihoodBuffer = new CircularBuffer(lw);
  },

  reset() {
    this.likelihoodBuffer.clear();
  },

  predict(observation) {
    const likelihood = this.likelihood(observation);
    if (bimodal) {
      this.regression(observation);
    }
    this.updateResults(likelihood);
    return this.results;
  },

  updateResults(instantLikelihood) {
    this.results.instantLikelihood = instantLikelihood;
    this.likelihoodBuffer.push(Math.log(instantLikelihood));
    this.results.logLikelihood = 0;
    const bufSize = this.likelihoodBuffer.length;
    for (let i = 0; i < bufSize; i += 1) {
      this.results.logLikelihood += this.likelihoodBuffer.get(i);
    }
    this.results.logLikelihood /= bufSize;
  },
});

export default function withAbtractPrediction(o, likelihoodWindow = 1) {
  if (!isBaseModel(o)) {
    throw new Error('The base object must include a standard set of parameters (`params` key), @see `ModelBase`.');
  }
  const results = Object.assign(
    { instantLikelihood: 0, logLikelihood: 0 },
    o.params.bimodal ? { outputValues: [], outputCovariance: [] } : {},
  );
  return Object.assign(
    o,
    predictionBasePrototype(o.params.bimodal),
    { results, likelihoodBuffer: new CircularBuffer(likelihoodWindow) },
  );
}
