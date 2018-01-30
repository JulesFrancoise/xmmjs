import { isBaseModel } from './model_base_mixin';
import CircularBuffer from '../common/circular_buffer';

/**
 * Prototype for models with prediction capabilities
 * @param  {Boolean} bimodal Specifies whether the model is bimodal
 * @return {Object}
 * @ignore
 */
const predictionBasePrototype = bimodal => (/** @lends withAbtractPrediction */{
  /**
   * Likelihood Buffer
   * @type {CircularBuffer}
   * @private
   */
  likelihoodBuffer: CircularBuffer(1),

  /**
   * Likelihood Window (used to smooth the log-likelihoods over several frames)
   * @return {Number}
   */
  get likelihoodWindow() {
    return this.likelihoodBuffer.capacity;
  },

  /**
   * Likelihood Window (used to smooth the log-likelihoods over several frames)
   * @param {Number} [lw] Size (in frames) of the likelihood smoothing window
   */
  set likelihoodWindow(lw) {
    this.likelihoodBuffer = CircularBuffer(lw);
  },

  /**
   * Reset the prediction process
   */
  reset() {
    this.likelihoodBuffer.clear();
  },

  /**
   * Update the predictions with a new observation
   * @param  {Array<Number>} observation Observation vector
   * @return {Object} Prediction results
   *
   * @todo document results data structure
   */
  predict(observation) {
    const likelihood = this.likelihood(observation);
    if (bimodal) {
      this.regression(observation);
    }
    this.updateResults(likelihood);
    return this.results;
  },

  /**
   * Update the prediction results
   * @param  {Number} instantLikelihood Instantaneous likelihood
   * @private
   */
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

/**
 * Add ABSTRACT prediction capabilities to an existing model
 * @param  {Modelbase} o                 Source model
 * @param  {Number} [likelihoodWindow=1] Size of the likelihood smoothing window
 * @return {Modelbase}
 */
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
    { results, likelihoodBuffer: CircularBuffer(likelihoodWindow) },
  );
}
