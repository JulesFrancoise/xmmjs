import validateParameters from '../common/validation';
import { isBaseModel } from '../core/model_base_mixin';
import { GMMPredictor } from '../gmm';

const hmmParameterSpec = (states, transitionMode) => ({
  states: {
    required: true,
    check: { min: 1 },
  },
  gaussians: {
    required: true,
    check: { min: 1 },
  },
  regularization: {
    required: true,
    check: ({ absolute, relative }) =>
      (absolute && relative && absolute > 0 && relative > 0),
  },
  transitionMode: {
    required: true,
    check: ['ergodic', 'leftright'],
  },
  covarianceMode: {
    required: true,
    check: ['full', 'diagonal'],
  },
  prior: {
    required: true,
    check: m => transitionMode === 'leftright' || m.length === states,
  },
  transition: {
    required: true,
    check: m => (transitionMode === 'leftright' ?
      m.length === 2 * states :
      m.length === states),
  },
  xStates: {
    required: true,
    check: m => m.length === states,
  },
});


/**
 * HMM Base prototype
 * @type {Object}
 * @ignore
 */
const hmmPredictionPrototype = /** @lends withHMMPrediction */ {
  forwardInitialized: false,
  isHierarchical: false,

  /**
   * Setup the Model by allocating GMM predictors to each of the hidden states
   * @return {HMMBaseModel} the model
   * @private
   */
  setup() {
    this.params.xStates = this.params.xStates.map(s => GMMPredictor(s).reset());
    return this;
  },

  /**
   * Reset the prediction process
   * @return {HMMBaseModel} the model
   */
  reset() {
    this.likelihoodBuffer.clear();
    this.params.xStates.forEach((s) => { s.reset(); });
    return this;
  },

  /**
   * Compute the likelihood of an observation given the HMM's parameters
   * @param  {Array<Number>} observation Observation vector
   * @return {Number}
   */
  likelihood(observation) {
    const ct = (this.forwardInitialized) ?
      this.updateForwardAlgorithm(observation) :
      this.initializeForwardAlgorithm(observation);
    this.updateAlphaWindow();
    this.results.progress = 0.0;
    for (let i = this.windowMinindex; i < this.windowMaxindex; i += 1) {
      if (this.isHierarchical) {
        this.results.progress += (this.alphaH[0][i] + this.alphaH[1][i] + this.alphaH[2][i]) *
          (i / this.windowNormalizationConstant);
      } else {
        this.results.progress += (this.alpha[i] * i) /
          this.windowNormalizationConstant;
      }
    }
    this.results.progress /= this.params.states - 1;
    return 1 / ct;
  },

  /**
   * Update the state probabilities filtering window (for multiclass
   * hierarchical HMM I think...)
   * @private
   */
  updateAlphaWindow() {
    this.results.likeliestState = 0;
    // Get likeliest State
    let bestAlpha = this.isHierarchical ?
      (this.alphaH[0][0] + this.alphaH[1][0]) : this.alpha[0];
    for (let i = 1; i < this.params.states; i += 1) {
      if (this.isHierarchical) {
        if ((this.alphaH[0][i] + this.alphaH[1][i]) > bestAlpha) {
          bestAlpha = this.alphaH[0][i] + this.alphaH[1][i];
          this.results.likeliestState = i;
        }
      } else if (this.alpha[i] > bestAlpha) {
        bestAlpha = this.alpha[i];
        this.results.likeliestState = i;
      }
    }

    // Compute Window
    this.windowMinindex = this.results.likeliestState - Math.floor(this.params.states / 2);
    this.windowMaxindex = this.results.likeliestState + Math.floor(this.params.states / 2);
    this.windowMinindex = (this.windowMinindex >= 0) ? this.windowMinindex : 0;
    this.windowMaxindex = (this.windowMaxindex <= this.params.states) ?
      this.windowMaxindex : this.params.states;
    this.windowNormalizationConstant = 0.0;
    for (let i = this.windowMinindex; i < this.windowMaxindex; i += 1) {
      this.windowNormalizationConstant += this.isHierarchical ?
        (this.alphaH[0][i] + this.alphaH[1][i]) :
        this.alpha[i];
    }
  },
};

/**
 * Bimodal (regression) HMM Prototype
 * @type {Object}
 * @ignore
 */
const hmmBimodalPredictionPrototype = /** @lends withHMMPrediction */ {
  /**
   * Estimate the output values corresponding to the input observation, by
   * regression given the HMM's parameters. This method is called Hidden
   * Mixture Regression (GMR).
   *
   * @param  {Array<Number>} inputObservation Observation on the input modality
   * @return {Array<Number>} Output values (length = outputDimension)
   */
  regression(inputObservation) {
    this.results.outputValues = Array(this.params.outputDimension).fill(0);
    this.results.outputCovariance = Array(this.params.covarianceMode === 'full' ? this.params.outputDimension ** 2 : this.params.outputDimension).fill(0);

    if (this.params.regressionEstimator === 'likeliest') {
      this.params.xStates[this.results.likeliestState].predict(inputObservation);
      this.results.outputValues =
        this.params.xStates[this.results.likeliestState].results.outputValues;
      return this.results.outputValues;
    }

    const clipMinState = (this.params.regressionEstimator === 'full') ?
      0 : this.windowMinindex;
    const clipMaxState = (this.params.regressionEstimator === 'full') ?
      this.params.states : this.windowMaxindex;
    let normalizationConstant = (this.params.regressionEstimator === 'full') ?
      1 : this.windowNormalizationConstant;

    if (normalizationConstant <= 0.0) normalizationConstant = 1;

    // Compute Regression
    for (let i = clipMinState; i < clipMaxState; i += 1) {
      this.params.xStates[i].likelihood(inputObservation);
      this.params.xStates[i].regression(inputObservation);
      const tmpPredictedOutput = this.params.xStates[i].results.outputValues;
      for (let d = 0; d < this.params.outputDimension; d += 1) {
        if (this.isHierarchical) {
          this.results.outputValues[d] +=
            (this.alphaH[0][i] + this.alphaH[1][i]) *
            (tmpPredictedOutput[d] / normalizationConstant);
          if (this.params.covarianceMode === 'full') {
            for (let d2 = 0; d2 < this.params.outputDimension; d2 += 1) {
              this.results.outputCovariance[(d * this.params.outputDimension) + d2] +=
                (this.alphaH[0][i] + this.alphaH[1][i]) *
                (this.alphaH[0][i] + this.alphaH[1][i]) *
                (this.params.xStates[i].results
                  .outputCovariance[(d * this.params.outputDimension) + d2] /
                normalizationConstant);
            }
          } else {
            this.results.outputCovariance[d] +=
              (this.alphaH[0][i] + this.alphaH[1][i]) *
              (this.alphaH[0][i] + this.alphaH[1][i]) *
              (this.params.xStates[i].results.outputCovariance[d] /
              normalizationConstant);
          }
        } else {
          this.results.outputValues[d] += this.alpha[i] *
            (tmpPredictedOutput[d] / normalizationConstant);
          if (this.params.covarianceMode === 'full') {
            for (let d2 = 0; d2 < this.params.outputDimension; d2 += 1) {
              this.results.outputCovariance[(d * this.params.outputDimension) + d2] +=
                (this.alpha[i] ** 2) *
                (this.params.xStates[i].results
                  .outputCovariance[(d * this.params.outputDimension) + d2] /
                normalizationConstant);
            }
          } else {
            this.results.outputCovariance[d] +=
              ((this.alpha[i] ** 2) *
              this.params.xStates[i].results.outputCovariance[d]) /
              normalizationConstant;
          }
        }
      }
    }
    return this.results.outputValues;
  },
};

/**
 * Add HMM prediction capabilities to a single-class model. Mostly, this checks
 * the validity of the model parameters
 *
 * @todo validate gaussian components
 *
 * @param  {HMMBaseModel} o Source Model
 * @return {HMMBaseModel}
 *
 * @throws {Error} is o is not a ModelBase
 */
export default function withHMMPrediction(o) {
  if (!isBaseModel(o)) {
    throw new Error('The base object must include a standard set of parameters (`params` key), @see `ModelBase`.');
  }
  validateParameters('HMM', hmmParameterSpec(o.params.states, o.params.transitionMode), o.params);
  return Object.assign(
    o,
    hmmPredictionPrototype,
    o.params.bimodal ? hmmBimodalPredictionPrototype : {},
    {
      alpha: new Array(o.params.states).fill(0),
      previous_alpha_: new Array(o.params.states).fill(0),
    },
  ).setup();
}
