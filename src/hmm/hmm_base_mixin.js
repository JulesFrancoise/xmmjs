import { isBaseModel } from '../core/model_base_mixin';

//
// TODO: hierarchical + exit probabilities methods.
//

/**
 * HMM Base prototype
 * @type {Object}
 * @ignore
 */
const hmmBasePrototype = /** @lends withHMMBase */ {
  /**
   * Specifies if the forward algorithm has been initialized
   * @type {Boolean}
   * @private
   */
  forwardInitialized: false,

  /**
   * Specifies if the containing multiclass model is isHierarchical
   * @todo check that
   * @type {Boolean}
   * @private
   */
  isHierarchical: false,

  /**
   * Initialize the forward algorithm (See rabiner, 1989)
   * @param  {Array<Number>} observation Observation vector
   * @return {Number}                    `ct` (inverse likelihood)
   */
  initializeForwardAlgorithm(observation) {
    let normConst = 0;
    if (this.params.transitionMode === 'ergodic') {
      for (let i = 0; i < this.params.states; i += 1) {
        this.alpha[i] = this.params.prior[i] *
          this.params.xStates[i].likelihood(observation);
        normConst += this.alpha[i];
      }
    } else {
      this.alpha = new Array(this.params.states).fill(0);
      this.alpha[0] = this.params.xStates[0].likelihood(observation);
      normConst += this.alpha[0];
    }
    this.forwardInitialized = true;
    if (normConst > 0) {
      for (let i = 0; i < this.params.states; i += 1) {
        this.alpha[i] /= normConst;
      }
      return 1 / normConst;
    }
    for (let j = 0; j < this.params.states; j += 1) {
      this.alpha[j] = 1 / this.params.states;
    }
    return 1;
  },

  /**
   * Update the forward algorithm (See rabiner, 1989)
   * @param  {Array<Number>} observation Observation vector
   * @return {Number}                    `ct` (inverse likelihood)
   */
  updateForwardAlgorithm(observation) {
    let normConst = 0;
    this.previousAlpha = this.alpha.slice();
    for (let j = 0; j < this.params.states; j += 1) {
      this.alpha[j] = 0;
      if (this.params.transitionMode === 'ergodic') {
        for (let i = 0; i < this.params.states; i += 1) {
          this.alpha[j] += this.previousAlpha[i] *
            this.params.transition[i][j];
        }
      } else {
        this.alpha[j] += this.previousAlpha[j] * this.params.transition[j * 2];
        if (j > 0) {
          this.alpha[j] += this.previousAlpha[j - 1] *
            this.params.transition[((j - 1) * 2) + 1];
        } else {
          this.alpha[0] += this.previousAlpha[this.params.states - 1] *
            this.params.transition[(this.params.states * 2) - 1];
        }
      }
      this.alpha[j] *= this.params.xStates[j].likelihood(observation);
      normConst += this.alpha[j];
    }
    if (normConst > 1e-300) {
      for (let j = 0; j < this.params.states; j += 1) {
        this.alpha[j] /= normConst;
      }
      return 1 / normConst;
    }
    return 0;
  },
};

/**
 * Add basic HMM capabilities to a single-class model. This enables the
 * computation of the likelihoods and regression operations common to
 * training and prediction
 *
 * @see withHMMTraining
 * @see withHMMPrediction
 *
 * @param  {ModelBase} o Source Model
 * @return {HMMBaseModel}
 *
 * @throws {Error} is o is not a ModelBase
 */
export default function withHMMBase(o) {
  if (!isBaseModel(o)) {
    throw new Error('The base object must include a standard set of parameters (`params` key), @see `ModelBase`.');
  }
  return Object.assign(o, hmmBasePrototype);
}
