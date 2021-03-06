import { isBaseModel } from '../core/model_base_mixin';
import GaussianDistribution from '../common/gaussian_distribution';

/**
 * GMM Base prototype
 * @type {Object}
 * @ignore
 */
const gmmBasePrototype = /** @lends withGMMBase */ {
  /**
   * Allocate the training variables
   * @private
   */
  allocate() {
    this.params.components = Array.from(
      Array(this.params.gaussians),
      () => new GaussianDistribution(
        this.params.inputDimension,
        this.params.outputDimension,
        this.params.covarianceMode,
      ),
    );
    this.params.mixtureCoeffs = Array(this.params.gaussians).fill(0);
    this.beta = new Array(this.params.gaussians).fill(0);
  },

  /**
   * Compute the likelihood of an observation given the GMM's parameters
   * @param  {Array<Number>} observation Observation vector
   * @return {Number}
   */
  likelihood(observation) {
    let likelihood = 0;
    for (let c = 0; c < this.params.gaussians; c += 1) {
      this.beta[c] = this.componentLikelihood(observation, c);
      likelihood += this.beta[c];
    }
    for (let c = 0; c < this.params.gaussians; c += 1) {
      this.beta[c] /= likelihood;
    }

    return likelihood;
  },

  /**
   * Compute the likelihood of an observation for a single component
   * @param  {Array<Number>} observation Observation vector
   * @param  {Number} mixtureComponent Component index
   * @return {Number}
   * @private
   */
  componentLikelihood(observation, mixtureComponent) {
    if (mixtureComponent >= this.params.gaussians) {
      throw new Error('The index of the Gaussian Mixture Component is out of bounds');
    }
    return this.params.mixtureCoeffs[mixtureComponent] * this.params
      .components[mixtureComponent].likelihood(observation);
  },

  /**
   * Update the inverse covariance of each Gaussian component
   * @private
   */
  updateInverseCovariances() {
    this.params.components.forEach((c) => {
      c.updateInverseCovariance();
    });
    try {
      this.params.components.forEach((c) => {
        c.updateInverseCovariance();
      });
    } catch (e) {
      throw new Error('Matrix inversion error: varianceoffset must be too small');
    }
  },

  /**
   * Normalize the mixing coefficients of the Gaussian mixture
   * @private
   */
  normalizeMixtureCoeffs() {
    let normConst = 0;
    for (let c = 0; c < this.params.gaussians; c += 1) {
      normConst += this.params.mixtureCoeffs[c];
    }
    if (normConst > 0) {
      for (let c = 0; c < this.params.gaussians; c += 1) {
        this.params.mixtureCoeffs[c] /= normConst;
      }
    } else {
      for (let c = 0; c < this.params.gaussians; c += 1) {
        this.params.mixtureCoeffs[c] = 1 / this.params.gaussians;
      }
    }
  },

  /**
   * Regularize the covariances
   * @private
   */
  regularize() {
    this.params.components.forEach((c) => {
      c.regularize(this.currentRegularization);
    });
  },
};

/**
 * Bimodal (regression) GMM Prototype
 * @type {Object}
 * @ignore
 */
const gmmBimodalPrototype = /** @lends withGMMBase */ {
  /**
   * Estimate the output values corresponding to the input observation, by
   * regression given the GMM's parameters. This method is called Gaussian
   * Mixture Regression (GMR).
   *
   * @param  {Array<Number>} inputObservation Observation on the input modality
   * @return {Array<Number>} Output values (length = outputDimension)
   */
  regression(inputObservation) {
    this.results.outputValues = Array(this.params.outputDimension).fill(0);
    this.results.outputCovariance = Array(this.params.covarianceMode === 'full' ? this.params.outputDimension ** 2 : this.params.outputDimension).fill(0);
    let tmpOutputValues;

    for (let c = 0; c < this.params.gaussians; c += 1) {
      tmpOutputValues = this.params.components[c].regression(inputObservation);
      for (let d = 0; d < this.params.outputDimension; d += 1) {
        this.results.outputValues[d] += this.beta[c] * tmpOutputValues[d];
        if (this.params.covarianceMode === 'full') {
          for (let d2 = 0; d2 < this.params.outputDimension; d2 += 1) {
            this.results.outputCovariance[(d * this.params.outputDimension) + d2]
              += (this.beta[c] ** 2)
              * this.params.components[c].outputCovariance[(d * this.params.outputDimension) + d2];
          }
        } else {
          this.results.outputCovariance[d] += (this.beta[c] ** 2)
            * this.params.components[c].outputCovariance[d];
        }
      }
    }
    return this.results.outputValues;
  },
};

/**
 * Add basic GMM capabilities to a single-class model. This enables the
 * computation of the likelihoods and regression operations common to
 * training and prediction
 *
 * @see withGMMTraining
 * @see withGMMPrediction
 *
 * @param  {ModelBase} o Source Model
 * @return {GMMBaseModel}
 *
 * @throws {Error} is o is not a ModelBase
 */
export default function withGMMBase(o) {
  if (!isBaseModel(o)) {
    throw new Error('The base object must include a standard set of parameters (`params` key), @see `ModelBase`.');
  }
  return Object.assign(
    o,
    gmmBasePrototype,
    o.params.bimodal ? gmmBimodalPrototype : {},
  );
}
