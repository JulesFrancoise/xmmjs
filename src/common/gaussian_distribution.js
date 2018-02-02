import Matrix from './matrix';

/**
 * Gaussian Distribution Prototype
 *
 * @type {Object}
 * @property {boolean} bimodal           Specifies if the distribution is
 * bimodal (for regression use)
 * @property {number}  inputDimension    input dimension
 * @property {number}  outputDimension   output dimension
 * @property {number}  dimension         Total dimension
 * @property {Array}   mean              Distribution mean
 * @property {Array}   covariance        Distribution covariance
 * @property {Array}   inverseCovariance Inverse covariance
 *
 * @ignore
 */
const baseGaussianPrototype = /** @lends GaussianDistribution */ {
  /**
   * Allocate the distribution
   * @private
   */
  allocate() {
    this.mean = new Array(this.dimension).fill(0);
    if (this.covarianceMode === 'full') {
      this.covariance = new Array(this.dimension ** 2).fill(0);
      this.inverseCovariance = new Array(this.dimension ** 2).fill(0);
    } else {
      this.covariance = new Array(this.dimension).fill(0);
      this.inverseCovariance = new Array(this.dimension).fill(0);
    }
    if (this.bimodal) {
      this.allocateBimodal();
    }
  },

  /**
   * @brief Estimate the likelihood of an observation vector.
   *
   * If the distribution is bimodal an the observation is a vector of the size
   * of the input modality, the likelihood is computed only on the
   * distribution for the input modality
   *
   * @param  {array} observation data observation
   * @return {number}
   */
  likelihood(observation) {
    if (this.covarianceDeterminant === 0) {
      throw new Error('Covariance Matrix is not invertible');
    }
    if (this.bimodal && observation.length === this.inputDimension) {
      return this.inputLikelihood(observation);
    }
    if (observation.length !== this.dimension) {
      throw new Error(`GaussianDistribution: observation has wrong dimension. Expected \`${this.dimension}\`, got \`${observation.length}\``);
    }

    let euclideanDistance = 0;
    if (this.covarianceMode === 'full') {
      for (let l = 0; l < this.dimension; l += 1) {
        let tmp = 0;
        for (let k = 0; k < this.dimension; k += 1) {
          tmp += this.inverseCovariance[(l * this.dimension) + k] *
            (observation[k] - this.mean[k]);
        }
        euclideanDistance += (observation[l] - this.mean[l]) * tmp;
      }
    } else {
      for (let l = 0; l < this.dimension; l += 1) {
        euclideanDistance += this.inverseCovariance[l] *
          (observation[l] - this.mean[l]) *
          (observation[l] - this.mean[l]);
      }
    }

    let p = Math.exp(-0.5 * euclideanDistance) /
      Math.sqrt(this.covarianceDeterminant * ((2 * Math.PI) ** this.dimension));

    if (p < 1e-180 || Number.isNaN(p) || Math.abs(p) === +Infinity) {
      p = 1e-180;
    }

    return p;
  },

  /**
   * Regularize the distribution, given a regularization vector of the same
   * dimension. Regularization adds the vector to the variance of the
   * distribution.
   *
   * @param  {Array} regularization regularization vector
   */
  regularize(regularization) {
    if (this.covarianceMode === 'full') {
      for (let d = 0; d < this.dimension; d += 1) {
        this.covariance[(d * this.dimension) + d] += regularization[d];
      }
    } else {
      for (let d = 0; d < this.dimension; d += 1) {
        this.covariance[d] += regularization[d];
      }
    }
  },

  /**
   * Update the inverse covariance of the distribution
   * @private
   */
  updateInverseCovariance() {
    if (this.covarianceMode === 'full') {
      const covMatrix = Matrix(this.dimension, this.dimension);

      covMatrix.data = this.covariance.slice();
      const inv = covMatrix.pinv();
      this.covarianceDeterminant = inv.determinant;
      this.inverseCovariance = inv.matrix.data;
    } else { // DIAGONAL COVARIANCE
      this.covarianceDeterminant = 1;
      for (let d = 0; d < this.dimension; d += 1) {
        if (this.covariance[d] <= 0) {
          throw new Error('Non-invertible matrix');
        }
        this.inverseCovariance[d] = 1 / this.covariance[d];
        this.covarianceDeterminant *= this.covariance[d];
      }
    }
    if (this.bimodal) {
      this.updateInverseCovarianceBimodal();
    }
  },

  /**
   * Convert to an ellipse allong two dimensions
   *
   * @param  {number} dimension1 first dimension
   * @param  {number} dimension2 second dimension
   * @return {Ellipse}
   */
  toEllipse(dimension1, dimension2) {
    if (dimension1 >= this.dimension || dimension2 >= this.dimension) {
      throw new Error('dimensions out of range');
    }

    const gaussianEllipse = {
      x: 0,
      y: 0,
      width: 0,
      height: 0,
      angle: 0,
    };
    gaussianEllipse.x = this.mean[dimension1];
    gaussianEllipse.y = this.mean[dimension2];

    // Represent 2D covariance with square matrix
    // |a b|
    // |b c|
    let a;
    let b;
    let c;
    if (this.covarianceMode === 'full') {
      a = this.covariance[(dimension1 * this.dimension) + dimension1];
      b = this.covariance[(dimension1 * this.dimension) + dimension2];
      c = this.covariance[(dimension2 * this.dimension) + dimension2];
    } else {
      a = this.covariance[dimension1];
      b = 0;
      c = this.covariance[dimension2];
    }

    // Compute Eigen Values to get width, height and angle
    const trace = a + c;
    const determinant = (a * c) - (b * b);
    const eigenVal1 = 0.5 * (trace + Math.sqrt((trace ** 2) - (4 * determinant)));
    const eigenVal2 = 0.5 * (trace - Math.sqrt((trace ** 2) - (4 * determinant)));
    gaussianEllipse.width = Math.sqrt(5.991 * eigenVal1);
    gaussianEllipse.height = Math.sqrt(5.991 * eigenVal2);
    gaussianEllipse.angle = Math.atan(b / (eigenVal1 - c));
    if (Number.isNaN(gaussianEllipse.angle)) {
      gaussianEllipse.angle = Math.PI / 2;
    }

    return gaussianEllipse;
  },

  /**
   * Modify the distribution along two dimensions given the equivalent values
   * as an Ellipse representation.
   *
   * @param  {Ellipse} gaussianEllipse The Ellipse corresponding to the 2D
   * covariance along the two target dimensions
   * @param  {number} dimension1      first dimension
   * @param  {number} dimension2      second dimension
   */
  fromEllipse(gaussianEllipse, dimension1, dimension2) {
    if (dimension1 >= this.dimension || dimension2 >= this.dimension) {
      throw new Error('dimensions out of range');
    }

    this.mean[dimension1] = gaussianEllipse.x;
    this.mean[dimension2] = gaussianEllipse.y;

    const eigenVal1 = (gaussianEllipse.width * gaussianEllipse.width) / 5.991;
    const eigenVal2 = (gaussianEllipse.height * gaussianEllipse.height) / 5.991;
    const tantheta = Math.tan(gaussianEllipse.angle);
    const b = ((eigenVal1 - eigenVal2) * tantheta) / ((tantheta ** 2) + 1);
    const c = eigenVal1 - (b / tantheta);
    const a = eigenVal2 + (b / tantheta);

    if (this.covarianceMode === 'full') {
      this.covariance[(dimension1 * this.dimension) + dimension1] = a;
      this.covariance[(dimension1 * this.dimension) + dimension2] = b;
      this.covariance[(dimension2 * this.dimension) + dimension1] = b;
      this.covariance[(dimension2 * this.dimension) + dimension2] = c;
    } else {
      this.covariance[dimension1] = a;
      this.covariance[dimension2] = c;
    }
    this.updateInverseCovariance();
  },
};

/**
 * Bimodal Gaussian Distribution Prototype, for Regression purposes
 *
 * @type {Object}
 * @property {boolean} bimodal           Specifies if the distribution is
 * bimodal (for regression use)
 * @property {number}  inputDimension    input dimension
 * @property {number}  outputDimension   output dimension
 * @property {number}  dimension         Total dimension
 * @property {Array}   mean              Distribution mean
 * @property {Array}   covariance        Distribution covariance
 * @property {Array}   inverseCovariance Inverse covariance
 * @property {Array}   inverseCovarianceInput Inverse covariance of the input
 * modality
 *
 * @ignore
 */
const bimodalGaussianPrototype = /** @lends GaussianDistribution */ {
  /**
   * Allocate the distribution
   * @private
   */
  allocateBimodal() {
    if (this.covarianceMode === 'full') {
      this.inverseCovarianceInput = new Array(this.inputDimension ** 2).fill(0);
    } else {
      this.inverseCovarianceInput = new Array(this.inputDimension).fill(0);
    }
  },

  /**
   * Estimate the likelihood of an observation for the input modality only.
   * Called by `likelihood` when relevant.
   * @param  {Array} inputObservation observation (input modality only)
   * @return {number}
   * @private
   */
  inputLikelihood(inputObservation) {
    if (this.covarianceDeterminantInput === 0) {
      throw new Error('Covariance Matrix of input modality is not invertible');
    }

    let euclideanDistance = 0;
    if (this.covarianceMode === 'full') {
      for (let l = 0; l < this.inputDimension; l += 1) {
        let tmp = 0;
        for (let k = 0; k < this.inputDimension; k += 1) {
          tmp += this.inverseCovarianceInput[(l * this.inputDimension) + k] *
            (inputObservation[k] - this.mean[k]);
        }
        euclideanDistance += (inputObservation[l] - this.mean[l]) * tmp;
      }
    } else {
      for (let l = 0; l < this.inputDimension; l += 1) {
        euclideanDistance += this.inverseCovariance[l] *
          (inputObservation[l] - this.mean[l]) *
          (inputObservation[l] - this.mean[l]);
      }
    }

    let p = Math.exp(-0.5 * euclideanDistance) /
               Math.sqrt(this.covarianceDeterminantInput *
                    ((2 * Math.PI) ** this.inputDimension));

    if (p < 1e-180 || Number.isNaN(p) || Math.abs(p) === +Infinity) p = 1e-180;

    return p;
  },

  /**
   * Estimate the output values associated with an input observation by
   * regression, given the distribution parameters.
   *
   * @todo Clarify the maths here.
   *
   * @param  {Array} inputObservation observation (input modality only)
   * @return {Array} Output values
   */
  regression(inputObservation) {
    const outputDimension = this.dimension - this.inputDimension;
    const prediction = Array(outputDimension).fill(0);

    if (this.covarianceMode === 'full') {
      for (let d = 0; d < outputDimension; d += 1) {
        prediction[d] = this.mean[this.inputDimension + d];
        for (let e = 0; e < this.inputDimension; e += 1) {
          let tmp = 0;
          for (let f = 0; f < this.inputDimension; f += 1) {
            tmp += this.inverseCovarianceInput[(e * this.inputDimension) + f] *
              (inputObservation[f] - this.mean[f]);
          }
          prediction[d] += tmp *
            this.covariance[((d + this.inputDimension) * this.dimension) + e];
        }
      }
    } else {
      for (let d = 0; d < outputDimension; d += 1) {
        prediction[d] = this.mean[this.inputDimension + d];
      }
    }
    return prediction;
  },

  /**
   * Update the inverse covariance
   * @private
   */
  updateInverseCovarianceBimodal() {
    if (this.covarianceMode === 'full') {
      const covMatrixInput = Matrix(this.inputDimension, this.inputDimension);
      for (let d1 = 0; d1 < this.inputDimension; d1 += 1) {
        for (let d2 = 0; d2 < this.inputDimension; d2 += 1) {
          covMatrixInput.data[(d1 * this.inputDimension) + d2] =
            this.covariance[(d1 * this.dimension) + d2];
        }
      }
      const invInput = covMatrixInput.pinv();
      this.covarianceDeterminantInput = invInput.determinant;
      this.inverseCovarianceInput = invInput.matrix.data;
    } else { // DIAGONAL COVARIANCE
      this.covarianceDeterminantInput = 1;
      for (let d = 0; d < this.inputDimension; d += 1) {
        if (this.covariance[d] <= 0) {
          throw new Error('Non-invertible matrix');
        }
        this.inverseCovarianceInput[d] = 1 / this.covariance[d];
        this.covarianceDeterminantInput *= this.covariance[d];
      }
    }
    this.updateOutputCovariance();
  },

  /**
   * Update the output covariance
   * @private
   */
  updateOutputCovariance() {
    if (this.covarianceMode === 'diagonal') {
      this.outputCovariance = this.covariance.slice(0, this.inputDimension);
      return;
    }

    // CASE: FULL COVARIANCE
    const covMatrixInput = Matrix(this.inputDimension, this.inputDimension);
    for (let d1 = 0; d1 < this.inputDimension; d1 += 1) {
      for (let d2 = 0; d2 < this.inputDimension; d2 += 1) {
        covMatrixInput.data[(d1 * this.inputDimension) + d2] =
          this.covariance[(d1 * this.dimension) + d2];
      }
    }
    const inv = covMatrixInput.pinv();
    const covarianceGS = Matrix(this.inputDimension, this.outputDimension);
    for (let d1 = 0; d1 < this.inputDimension; d1 += 1) {
      for (let d2 = 0; d2 < this.outputDimension; d2 += 1) {
        covarianceGS.data[(d1 * this.outputDimension) + d2] =
          this.covariance[(d1 * this.dimension) + this.inputDimension + d2];
      }
    }
    const covarianceSG = Matrix(this.outputDimension, this.inputDimension);
    for (let d1 = 0; d1 < this.outputDimension; d1 += 1) {
      for (let d2 = 0; d2 < this.inputDimension; d2 += 1) {
        covarianceSG.data[(d1 * this.inputDimension) + d2] =
          this.covariance[((this.inputDimension + d1) * this.dimension) + d2];
      }
    }
    const tmptmptmp = inv.matrix.product(covarianceGS);
    const covarianceMod = covarianceSG.product(tmptmptmp);
    this.outputCovariance = Array(this.outputDimension ** 2).fill(0);
    for (let d1 = 0; d1 < this.outputDimension; d1 += 1) {
      for (let d2 = 0; d2 < this.outputDimension; d2 += 1) {
        this.outputCovariance[(d1 * this.outputDimension) + d2] =
          this.covariance[((this.inputDimension + d1) * this.dimension) +
            this.inputDimension + d2] -
            covarianceMod.data[(d1 * this.outputDimension) + d2];
      }
    }
  },
};

/**
 * Multivariate Gaussian Distribution factory function.
 * Full covariance, optionally multimodal with support for regression.
 *
 * @function
 * @param {Number} [inputDimension=1]      Dimension of the input modality
 * @param {Number} [outputDimension=0]     Dimension of the output
 * modality (positive for regression, otherwise 0 for recognition).
 * @param {String} [covarianceMode='full'] covariance mode (full vs
 * diagonal)
 * @return {baseGaussianPrototype|bimodalGaussianPrototype}
 *
 * @property {boolean} bimodal           Specifies if the distribution is
 * bimodal (for regression use)
 * @property {number}  inputDimension    input dimension
 * @property {number}  outputDimension   output dimension
 * @property {number}  dimension         Total dimension
 * @property {Array}   mean              Distribution mean
 * @property {Array}   covariance        Distribution covariance
 * @property {Array}   inverseCovariance Inverse covariance
 */
export default function GaussianDistribution(
  inputDimension = 1,
  outputDimension = 0,
  covarianceMode = 'full',
) {
  const bimodal = outputDimension > 0;
  const dimension = inputDimension + outputDimension;
  const proto = bimodal ?
    Object.assign({}, baseGaussianPrototype, bimodalGaussianPrototype) :
    baseGaussianPrototype;
  const data = Object.assign(
    {
      bimodal,
      dimension,
      inputDimension,
      outputDimension,
      covarianceMode,
      covarianceDeterminant: 0,
    },
    bimodal ? { covarianceDeterminantInput: 0 } : {},
  );
  const dist = Object.assign(
    Object.create(proto),
    data,
  );
  dist.allocate();
  return dist;
}
