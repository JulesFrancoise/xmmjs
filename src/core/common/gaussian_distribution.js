import Matrix from './matrix';
import createEllipse from './ellipse';

/**
 * Multivariate Gaussian Distribution.
 *
 * Full covariance, optionally multimodal with support for regression.
 */
export default class GaussianDistribution {
  /**
   * @param {Boolean} [bimodal=false] specify if the distribution is bimodal
   * for use in regression
   * @param {Number}  [dimension=1] dimension of the distribution
   * @param {Number}  [inputDimension=0] dimension of the input modality in
   * bimodal mode.
   * @param {String}  [covarianceMode='full'] covariance mode (full vs
   * diagonal)
   */
  constructor(
    bimodal = false,
    dimension = 1,
    inputDimension = 0,
    covarianceMode = 'full',
  ) {
    this.bimodal = bimodal;
    this.dimension = dimension;
    this.inputDimension = inputDimension;
    this.covarianceMode = covarianceMode;
    this.covarianceDeterminant = 0;
    this.covarianceDeterminantInput = 0;
    this.allocate();
  }

  allocate() {
    this.mean = new Array(this.dimension).fill(0);
    if (this.covarianceMode === 'full') {
      this.covariance = new Array(this.dimension ** 2).fill(0);
      this.inverseCovariance = new Array(this.dimension ** 2).fill(0);
      if (this.bimodal) {
        this.inverseCovarianceInput = new Array(this.inputDimension ** 2).fill(0);
      }
    } else {
      this.covariance = new Array(this.dimension).fill(0);
      this.inverseCovariance = new Array(this.dimension).fill(0);
      if (this.bimodal) {
        this.inverseCovarianceInput = new Array(this.inputDimension).fill(0);
      }
    }
  }

  // === Likelihood & Regression ===
  // ===============================

  /**
   * Get Likelihood of a data vector
   * @param {Array} observation data observation
   * @return {Number} likelihood
   */
  likelihood(observation) {
    if (this.covarianceDeterminant === 0) {
      throw new Error('Covariance Matrix is not invertible');
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
  }

  inputLikelihood(inputObservation) {
    if (!this.bimodal) {
      throw new Error('`likelihood_input` cannot be used when `this.bimodal` is off.');
    }

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
  }

  bimodalLikelihood(inputObservation, outputObservation) {
    if (!this.bimodal) {
      throw new Error('`likelihood_bimodal` cannot be used when `this.bimodal` is off.');
    }

    if (this.covarianceDeterminant === 0) {
      throw new Error('Covariance Matrix is not invertible');
    }

    const outputDimension = this.dimension - this.inputDimension;
    let euclideanDistance = 0;
    if (this.covarianceMode === 'full') {
      for (let l = 0; l < this.dimension; l += 1) {
        let tmp = 0;
        for (let k = 0; k < this.inputDimension; k += 1) {
          tmp += this.inverseCovariance[(l * this.dimension) + k] *
            (inputObservation[k] - this.mean[k]);
        }
        for (let k = 0; k < outputDimension; k += 1) {
          tmp += this.inverseCovariance[(l * this.dimension) + this.inputDimension + k] *
            (outputObservation[k] - this.mean[this.inputDimension + k]);
        }
        if (l < this.inputDimension) {
          euclideanDistance += (inputObservation[l] - this.mean[l]) * tmp;
        } else {
          euclideanDistance +=
              (outputObservation[l - this.inputDimension] - this.mean[l]) *
              tmp;
        }
      }
    } else {
      for (let l = 0; l < this.inputDimension; l += 1) {
        euclideanDistance += this.inverseCovariance[l] *
                             (inputObservation[l] - this.mean[l]) *
                             (inputObservation[l] - this.mean[l]);
      }
      for (let l = this.inputDimension; l < this.dimension; l += 1) {
        euclideanDistance +=
            this.inverseCovariance[l] *
            (outputObservation[l - this.inputDimension] - this.mean[l]) *
            (outputObservation[l - this.inputDimension] - this.mean[l]);
      }
    }

    let p = Math.exp(-0.5 * euclideanDistance) /
      Math.sqrt(this.covarianceDeterminant * ((2 * Math.PI) ** this.dimension));

    if (p < 1e-180 || Number.isNaN(p) || Math.abs(p) === +Infinity) {
      p = 1e-180;
    }

    return p;
  }

  regression(inputObservation) {
    if (!this.bimodal) {
      throw new Error('`regression` cannot be used when `this.bimodal` is off.');
    }

    const outputDimension = this.dimension - this.inputDimension;
    const predictedOutput = Array(outputDimension).fill(0);

    if (this.covarianceMode === 'full') {
      for (let d = 0; d < outputDimension; d += 1) {
        predictedOutput[d] = this.mean[this.inputDimension + d];
        for (let e = 0; e < this.inputDimension; e += 1) {
          let tmp = 0;
          for (let f = 0; f < this.inputDimension; f += 1) {
            tmp += this.inverseCovarianceInput[(e * this.inputDimension) + f] *
              (inputObservation[f] - this.mean[f]);
          }
          predictedOutput[d] += tmp *
            this.covariance[((d + this.inputDimension) * this.dimension) + e];
        }
      }
    } else {
      for (let d = 0; d < outputDimension; d += 1) {
        predictedOutput[d] = this.mean[this.inputDimension + d];
      }
    }
  }

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
  }

  updateInverseCovariance() {
    if (this.covarianceMode === 'full') {
      const covMatrix = new Matrix(this.dimension, this.dimension);

      covMatrix.data = this.covariance.slice();
      const inv = covMatrix.pinv();
      this.covarianceDeterminant = inv.determinant;
      this.inverseCovariance = inv.matrix.data;

      // If regression active: create inverse covariance matrix for input
      // modality.
      if (this.bimodal) {
        const covMatrixInput = new Matrix(this.inputDimension, this.inputDimension);
        for (let d1 = 0; d1 < this.inputDimension; d1 += 1) {
          for (let d2 = 0; d2 < this.inputDimension; d2 += 1) {
            this.covMatrixInput.data[(d1 * this.inputDimension) + d2] =
              this.covariance[(d1 * this.dimension) + d2];
          }
        }
        const invInput = covMatrixInput.pinv();
        this.covarianceDeterminantInput = invInput.determinant;
        this.inverseCovarianceInput = invInput.matrix.data;
      }
    } else { // DIAGONAL COVARIANCE
      this.covarianceDeterminant = 1;
      this.covarianceDeterminantInput = 1;
      for (let d = 0; d < this.dimension; d += 1) {
        if (this.covariance[d] <= 0) {
          throw new Error('Non-invertible matrix');
        }
        this.inverseCovariance[d] = 1 / this.covariance[d];
        this.covarianceDeterminant *= this.covariance[d];
        if (this.bimodal && d < this.inputDimension) {
          this.inverseCovarianceInput[d] = 1 / this.covariance[d];
          this.covarianceDeterminantInput *= this.covariance[d];
        }
      }
    }
    if (this.bimodal) {
      this.updateOutputCovariance();
    }
  }

  updateOutputCovariance() {
    if (!this.bimodal) {
      throw new Error('`updateOutputVariances` cannot be used when `this.bimodal` is off.');
    }

    const outputDimension = this.dimension - this.inputDimension;

    // CASE: DIAGONAL COVARIANCE
    if (this.covarianceMode === 'diagonal') {
      this.outputCovariance = this.covariance.slice(0, this.inputDimension);
      return;
    }

    // CASE: FULL COVARIANCE
    const covMatrixInput = new Matrix(this.inputDimension, this.inputDimension);
    for (let d1 = 0; d1 < this.inputDimension; d1 += 1) {
      for (let d2 = 0; d2 < this.inputDimension; d2 += 1) {
        covMatrixInput.data[(d1 * this.inputDimension) + d2] =
          this.covariance[(d1 * this.dimension) + d2];
      }
    }
    const inv = covMatrixInput.pinv();
    const covarianceGS = new Matrix(this.inputDimension, outputDimension);
    for (let d1 = 0; d1 < this.inputDimension; d1 += 1) {
      for (let d2 = 0; d2 < outputDimension; d2 += 1) {
        covarianceGS.data[(d1 * outputDimension) + d2] =
          this.covariance[(d1 * this.dimension) + this.inputDimension + d2];
      }
    }
    const covarianceSG = new Matrix(outputDimension, this.inputDimension);
    for (let d1 = 0; d1 < outputDimension; d1 += 1) {
      for (let d2 = 0; d2 < this.inputDimension; d2 += 1) {
        covarianceSG.data[(d1 * this.inputDimension) + d2] =
          this.covariance[((this.inputDimension + d1) * this.dimension) + d2];
      }
    }
    const tmptmptmp = inv.matrix.product(covarianceGS);
    const covarianceMod = covarianceSG.product(tmptmptmp);
    this.outputCovariance.resize(outputDimension * outputDimension);
    for (let d1 = 0; d1 < outputDimension; d1 += 1) {
      for (let d2 = 0; d2 < outputDimension; d2 += 1) {
        this.outputCovariance[(d1 * outputDimension) + d2] =
          this.covariance[((this.inputDimension + d1) * this.dimension) +
            this.inputDimension + d2] -
            covarianceMod.data[(d1 * outputDimension) + d2];
      }
    }
  }

  toEllipse(dimension1, dimension2) {
    if (dimension1 >= this.dimension || dimension2 >= this.dimension) {
      throw new Error('dimensions out of range');
    }

    const gaussianEllipse = createEllipse();
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
  }

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
  }
}
