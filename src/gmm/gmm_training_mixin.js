import ModelBase from '../core/model_base_mixin';
import withKMeansTraining from '../kmeans/kmeans_training_mixin';

/**
 * GMM Training Prototype
 * @type {Object}
 * @ignore
 */
const gmmTrainerPrototype = /** @lends withGMMTraining */ {
  /**
   * Initialize the EM Training process
   * @param  {TrainingSet} trainingSet Training set
   */
  initTraining(trainingSet) {
    this.allocate();
    this.initParametersToDefault(trainingSet.standardDeviation());
    this.initMeansWithKMeans(trainingSet);
    this.initCovariances(trainingSet);
    this.regularize();
    this.updateInverseCovariances();
  },

  /**
   * Initialize the model parameters to their default values
   * @param  {Array<Number>} dataStddev Standard deviation of the training data
   * @private
   */
  initParametersToDefault(dataStddev) {
    let normCoeffs = 0;
    this.currentRegularization = dataStddev.map((std) => Math.max(
      this.params.regularization.absolute,
      this.params.regularization.relative * std,
    ));
    for (let c = 0; c < this.params.gaussians; c += 1) {
      if (this.params.covarianceMode === 'full') {
        this.params.components[c].covariance = Array(this.params.dimension ** 2)
          .fill(this.params.regularization.absolute / 2);
      } else {
        this.params.components[c].covariance = Array(this.params.dimension).fill(0);
      }
      this.params.components[c].regularize(this.currentRegularization);
      this.params.mixtureCoeffs[c] = 1 / this.params.gaussians;
      normCoeffs += this.params.mixtureCoeffs[c];
    }
    for (let c = 0; c < this.params.gaussians; c += 1) {
      this.params.mixtureCoeffs[c] /= normCoeffs;
    }
  },

  /**
   * Initialize the means of the model using a K-Means algorithm
   *
   * @see withKMeansTraining
   *
   * @param  {TrainingSet} trainingSet training set
   * @private
   */
  initMeansWithKMeans(trainingSet) {
    if (!trainingSet || trainingSet.empty()) return;
    const kmeans = withKMeansTraining(
      ModelBase({
        inputDimension: this.params.inputDimension,
        outputDimension: this.params.outputDimension,
      }),
      this.params.gaussians,
      { initialization: 'data' },
    );
    const kmeansParams = kmeans.train(trainingSet);
    for (let c = 0; c < this.params.gaussians; c += 1) {
      this.params.components[c].mean = kmeansParams.centers[c];
    }
  },

  /**
   * Initialize the covariances of the model from the training set
   *
   * @param  {TrainingSet} trainingSet training set
   * @private
   */
  initCovariances(trainingSet) {
    // TODO: simplify with covariance symmetricity
    // TODO: If Kmeans, covariances from cluster members
    if (!trainingSet || trainingSet.empty()) return;

    for (let n = 0; n < this.params.gaussians; n += 1) {
      this.params.components[n].covariance = Array((this.params.covarianceMode === 'full') ? this.params.dimension ** 2 : this.params.dimension).fill(0);
    }

    const gmeans = Array(this.params.gaussians * this.params.dimension).fill(0);
    const factor = Array(this.params.gaussians).fill(0);
    trainingSet.forEach((phrase) => {
      const step = Math.floor(phrase.length / this.params.gaussians);
      let offset = 0;
      for (let n = 0; n < this.params.gaussians; n += 1) {
        for (let t = 0; t < step; t += 1) {
          for (let d1 = 0; d1 < this.params.dimension; d1 += 1) {
            gmeans[(n * this.params.dimension) + d1] += phrase.get(offset + t, d1);
            if (this.params.covarianceMode === 'full') {
              for (let d2 = 0; d2 < this.params.dimension; d2 += 1) {
                this.params.components[n]
                  .covariance[(d1 * this.params.dimension) + d2]
                    += phrase.get(offset + t, d1) * phrase.get(offset + t, d2);
              }
            } else {
              this.params.components[n].covariance[d1] += phrase.get(offset + t, d1) ** 2;
            }
          }
        }
        offset += step;
        factor[n] += step;
      }
    });

    for (let n = 0; n < this.params.gaussians; n += 1) {
      for (let d1 = 0; d1 < this.params.dimension; d1 += 1) {
        gmeans[(n * this.params.dimension) + d1] /= factor[n];
        if (this.params.covarianceMode === 'full') {
          for (let d2 = 0; d2 < this.params.dimension; d2 += 1) {
            this.params.components[n].covariance[(d1 * this.params.dimension) + d2] /= factor[n];
          }
        } else {
          this.params.components[n].covariance[d1] /= factor[n];
        }
      }
    }

    for (let n = 0; n < this.params.gaussians; n += 1) {
      for (let d1 = 0; d1 < this.params.dimension; d1 += 1) {
        if (this.params.covarianceMode === 'full') {
          for (let d2 = 0; d2 < this.params.dimension; d2 += 1) {
            this.params.components[n].covariance[(d1 * this.params.dimension) + d2]
              -= gmeans[(n * this.params.dimension) + d1]
              * gmeans[(n * this.params.dimension) + d2];
          }
        } else {
          this.params.components[n].covariance[d1]
            -= gmeans[(n * this.params.dimension) + d1] ** 2;
        }
      }
    }
  },

  /**
   * Update the EM Training process (1 EM iteration).
   * @param  TrainingSet trainingSet training set
   */
  updateTraining(trainingSet) {
    let logProb = 0;
    let totalLength = 0;
    trainingSet.forEach((phrase) => {
      totalLength += phrase.length;
    });
    const phraseIndices = Object.keys(trainingSet.phrases);

    const p = Array.from(
      Array(this.params.gaussians),
      () => new Array(totalLength).fill(0),
    );
    const E = Array(this.params.gaussians).fill(0);
    let tbase = 0;

    trainingSet.forEach((phrase) => {
      for (let t = 0; t < phrase.length; t += 1) {
        let normConst = 0;
        for (let c = 0; c < this.params.gaussians; c += 1) {
          p[c][tbase + t] = this.componentLikelihood(phrase.getFrame(t), c);

          if (p[c][tbase + t] === 0
            || Number.isNaN(p[c][tbase + t])
            || p[c][tbase + t] === +Infinity) {
            p[c][tbase + t] = 1e-100;
          }
          normConst += p[c][tbase + t];
        }
        for (let c = 0; c < this.params.gaussians; c += 1) {
          p[c][tbase + t] /= normConst;
          E[c] += p[c][tbase + t];
        }
        logProb += Math.log(normConst);
      }
      tbase += phrase.length;
    });

    // Estimate Mixture coefficients
    for (let c = 0; c < this.params.gaussians; c += 1) {
      this.params.mixtureCoeffs[c] = E[c] / totalLength;
    }

    // Estimate means
    for (let c = 0; c < this.params.gaussians; c += 1) {
      for (let d = 0; d < this.params.dimension; d += 1) {
        this.params.components[c].mean[d] = 0;
        tbase = 0;
        for (let pix = 0; pix < phraseIndices.length; pix += 1) {
          const phrase = trainingSet.phrases[phraseIndices[pix]];
          for (let t = 0; t < phrase.length; t += 1) {
            this.params.components[c].mean[d] += p[c][tbase + t] * phrase.get(t, d);
          }
          tbase += phrase.length;
        }
        this.params.components[c].mean[d] /= E[c];
      }
    }

    // estimate covariances
    if (this.params.covarianceMode === 'full') {
      for (let c = 0; c < this.params.gaussians; c += 1) {
        for (let d1 = 0; d1 < this.params.dimension; d1 += 1) {
          for (let d2 = d1; d2 < this.params.dimension; d2 += 1) {
            this.params.components[c].covariance[(d1 * this.params.dimension) + d2] = 0;
            tbase = 0;
            for (let pix = 0; pix < phraseIndices.length; pix += 1) {
              const phrase = trainingSet.phrases[phraseIndices[pix]];
              for (let t = 0; t < phrase.length; t += 1) {
                this.params.components[c].covariance[(d1 * this.params.dimension) + d2]
                  += p[c][tbase + t] * (phrase.get(t, d1) - this.params.components[c].mean[d1])
                  * (phrase.get(t, d2) - this.params.components[c].mean[d2]);
              }
              tbase += phrase.length;
            }
            this.params.components[c].covariance[(d1 * this.params.dimension) + d2] /= E[c];
            if (d1 !== d2) {
              this.params.components[c].covariance[(d2 * this.params.dimension) + d1] = this.params
                .components[c].covariance[(d1 * this.params.dimension) + d2];
            }
          }
        }
      }
    } else {
      for (let c = 0; c < this.params.gaussians; c += 1) {
        for (let d1 = 0; d1 < this.params.dimension; d1 += 1) {
          this.params.components[c].covariance[d1] = 0;
          tbase = 0;
          for (let pix = 0; pix < phraseIndices.length; pix += 1) {
            const phrase = trainingSet.phrases[phraseIndices[pix]];
            for (let t = 0; t < phrase.length; t += 1) {
              const value = (phrase.get(t, d1) - this.params.components[c].mean[d1]);
              this.params.components[c].covariance[d1] += p[c][tbase + t] * value * value;
            }
            tbase += phrase.length;
          }
          this.params.components[c].covariance[d1] /= E[c];
        }
      }
    }

    this.regularize();
    this.updateInverseCovariances();

    return logProb;
  },

  /**
   * Terminate the EM Training process
   */
  terminateTraining() {},
};

/**
 * Add GMM Training capabilities to a GMM Model
 * @param  {GMMBase} o               Source GMM Model
 * @param  {Number} [gaussians=1]    Number of Gaussian components
 * @param  {Object} [regularization] Regularization parameters
 * @param  {Number} [regularization.absolute=1e-3] Absolute regularization
 * @param  {Number} [regularization.relative=1e-2] Relative Regularization
 (relative to the training set's variance along each dimension)
 * @param  {String} [covarianceMode='full'] Covariance mode ('full' or diagonal)
 * @return {BMMBase}
 */
export default function withGMMTraining(
  o,
  gaussians = 1,
  regularization = { absolute: 1e-3, relative: 1e-2 },
  covarianceMode = 'full',
) {
  if (!Object.keys(o).includes('params')) {
    throw new Error('The base object must include a standard set of parameters (`params` key), @see `ModelBase`.');
  }
  return Object.assign(
    o,
    gmmTrainerPrototype,
    {
      params: {
        ...o.params,
        gaussians,
        regularization,
        covarianceMode,
      },
    },
  );
}
