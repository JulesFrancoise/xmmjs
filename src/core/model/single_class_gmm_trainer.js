import AbstractTrainer from './abstract_trainer';
import GaussianDistribution from '../common/gaussian_distribution';

class SingleClassGmmTrainer extends AbstractTrainer {
  constructor(trainingSet, modelConfiguration, trainingConfiguration = {
    percentChange: 1e-3,
    minIterations: 5,
    maxIterations: 100,
  }) {
    super(trainingSet, modelConfiguration, trainingConfiguration);
    this.model.gaussians = modelConfiguration.gaussians;
    this.model.regularization = modelConfiguration.regularization;
    this.model.covarianceMode = modelConfiguration.covarianceMode || 'full';
    this.model.components = Array.from(
      Array(this.gaussians),
      () => new GaussianDistribution(
        this.model.bimodal,
        this.model.dimension,
        this.model.inputDimension,
        this.model.covarianceMode,
      ),
    );
    this.model.mixtureCoeffs = Array.from(this.model.gaussians).fill(0);
  }

  initTraining(trainingSet) {
    this.initParametersToDefault(trainingSet.standardDeviation());
    this.initMeansWithKMeans(trainingSet);
    this.initCovariances(trainingSet);
    this.addCovarianceOffset();
    this.updateInverseCovariances();
  }

  initParametersToDefault(dataStddev) {
    let normCoeffs = 0;
    this.currentRegularization = dataStddev.map(std => Math.max(
      this.model.regularization.absolute,
      this.model.regularization.relative * std,
    ));
    for (let c = 0; c < this.model.gaussians; c += 1) {
      if (this.model.covarianceMode === 'full') {
        this.model.components[c].covariance = Array(this.model.dimension ** 2)
          .fill(this.model.regularization.absolute / 2);
      } else {
        this.model.components[c].covariance = Array(this.model.dimension).fill(0);
      }
      this.model.components[c].regularize(this.currentRegularization);
      this.model.mixtureCoeffs[c] = 1 / this.model.gaussians;
      normCoeffs += this.model.mixtureCoeffs[c];
    }
    for (let c = 0; c < this.model.gaussians; c += 1) {
      this.model.mixtureCoeffs[c] /= normCoeffs;
    }
  }

  initMeansWithKMeans(trainingSet) {
    if (!trainingSet || trainingSet.empty()) return;
    // const kmeans = new KMeans(this.model.gaussians);
    // kmeans.initializationMode = 'biased';
    // kmeans.train(trainingSet);
    for (let c = 0; c < this.model.gaussians; c += 1) {
      for (let d = 0; d < this.model.dimension; d += 1) {
        this.model.components[c].mean[d] = Math.random();
      }
    }
  }

  initCovariances(trainingSet) {
    // TODO: simplify with covariance symmetricity
    // TODO: If Kmeans, covariances from cluster members
    if (!trainingSet || trainingSet.empty()) return;

    for (let n = 0; n < this.model.gaussians; n += 1) {
      this.model.components[n].covariance = Array((this.model.covarianceMode === 'full') ? this.model.dimension ** 2 : this.model.dimension).fill(0);
    }

    const gmeans = Array(this.model.gaussians * this.model.dimension).fill(0);
    const factor = Array(this.model.gaussians).fill(0);
    trainingSet.forEach((phrase) => {
      const step = phrase.length / this.model.gaussians;
      let offset = 0;
      for (let n = 0; n < this.model.gaussians; n += 1) {
        for (let t = 0; t < step; t += 1) {
          for (let d1 = 0; d1 < this.model.dimension; d1 += 1) {
            gmeans[(n * this.model.dimension) + d1] += phrase.get(offset + t, d1);
            if (this.model.covarianceMode === 'full') {
              for (let d2 = 0; d2 < this.model.dimension; d2 += 1) {
                this.model.components[n]
                  .covariance[(d1 * this.model.dimension) + d2] +=
                  phrase.get(offset + t, d1) * phrase.get(offset + t, d2);
              }
            } else {
              this.model.components[n].covariance[d1] +=
                phrase.get(offset + t, d1) ** 2;
            }
          }
        }
        offset += step;
        factor[n] += step;
      }
    });

    for (let n = 0; n < this.model.gaussians; n += 1) {
      for (let d1 = 0; d1 < this.model.dimension; d1 += 1) {
        gmeans[(n * this.model.dimension) + d1] /= factor[n];
        if (this.model.covarianceMode === 'full') {
          for (let d2 = 0; d2 < this.model.dimension; d2 += 1) {
            this.model.components[n].covariance[(d1 * this.model.dimension) + d2] /= factor[n];
          }
        } else {
          this.model.components[n].covariance[d1] /= factor[n];
        }
      }
    }

    for (let n = 0; n < this.model.gaussians; n += 1) {
      for (let d1 = 0; d1 < this.model.dimension; d1 += 1) {
        if (this.model.covarianceMode === 'full') {
          for (let d2 = 0; d2 < this.model.dimension; d2 += 1) {
            this.model.components[n].covariance[(d1 * this.model.dimension) + d2] -=
              gmeans[(n * this.model.dimension) + d1] *
              gmeans[(n * this.model.dimension) + d2];
          }
        } else {
          this.model.components[n].covariance[d1] -=
            gmeans[(n * this.model.dimension) + d1] ** 2;
        }
      }
    }
  }

  addCovarianceOffset() {
    this.model.components.forEach((c) => {
      c.regularize(this.currentRegularization);
    });
  }

  updateInverseCovariances() {
    this.model.components.forEach((c) => {
      c.updateInverseCovariance();
    });
    try {
      this.model.components.forEach((c) => {
        c.updateInverseCovariance();
      });
    } catch (e) {
      throw new Error('Matrix inversion error: varianceoffset must be too small');
    }
  }

  updateTraining(trainingSet) {
    let logProb = 0;
    let totalLength = 0;
    trainingSet.forEach((phrase) => {
      totalLength += phrase.length;
    });
    const phraseIndices = Object.keys(trainingSet.phrases);

    const p = Array.from(
      Array(this.model.gaussians),
      () => new Array(totalLength).fill(0),
    );
    const E = Array(this.model.gaussians).fill(0);
    let tbase = 0;

    trainingSet.forEach((phrase) => {
      for (let t = 0; t < phrase.length; t += 1) {
        let normConst = 0;
        for (let c = 0; c < this.model.gaussians; c += 1) {
          p[c][tbase + t] = this.obsProb(phrase.getFrame(t), c);

          if (p[c][tbase + t] === 0 ||
            Number.isNaN(p[c][tbase + t]) ||
            p[c][tbase + t] === +Infinity) {
            p[c][tbase + t] = 1e-100;
          }
          normConst += p[c][tbase + t];
        }
        for (let c = 0; c < this.model.gaussians; c += 1) {
          p[c][tbase + t] /= normConst;
          E[c] += p[c][tbase + t];
        }
        logProb += Math.log(normConst);
      }
      tbase += phrase.length;
    });

    // Estimate Mixture coefficients
    for (let c = 0; c < this.model.gaussians; c += 1) {
      this.model.mixtureCoeffs[c] = E[c] / totalLength;
    }

    // Estimate means
    for (let c = 0; c < this.model.gaussians; c += 1) {
      for (let d = 0; d < this.model.dimension; d += 1) {
        this.model.components[c].mean[d] = 0;
        tbase = 0;
        for (let pix = 0; pix < phraseIndices.length; pix += 1) {
          const phrase = trainingSet.phrases[phraseIndices[pix]];
          for (let t = 0; t < phrase.length; t += 1) {
            this.model.components[c].mean[d] +=
              p[c][tbase + t] * phrase.get(t, d);
          }
          tbase += phrase.length;
        }
        this.model.components[c].mean[d] /= E[c];
      }
    }

    // estimate covariances
    if (this.model.covariance_mode === 'full') {
      for (let c = 0; c < this.model.gaussians; c += 1) {
        for (let d1 = 0; d1 < this.model.dimension; d1 += 1) {
          for (let d2 = d1; d2 < this.model.dimension; d2 += 1) {
            this.model.components[c].covariance[(d1 * this.model.dimension) + d2] = 0;
            tbase = 0;
            for (let pix = 0; pix < phraseIndices.length; pix += 1) {
              const phrase = trainingSet.phrases[phraseIndices[pix]];
              for (let t = 0; t < phrase.length; t += 1) {
                this.model.components[c].covariance[(d1 * this.model.dimension) + d2] +=
                  p[c][tbase + t] *
                  (phrase.get(t, d1) - this.model.components[c].mean[d1]) *
                  (phrase.get(t, d2) - this.model.components[c].mean[d2]);
              }
              tbase += phrase.length;
            }
            this.model.components[c].covariance[(d1 * this.model.dimension) + d2] /= E[c];
            if (d1 !== d2) {
              this.model.components[c].covariance[(d2 * this.model.dimension) + d1] =
                this.model.components[c].covariance[(d1 * this.model.dimension) + d2];
            }
          }
        }
      }
    } else {
      for (let c = 0; c < this.model.gaussians; c += 1) {
        for (let d1 = 0; d1 < this.model.dimension; d1 += 1) {
          this.model.components[c].covariance[d1] = 0;
          tbase = 0;
          for (let pix = 0; pix < phraseIndices.length; pix += 1) {
            const phrase = trainingSet.phrases[phraseIndices[pix]];
            for (let t = 0; t < phrase.length; t += 1) {
              const value = (phrase.get(t, d1) - this.model.components[c].mean[d1]);
              this.model.components[c].covariance[d1] +=
                    p[c][tbase + t] * value * value;
            }
            tbase += phrase.length;
          }
          this.model.components[c].covariance[d1] /= E[c];
        }
      }
    }

    this.addCovarianceOffset();
    this.updateInverseCovariances();

    return logProb;
  }

  normalizeMixtureCoeffs() {
    let normConst = 0;
    for (let c = 0; c < this.model.gaussians; c += 1) {
      normConst += this.model.mixtureCoeffs[c];
    }
    if (normConst > 0) {
      for (let c = 0; c < this.model.gaussians; c += 1) {
        this.model.mixtureCoeffs[c] /= normConst;
      }
    } else {
      for (let c = 0; c < this.model.gaussians; c += 1) {
        this.model.mixtureCoeffs[c] = 1 / this.model.gaussians;
      }
    }
  }

  terminateTraining() {} // eslint-disable-line

  obsProb(observation, mixtureComponent = -1) {
    // TODO: lift?
    if (mixtureComponent >= this.gaussians) {
      throw new Error('The index of the Gaussian Mixture Component is out of bounds');
    }
    let p = 0;
    if (mixtureComponent < 0) {
      for (let m = 0; m < this.gaussians; m += 1) {
        p += this.obsProb(observation, m);
      }
    } else {
      p = this.model.mixtureCoeffs[mixtureComponent] *
        this.model.components[mixtureComponent].likelihood(observation);
    }
    return p;
  }
}

export default function train(trainingSet, modelConfiguration, trainingConfiguration = {
  percentChange: 1e-3,
  minIterations: 5,
  maxIterations: 100,
}) {
  return new SingleClassGmmTrainer(
    trainingSet,
    modelConfiguration,
    trainingConfiguration,
  ).train(trainingSet);
}
