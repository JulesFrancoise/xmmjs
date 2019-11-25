import TrainingSet from '../training_set';
import ModelBase from '../core/model_base_mixin';
import withGMMBase from '../gmm/gmm_base_mixin';
import { trainGMM } from '../gmm';

const TRANSITION_REGULARIZATION = 1e-5;

/**
 * HMM Training Prototype
 * @type {Object}
 * @ignore
 */
const hmmTrainerPrototype = /** @lends withHMMTraining */ {
  /**
   * Initialize the EM Training process
   * @param  {TrainingSet} trainingSet Training set
   */
  initTraining(trainingSet) {
    if (!trainingSet || trainingSet.empty()) return;

    this.allocate(trainingSet);
    this.initParametersToDefault(trainingSet.standardDeviation());
    if (this.params.gaussians > 1) {
      this.initMeansCovariancesWithGMMEM(trainingSet);
    } else {
      this.initMeansWithAllPhrases(trainingSet);
      this.initCovariancesFullyObserved(trainingSet);
    }
  },

  /**
   * Allocate the model's parameters and training variables
   * @param  {TrainingSet} trainingSet The training set
   * @private
   */
  allocate(trainingSet) {
    const {
      inputDimension,
      outputDimension,
      gaussians,
      regularization,
      covarianceMode,
    } = this.params;
    this.params.xStates = Array.from(
      new Array(this.params.states),
      () => withGMMBase(ModelBase({
        inputDimension,
        outputDimension,
        gaussians,
        regularization,
        covarianceMode,
      })),
    );
    this.params.xStates.forEach((s) => s.allocate());
    this.alpha = new Array(this.params.states).fill(0);
    this.previousAlpha = new Array(this.params.states).fill(0);
    this.beta = new Array(this.params.states).fill(0);
    this.previousBeta = new Array(this.params.states).fill(0);

    // Initialize Algorithm variables
    // ---------------------------------------
    const nbPhrases = trainingSet.size();
    this.gammaSequence = new Array(nbPhrases).fill(null);
    this.epsilonSequence = new Array(nbPhrases).fill(null);
    this.gammaSequenceperMixture = new Array(nbPhrases).fill(null);
    let maxT = 0;
    let i = 0;
    trainingSet.forEach((phrase) => {
      const T = phrase.length;
      this.gammaSequence[i] = Array.from(
        new Array(T),
        () => new Array(this.params.states).fill(0),
      );
      if (this.params.transitionMode === 'ergodic') {
        this.epsilonSequence[i] = Array.from(
          new Array(T),
          () => Array.from(
            new Array(this.params.states),
            () => new Array(this.params.states).fill(0),
          ),
        );
      } else {
        this.epsilonSequence[i] = Array.from(
          new Array(T),
          () => new Array(this.params.states * 2).fill(0),
        );
      }
      this.gammaSequenceperMixture[i] = new Array(this.params.gaussians).fill(0);
      for (let c = 0; c < this.params.gaussians; c += 1) {
        this.gammaSequenceperMixture[i][c] = Array.from(
          new Array(T),
          () => new Array(this.params.states).fill(0),
        );
      }
      if (T > maxT) {
        maxT = T;
      }
      i += 1;
    });

    this.gammaSum = new Array(this.params.states).fill(0);
    this.gammaSumPerMixture = new Array(this.params.states * this.params.gaussians).fill(0);
  },

  /**
   * Update the EM Training process (1 EM iteration).
   * @param  TrainingSet trainingSet training set
   */
  updateTraining(trainingSet) {
    let logProb = 0;

    // Forward-backward for each phrase
    // =================================================
    let phraseIndex = 0;
    trainingSet.forEach((phrase) => {
      if (phrase.length > 0) {
        logProb += this.baumWelchForwardBackward(phrase, phraseIndex);
      }
      phraseIndex += 1;
    });
    this.baumWelchGammaSum(trainingSet);

    // Re-estimate model parameters
    // =================================================

    // set covariance and mixture coefficients to zero for each state
    for (let i = 0; i < this.params.states; i += 1) {
      for (let c = 0; c < this.params.gaussians; c += 1) {
        this.params.xStates[i].params.mixtureCoeffs[c] = 0;
        if (this.params.covarianceMode === 'full') {
          this.params.xStates[i].params.components[c].covariance = new Array(
            this.params.dimension ** 2,
          ).fill(0);
        } else {
          this.params.xStates[i].params.components[c].covariance = new Array(
            this.params.dimension,
          ).fill(0);
        }
      }
    }

    this.baumWelchEstimateMixtureCoefficients(trainingSet);
    this.baumWelchEstimateMeans(trainingSet);
    this.baumWelchEstimateCovariances(trainingSet);
    if (this.params.transitionMode === 'ergodic') {
      this.baumWelchEstimatePrior(trainingSet);
    }
    this.baumWelchEstimateTransitions(trainingSet);
    return logProb;
  },

  /**
   * terminate the EM Training process
   * @param  TrainingSet trainingSet training set
   */
  terminateTraining() {
    this.normalizeTransitions();
    this.gammaSequence = null;
    this.epsilonSequence = null;
    this.gammaSequenceperMixture = null;
    this.alphaSeq = null;
    this.betaSeq = null;
    this.gammaSum = null;
    this.gammaSumPerMixture = null;
    this.params.xStates = this.params.xStates.map((s) => s.params);
  },

  /**
   * Initialize the model parameters to their default values
   * @param  {Array<Number>} dataStddev Standard deviation of the training data
   * @private
   */
  initParametersToDefault(dataStddev) {
    if (this.params.transitionMode === 'ergodic') {
      this.setErgodic();
    } else {
      this.setLeftRight();
    }
    const currentRegularization = dataStddev.map((std) => Math.max(
      this.params.regularization.absolute,
      this.params.regularization.relative * std,
    ));
    const initCovariance = (this.params.covarianceMode === 'full')
      ? () => new Array(this.params.dimension ** 2)
        .fill(this.params.regularization.absolute / 2)
      : () => new Array(this.params.dimension).fill(0);
    for (let i = 0; i < this.params.states; i += 1) {
      // this.params.xStates[i].initParametersToDefault(dataStddev);
      const s = this.params.xStates[i];
      s.currentRegularization = currentRegularization;
      for (let c = 0; c < this.params.gaussians; c += 1) {
        s.params.components[c].covariance = initCovariance();
        s.params.components[c].regularize(currentRegularization);
        s.params.mixtureCoeffs[c] = 1 / this.params.gaussians;
      }
    }
  },

  /**
   * Initialize the means of each state using all available phrases in the
   * training set
   * @param  {TrainingSet} trainingSet Training set
   * @private
   */
  initMeansWithAllPhrases(trainingSet) {
    if (!trainingSet || trainingSet.empty()) return;

    for (let n = 0; n < this.params.states; n += 1) {
      for (let d = 0; d < this.params.dimension; d += 1) {
        this.params.xStates[n].params.components[0].mean[d] = 0.0;
      }
    }

    const factor = new Array(this.params.states).fill(0);
    trainingSet.forEach((phrase) => {
      const step = Math.floor(phrase.length / this.params.states);
      let offset = 0;
      for (let n = 0; n < this.params.states; n += 1) {
        for (let t = 0; t < step; t += 1) {
          for (let d = 0; d < this.params.dimension; d += 1) {
            this.params.xStates[n].params.components[0].mean[d] += phrase.get(offset + t, d);
          }
        }
        offset += step;
        factor[n] += step;
      }
    });
    for (let n = 0; n < this.params.states; n += 1) {
      for (let d = 0; d < this.params.dimension; d += 1) {
        this.params.xStates[n].params.components[0].mean[d] /= factor[n];
      }
    }
  },

  /**
   * Initialize the covariance by direct (fully-observed) estimation from the
   * training data.
   * @param  {[type]} trainingSet [description]
   * @private
   */
  initCovariancesFullyObserved(trainingSet) {
    if (!trainingSet || trainingSet.empty()) return;

    for (let n = 0; n < this.params.states; n += 1) {
      this.params.xStates[n].params.components[0].covariance = new Array(
        this.params.dimension ** (this.params.covarianceMode === 'full' ? 2 : 1),
      ).fill(0);
    }

    const factor = new Array(this.params.states).fill(0);
    const othermeans = new Array(this.params.states * this.params.dimension)
      .fill(0);
    trainingSet.forEach((phrase) => {
      const step = Math.floor(phrase.length / this.params.states);
      let offset = 0;
      for (let n = 0; n < this.params.states; n += 1) {
        for (let t = 0; t < step; t += 1) {
          for (let d1 = 0; d1 < this.params.dimension; d1 += 1) {
            othermeans[((n * this.params.dimension)) + d1] += phrase.get(offset + t, d1);
            if (this.params.covarianceMode === 'full') {
              for (let d2 = 0; d2 < this.params.dimension; d2 += 1) {
                this.params.xStates[n].params.components[0]
                  .covariance[(d1 * this.params.dimension) + d2]
                    += phrase.get(offset + t, d1) * phrase.get(offset + t, d2);
              }
            } else {
              this.params.xStates[n].params.components[0].covariance[d1]
                += phrase.get(offset + t, d1) ** 2;
            }
          }
        }
        offset += step;
        factor[n] += step;
      }
    });

    for (let n = 0; n < this.params.states; n += 1) {
      for (let d1 = 0; d1 < this.params.dimension; d1 += 1) {
        othermeans[(n * this.params.dimension) + d1] /= factor[n];
        if (this.params.covarianceMode === 'full') {
          for (let d2 = 0; d2 < this.params.dimension; d2 += 1) {
            this.params.xStates[n].params.components[0]
              .covariance[(d1 * this.params.dimension) + d2] /= factor[n];
          }
        } else {
          this.params.xStates[n].params.components[0].covariance[d1] /= factor[n];
        }
      }
    }

    for (let n = 0; n < this.params.states; n += 1) {
      for (let d1 = 0; d1 < this.params.dimension; d1 += 1) {
        if (this.params.covarianceMode === 'full') {
          for (let d2 = 0; d2 < this.params.dimension; d2 += 1) {
            this.params.xStates[n].params.components[0]
              .covariance[(d1 * this.params.dimension) + d2]
                -= othermeans[(n * this.params.dimension) + d1]
                * othermeans[(n * this.params.dimension) + d2];
          }
        } else {
          this.params.xStates[n].params.components[0].covariance[d1]
            -= othermeans[(n * this.params.dimension) + d1]
            * othermeans[(n * this.params.dimension) + d1];
        }
      }
      this.params.xStates[n].regularize();
      this.params.xStates[n].updateInverseCovariances();
    }
  },

  /**
   * Initialize the means and covariance of each state's observation probability
   * distribution using the Expectation-Maximization algorithm for GMMs
   * @param  {[type]} trainingSet [description]
   * @private
   */
  initMeansCovariancesWithGMMEM(trainingSet) {
    for (let n = 0; n < this.params.states; n += 1) {
      const ts = TrainingSet(this.params);
      // eslint-disable-next-line no-loop-func
      trainingSet.forEach((phrase, phraseIndex) => {
        const step = Math.floor(phrase.length / this.params.states);
        if (step > 0) {
          ts.push(phraseIndex, phrase.label);
          for (let t = n * step; t < (n + 1) * step; t += 1) {
            ts.getPhrase(phraseIndex).push(phrase.getFrame(t));
          }
        }
      });
      if (!ts.empty()) {
        const gmmParams = trainGMM(ts, this.params);
        for (let c = 0; c < this.params.gaussians; c += 1) {
          this.params.xStates[n].params.components[c].mean = gmmParams.components[c].mean;
          this.params.xStates[n].params.components[c].covariance = gmmParams
            .components[c].covariance;
          this.params.xStates[n].updateInverseCovariances();
        }
      }
    }
  },

  /**
   * Initialize the transition matrix to an ergodic transition matrix
   * @private
   */
  setErgodic() {
    const p = 1 / this.params.states;
    this.params.prior = new Array(this.params.states).fill(p);
    this.params.transition = Array.from(
      new Array(this.params.states),
      () => new Array(this.params.states).fill(p),
    );
  },

  /**
   * Initialize the transition matrix to a left-right transition matrix
   * @private
   */
  setLeftRight() {
    this.params.prior = new Array(this.params.states).fill(0);
    this.params.prior[0] = 1;
    this.params.transition = new Array(this.params.states * 2).fill(0.5);
    this.params.transition[(this.params.states - 1) * 2] = 1;
    this.params.transition[((this.params.states - 1) * 2) + 1] = 0;
  },

  /**
   * Normalize the hidden state transition parameters
   * (prior + transition matrix)
   * @private
   */
  normalizeTransitions() {
    if (this.params.transitionMode === 'ergodic') {
      const normPrior = this.params.prior.reduce((a, b) => a + b, 0);
      for (let i = 0; i < this.params.states; i += 1) {
        this.params.prior[i] /= normPrior;
        let transitionNorm = 0;
        for (let j = 0; j < this.params.states; j += 1) {
          transitionNorm += this.params.transition[i][j];
        }
        for (let j = 0; j < this.params.states; j += 1) {
          this.params.transition[i][j] /= transitionNorm;
        }
      }
    } else {
      for (let i = 0; i < this.params.states; i += 1) {
        const transitionNorm = this.params.transition[i * 2] + this.params.transition[(i * 2) + 1];
        this.params.transition[i * 2] /= transitionNorm;
        this.params.transition[(i * 2) + 1] /= transitionNorm;
      }
    }
  },

  /**
   * Initialize the backward algorithm (see rabiner, 1989)
   * @param  {Number} ct Inverse probability at time T - 1 (last observation of
   * the sequence)
   * @private
   */
  initializeBackwardAlgorithm(ct) {
    for (let i = 0; i < this.params.states; i += 1) {
      this.beta[i] = ct;
    }
  },

  /**
   * Initialize the backward algorithm (see rabiner, 1989)
   * @param  {Number} ct Inverse probability at time t
   * @param  {Array<Number>} observation Observation vector
   * @private
   */
  updateBackwardAlgorithm(ct, observation) {
    this.previousBeta = this.beta.slice();
    for (let i = 0; i < this.params.states; i += 1) {
      this.beta[i] = 0;
      if (this.params.transitionMode === 'ergodic') {
        for (let j = 0; j < this.params.states; j += 1) {
          this.beta[i] += this.params.transition[i][j] * this.previousBeta[j]
            * this.params.xStates[j].likelihood(observation);
        }
      } else {
        this.beta[i] += this.params.transition[i * 2] * this.previousBeta[i]
          * this.params.xStates[i].likelihood(observation);
        if (i < this.params.states - 1) {
          this.beta[i] += this.params.transition[(i * 2) + 1] * this.previousBeta[i + 1]
            * this.params.xStates[i + 1].likelihood(observation);
        }
      }
      this.beta[i] *= ct;
      if (Number.isNaN(this.beta[i]) || Math.abs(this.beta[i]) === +Infinity) {
        this.beta[i] = 1e100;
      }
    }
  },

  /**
   * Forward algorithm update step for the Baum-Welch algorithms. It is similar
   * to `updateForwardAlgorithm` except it takes precomputed observation
   * likelihoods as argument.
   * @param  {Array<Number>} observationLikelihoods observation likelihoods
   * @private
   */
  baumWelchForwardUpdate(observationLikelihoods) {
    let normConst = 0;
    this.previousAlpha = this.alpha.slice();
    for (let j = 0; j < this.params.states; j += 1) {
      this.alpha[j] = 0;
      if (this.params.transitionMode === 'ergodic') {
        for (let i = 0; i < this.params.states; i += 1) {
          this.alpha[j] += this.previousAlpha[i]
            * this.params.transition[i][j];
        }
      } else {
        this.alpha[j] += this.previousAlpha[j] * this.params.transition[j * 2];
        if (j > 0) {
          this.alpha[j] += this.previousAlpha[j - 1] * this.params.transition[((j - 1) * 2) + 1];
        } else {
          this.alpha[0] += this.previousAlpha[this.params.states - 1]
            * this.params.transition[(this.params.states * 2) - 1];
        }
      }
      this.alpha[j] *= observationLikelihoods[j];
      normConst += this.alpha[j];
    }
    if (Number.isNaN(normConst)) {
      throw new Error('Holy molly');
    }
    if (normConst > 1e-300) {
      for (let j = 0; j < this.params.states; j += 1) {
        this.alpha[j] /= normConst;
      }
      return 1 / normConst;
    }
    return 0;
  },

  /**
   * Backward algorithm update step for the Baum-Welch algorithms. It is similar
   * to `updatebackwardAlgorithm` except it takes precomputed observation
   * likelihoods as argument.
   * @param  {Number} ct Inverse probability at time t
   * @param  {Array<Number>} observationLikelihoods observation likelihoods
   * @private
   */
  baumWelchBackwardUpdate(ct, observationLikelihoods) {
    this.previousBeta = this.beta.slice();
    for (let i = 0; i < this.params.states; i += 1) {
      this.beta[i] = 0;
      if (this.params.transitionMode === 'ergodic') {
        for (let j = 0; j < this.params.states; j += 1) {
          this.beta[i] += this.params.transition[i][j] * this.previousBeta[j]
            * observationLikelihoods[j];
        }
      } else {
        this.beta[i] += this.params.transition[i * 2] * this.previousBeta[i]
          * observationLikelihoods[i];
        if (i < this.params.states - 1) {
          this.beta[i] += this.params.transition[(i * 2) + 1]
            * this.previousBeta[i + 1] * observationLikelihoods[i + 1];
        }
      }
      this.beta[i] *= ct;
      if (Number.isNaN(this.beta[i]) || Math.abs(this.beta[i]) === +Infinity) {
        this.beta[i] = 1e100;
      }
    }
  },

  /**
   * Forward-Backward algorithm for the Baum-Welch training algorithm
   * @param  {Phrase} currentPhrase Current data phrase
   * @param  {Number} phraseIndex   Current phrase index
   * @return {Number} Log-likelihood
   * @private
   */
  baumWelchForwardBackward(currentPhrase, phraseIndex) {
    const T = currentPhrase.length;

    const ct = new Array(T).fill(0);
    let logProb;
    this.alphaSeq = [];
    this.betaSeq = [];

    const observationProbabilities = Array.from(
      new Array(T),
      () => new Array(this.params.states).fill(0),
    );
    for (let t = 0; t < T; t += 1) {
      for (let i = 0; i < this.params.states; i += 1) {
        observationProbabilities[t][i] = this.params.xStates[i]
          .likelihood(currentPhrase.getFrame(t));
      }
    }

    // Forward algorithm
    ct[0] = this.initializeForwardAlgorithm(currentPhrase.getFrame(0));
    logProb = -Math.log(ct[0]);
    this.alphaSeq.push(this.alpha.slice());

    for (let t = 1; t < T; t += 1) {
      ct[t] = this.baumWelchForwardUpdate(observationProbabilities[t]);
      logProb -= Math.log(ct[t]);
      this.alphaSeq.push(this.alpha.slice());
    }

    // Backward algorithm
    this.initializeBackwardAlgorithm(ct[T - 1]);
    this.betaSeq.push(this.beta.slice());

    for (let t = T - 2; t >= 0; t -= 1) {
      this.baumWelchBackwardUpdate(ct[t], observationProbabilities[t + 1]);
      this.betaSeq.push(this.beta.slice());
    }
    this.betaSeq.reverse();

    // Compute Gamma Variable
    for (let t = 0; t < T; t += 1) {
      for (let i = 0; i < this.params.states; i += 1) {
        this.gammaSequence[phraseIndex][t][i] = (this.alphaSeq[t][i] * this.betaSeq[t][i]) / ct[t];
      }
    }

    // Compute Gamma variable for each mixture component
    let normConst;

    for (let t = 0; t < T; t += 1) {
      for (let i = 0; i < this.params.states; i += 1) {
        normConst = 0;
        if (this.params.gaussians === 1) {
          const oo = observationProbabilities[t][i];
          this.gammaSequenceperMixture[phraseIndex][0][t][i] = this
            .gammaSequence[phraseIndex][t][i] * oo;
          normConst += oo;
        } else {
          for (let c = 0; c < this.params.gaussians; c += 1) {
            const oo = this.params.xStates[i]
              .componentLikelihood(currentPhrase.getFrame(t), c);
            this.gammaSequenceperMixture[phraseIndex][c][t][i] = this
              .gammaSequence[phraseIndex][t][i] * oo;
            normConst += oo;
          }
        }
        if (normConst > 0) {
          for (let c = 0; c < this.params.gaussians; c += 1) {
            this.gammaSequenceperMixture[phraseIndex][c][t][i] /= normConst;
          }
        }
      }
    }

    // Compute Epsilon Variable
    if (this.params.transitionMode === 'ergodic') {
      for (let t = 0; t < T - 1; t += 1) {
        for (let i = 0; i < this.params.states; i += 1) {
          for (let j = 0; j < this.params.states; j += 1) {
            this.epsilonSequence[phraseIndex][t][i][j] = this.alphaSeq[t][i]
              * this.params.transition[i][j] * this.betaSeq[t + 1][j];
            this.epsilonSequence[phraseIndex][t][i][j] *= observationProbabilities[t + 1][j];
          }
        }
      }
    } else {
      for (let t = 0; t < T - 1; t += 1) {
        for (let i = 0; i < this.params.states; i += 1) {
          this.epsilonSequence[phraseIndex][t][i * 2] = this.alphaSeq[t][i]
            * this.params.transition[i * 2] * this.betaSeq[t + 1][i];
          this.epsilonSequence[phraseIndex][t][i * 2] *= observationProbabilities[t + 1][i];
          if (i < this.params.states - 1) {
            this.epsilonSequence[phraseIndex][t][(i * 2) + 1] = this.alphaSeq[t][i]
              * this.params.transition[(i * 2) + 1] * this.betaSeq[t + 1][i + 1];
            this.epsilonSequence[phraseIndex][t][(i * 2) + 1]
              *= observationProbabilities[t + 1][i + 1];
          }
        }
      }
    }

    return logProb;
  },

  /**
   * Sums the Gamma variables used for parameter estimation during training
   * @param  {TrainingSet} trainingSet Training Set
   * @private
   */
  baumWelchGammaSum(trainingSet) {
    for (let i = 0; i < this.params.states; i += 1) {
      this.gammaSum[i] = 0;
      for (let c = 0; c < this.params.gaussians; c += 1) {
        this.gammaSumPerMixture[(i * this.params.gaussians) + c] = 0;
      }
    }

    let phraseIndex = 0;
    trainingSet.forEach((phrase) => {
      for (let i = 0; i < this.params.states; i += 1) {
        for (let t = 0; t < phrase.length; t += 1) {
          this.gammaSum[i] += this.gammaSequence[phraseIndex][t][i];
          for (let c = 0; c < this.params.gaussians; c += 1) {
            this.gammaSumPerMixture[(i * this.params.gaussians) + c]
              += this.gammaSequenceperMixture[phraseIndex][c][t][i];
          }
        }
      }
      phraseIndex += 1;
    });
  },

  /**
   * Estimate the mixture coefficients of the GMM observation probability
   * distribution at each state.
   * @param  {TrainingSet} trainingSet Training Set
   * @private
   */
  baumWelchEstimateMixtureCoefficients(trainingSet) {
    let phraseIndex = 0;
    trainingSet.forEach((phrase) => {
      for (let i = 0; i < this.params.states; i += 1) {
        for (let t = 0; t < phrase.length; t += 1) {
          for (let c = 0; c < this.params.gaussians; c += 1) {
            this.params.xStates[i].params.mixtureCoeffs[c] += this
              .gammaSequenceperMixture[phraseIndex][c][t][i];
          }
        }
      }
      phraseIndex += 1;
    });

    // Scale mixture coefficients
    for (let i = 0; i < this.params.states; i += 1) {
      this.params.xStates[i].normalizeMixtureCoeffs();
    }
  },

  /**
   * Estimate the means of the GMM observation probability
   * distribution at each state.
   * @param  {TrainingSet} trainingSet Training Set
   * @private
   */
  baumWelchEstimateMeans(trainingSet) {
    for (let i = 0; i < this.params.states; i += 1) {
      for (let c = 0; c < this.params.gaussians; c += 1) {
        this.params.xStates[i].params.components[c].mean.fill(0);
      }
    }

    // Re-estimate Mean
    let phraseIndex = 0;
    trainingSet.forEach((phrase) => {
      for (let i = 0; i < this.params.states; i += 1) {
        for (let t = 0; t < phrase.length; t += 1) {
          for (let c = 0; c < this.params.gaussians; c += 1) {
            for (let d = 0; d < this.params.dimension; d += 1) {
              this.params.xStates[i].params.components[c].mean[d]
                += this.gammaSequenceperMixture[phraseIndex][c][t][i] * phrase.get(t, d);
            }
          }
        }
      }
      phraseIndex += 1;
    });

    // Normalize mean
    for (let i = 0; i < this.params.states; i += 1) {
      for (let c = 0; c < this.params.gaussians; c += 1) {
        for (let d = 0; d < this.params.dimension; d += 1) {
          if (this.gammaSumPerMixture[(i * this.params.gaussians) + c] > 0) {
            this.params.xStates[i].params.components[c].mean[d]
              /= this.gammaSumPerMixture[(i * this.params.gaussians) + c];
          }
          if (Number.isNaN(this.params.xStates[i].params.components[c].mean[d])) {
            throw new Error('Convergence Error');
          }
        }
      }
    }
  },

  /**
   * Estimate the covariances of the GMM observation probability
   * distribution at each state.
   * @param  {TrainingSet} trainingSet Training Set
   * @private
   */
  baumWelchEstimateCovariances(trainingSet) {
    let phraseIndex = 0;
    trainingSet.forEach((phrase) => {
      for (let i = 0; i < this.params.states; i += 1) {
        for (let t = 0; t < phrase.length; t += 1) {
          for (let c = 0; c < this.params.gaussians; c += 1) {
            for (let d1 = 0; d1 < this.params.dimension; d1 += 1) {
              if (this.params.covarianceMode === 'full') {
                for (let d2 = d1; d2 < this.params.dimension; d2 += 1) {
                  this.params.xStates[i].params.components[c]
                    .covariance[(d1 * this.params.dimension) + d2]
                      += this.gammaSequenceperMixture[phraseIndex][c][t][i]
                      * (phrase.get(t, d1) - this.params.xStates[i].params.components[c].mean[d1])
                      * (phrase.get(t, d2) - this.params.xStates[i].params.components[c].mean[d2]);
                }
              } else {
                const value = phrase.get(t, d1) - this.params
                  .xStates[i].params.components[c].mean[d1];
                this.params.xStates[i].params.components[c].covariance[d1]
                  += this.gammaSequenceperMixture[phraseIndex][c][t][i] * (value ** 2);
              }
            }
          }
        }
      }
      phraseIndex += 1;
    });

    // Scale covariance
    for (let i = 0; i < this.params.states; i += 1) {
      for (let c = 0; c < this.params.gaussians; c += 1) {
        if (this.gammaSumPerMixture[(i * this.params.gaussians) + c] > 0) {
          for (let d1 = 0; d1 < this.params.dimension; d1 += 1) {
            if (this.params.covarianceMode === 'full') {
              for (let d2 = d1; d2 < this.params.dimension; d2 += 1) {
                this.params.xStates[i].params.components[c]
                  .covariance[(d1 * this.params.dimension) + d2]
                    /= this.gammaSumPerMixture[(i * this.params.gaussians) + c];
                if (d1 !== d2) {
                  this.params.xStates[i].params.components[c]
                    .covariance[(d2 * this.params.dimension) + d1] = this.params
                      .xStates[i].params.components[c]
                      .covariance[(d1 * this.params.dimension) + d2];
                }
              }
            } else {
              this.params.xStates[i].params.components[c].covariance[d1]
                /= this.gammaSumPerMixture[(i * this.params.gaussians) + c];
            }
          }
        }
      }
      this.params.xStates[i].regularize();
      this.params.xStates[i].updateInverseCovariances();
    }
  },

  /**
   * Estimate the prior probabilities of the model
   * @param  {TrainingSet} trainingSet Training Set
   * @private
   */
  baumWelchEstimatePrior(trainingSet) {
    this.params.prior.fill(0);

    // Re-estimate Prior probabilities
    let sumprior = 0;
    for (let phraseIndex = 0;
      phraseIndex < trainingSet.size();
      phraseIndex += 1) {
      for (let i = 0; i < this.params.states; i += 1) {
        this.params.prior[i] += this.gammaSequence[phraseIndex][0][i];
        sumprior += this.params.prior[i];
      }
    }

    // Scale Prior vector
    if (sumprior > 0) {
      for (let i = 0; i < this.params.states; i += 1) {
        this.params.prior[i] /= sumprior;
      }
    } else {
      throw new Error('The Prior is all ZERO.....');
    }
  },

  /**
   * Estimate the transition probabilities of the model
   * @param  {TrainingSet} trainingSet Training Set
   * @private
   */
  baumWelchEstimateTransitions(trainingSet) {
    // Set transition matrix to 0
    this.params.transition = this.params.transitionMode === 'ergodic'
      ? Array.from(new Array(this.params.states), () => new Array(this.params.states).fill(0))
      : new Array(this.params.states * 2).fill(0);

    // Re-estimate Transition probabilities
    let phraseIndex = 0;
    trainingSet.forEach((phrase) => {
      if (phrase.length > 0) {
        for (let i = 0; i < this.params.states; i += 1) {
          // Experimental: A bit of regularization (sometimes avoids
          // numerical errors)
          if (this.params.transitionMode === 'leftright') {
            this.params.transition[i * 2] += TRANSITION_REGULARIZATION;
            if (i < this.params.states - 1) {
              this.params.transition[(i * 2) + 1] += TRANSITION_REGULARIZATION;
            } else {
              this.params.transition[i * 2] += TRANSITION_REGULARIZATION;
            }
          }
          // End Regularization
          if (this.params.transitionMode === 'ergodic') {
            for (let j = 0; j < this.params.states; j += 1) {
              for (let t = 0; t < phrase.length - 1; t += 1) {
                this.params.transition[i][j] += this.epsilonSequence[phraseIndex][t][i][j];
              }
            }
          } else {
            for (let t = 0; t < phrase.length - 1; t += 1) {
              this.params.transition[i * 2] += this.epsilonSequence[phraseIndex][t][i * 2];
            }
            if (i < this.params.states - 1) {
              for (let t = 0; t < phrase.length - 1; t += 1) {
                this.params.transition[(i * 2) + 1] += this
                  .epsilonSequence[phraseIndex][t][(i * 2) + 1];
              }
            }
          }
        }
      }
      phraseIndex += 1;
    });

    // Scale transition matrix
    if (this.params.transitionMode === 'ergodic') {
      for (let i = 0; i < this.params.states; i += 1) {
        for (let j = 0; j < this.params.states; j += 1) {
          this.params.transition[i][j] /= (this.gammaSum[i]
            + (2 * TRANSITION_REGULARIZATION));
          if (Number.isNaN(this.params.transition[i][j])) {
            throw new Error('Convergence Error. Check your training data or increase the variance offset');
          }
        }
      }
    } else {
      for (let i = 0; i < this.params.states; i += 1) {
        this.params.transition[i * 2] /= (this.gammaSum[i]
          + (2 * TRANSITION_REGULARIZATION));
        if (Number.isNaN(this.params.transition[i * 2])) {
          throw new Error('Convergence Error. Check your training data or increase the variance offset');
        }
        if (i < this.params.states - 1) {
          this.params.transition[(i * 2) + 1] /= (this.gammaSum[i]
            + (2 * TRANSITION_REGULARIZATION));
          if (Number.isNaN(this.params.transition[(i * 2) + 1])) {
            throw new Error('Convergence Error. Check your training data or increase the variance offset');
          }
        }
      }
    }
  },
};

/**
 * Add HMM Training capabilities to a HMM Model
 * @param  {HMMBase} o               Source HMM Model
 * @param  {Number} [states=1]       Number of hidden states
 * @param  {Number} [gaussians=1]    Number of Gaussian components
 * @param  {Object} [regularization] Regularization parameters
 * @param  {Number} [regularization.absolute=1e-3] Absolute regularization
 * @param  {Number} [regularization.relative=1e-2] Relative Regularization
 (relative to the training set's variance along each dimension)
 * @param  {String} [transitionMode='ergodic'] Structure of the transition
 * matrix ('ergodic' or 'left-right').
 * @param  {String} [covarianceMode='full'] Covariance mode ('full' or diagonal)
 * @return {BMMBase}
 */
export default function withHMMTraining(
  o,
  states = 1,
  gaussians = 1,
  regularization = { absolute: 1e-3, relative: 1e-2 },
  transitionMode = 'leftright',
  covarianceMode = 'full',
) {
  if (!Object.keys(o).includes('params')) {
    throw new Error('The base object must include a standard set of parameters (`params` key), @see `ModelBase`.');
  }
  return Object.assign(
    o,
    hmmTrainerPrototype,
    {
      params: {
        ...o.params,
        states,
        gaussians,
        regularization,
        transitionMode,
        covarianceMode,
      },
    },
  );
}
