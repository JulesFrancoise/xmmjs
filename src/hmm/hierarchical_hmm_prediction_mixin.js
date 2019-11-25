import { isBaseModel } from '../core/model_base_mixin';

const DEFAULT_EXITPROBABILITY_LAST_STATE = 0.1;

/**
 * Hierarchical HMM Base prototype
 * @type {Object}
 * @ignore
 */
const hierarchicalHmmPredictionPrototype = /** @lends withHierarchicalHMMPrediction */ {
  /**
   * Specificies if the forward algorithm has been initialized
   * @type {Boolean}
   * @private
   */
  forwardInitialized: false,

  /**
   * Setup the model (allocate transition parameters)
   * @return {HierarchicalHMM} [description]
   * @private
   */
  setup() {
    const numClasses = this.size();
    this.params.prior = new Array(numClasses).fill(1 / numClasses);
    this.params.transition = Array.from(
      new Array(numClasses),
      () => new Array(numClasses).fill(1 / numClasses),
    );
    this.params.exitTransition = new Array(numClasses).fill(0.1);
    Object.values(this.models).forEach((model) => {
      const m = model;
      m.isHierarchical = true;
    });
    this.updateExitProbabilities();
    return this;
  },

  /**
   * Update the exit probabilities of each sub-Markov model
   * @param  {Array<Number>|undefined} [exitProbabilities=undefined] Vector of
   * exit probabilities (optional)
   * @private
   */
  updateExitProbabilities(exitProbabilities = undefined) {
    const exitProb = (exitProbabilities !== undefined)
      ? exitProbabilities
      : new Array(this.params.states - 1).fill(0)
        .concat([DEFAULT_EXITPROBABILITY_LAST_STATE]);
    Object.keys(this.models).forEach((label) => {
      this.models[label].params.exitProbabilities = exitProb.slice();
    });
  },

  /**
   * Reset the prediction process. This is particularly important for temporal
   * models such as HMMs, that depends on previous observations.
   */
  reset() {
    Object.values(this.models).forEach((m) => m.reset());
    this.results = {
      labels: [],
      instantLikelihoods: [],
      smoothedLikelihoods: [],
      smoothedLogLikelihoods: [],
      smoothedNormalizedLikelihoods: [],
      exitLikelihood: [],
      likeliest: null,
      classes: {},
    };
    if (this.params.bimodal) {
      this.resetBimodal();
    }
    this.forwardInitialized = false;
  },

  /**
   * Make a prediction from a new observation (updates the results member)
   * @param  {Array<Number>} observation Observation vector
   */
  predict(observation) {
    if (this.forwardInitialized) {
      this.updateForwardAlgorithm(observation);
    } else {
      this.initializeForwardAlgorithm(observation);
    }
    Object.keys(this.models).sort().forEach((label) => {
      const model = this.models[label];
      model.updateAlphaWindow();
      model.updateProgress();
      model.updateResults(model.results.instantLikelihood);
    });
    this.updateResults();

    if (this.params.bimodal) {
      Object.values(this.models).forEach((m) => m.regression(observation));

      if (this.params.multiClassRegressionEstimator === 'likeliest') {
        this.results.outputValues = this.models[this.results.likeliest]
          .results.outputValues;
        this.results.outputCovariance = this.models[this.results.likeliest]
          .results.outputCovariance;
      } else {
        this.results.outputValues = new Array(this.outputDimension).fill(0);
        this.results.outputCovariance = new Array(
          this.params.covarianceMode === 'full'
            ? this.outputDimension ** 2
            : this.outputDimension,
        ).fill(0);
        let modelIndex = 0;
        Object.values(this.models).forEach((model) => {
          for (let d = 0; d < this.outputDimension; d += 1) {
            this.results.outputValues[d] += this.results.smoothedNormalizedLikelihoods[modelIndex]
              * model.second.results.outputValues[d];

            if (this.params.covarianceMode === 'full') {
              for (let d2 = 0; d2 < this.outputDimension; d2 += 1) {
                this.results.outputCovariance[(d * this.outputDimension) + d2]
                  += this.results.smoothedNormalizedLikelihoods[modelIndex]
                  * model.results.outputCovariance[(d * this.outputDimension) + d2];
              }
            } else {
              this.results.outputCovariance[d]
                += this.results.smoothedNormalizedLikelihoods[modelIndex]
                * model.second.results.outputCovariance[d];
            }
          }
          modelIndex += 1;
        });
      }
    }
  },

  /**
   * Initialize the forward algorithm of the hierarchical HMM
   * @param  {Array<Number>} observation Observation vector
   * @private
   */
  initializeForwardAlgorithm(observation) {
    let normConst = 0;
    let modelIndex = 0;
    const classes = Object.keys(this.models).sort();
    classes.forEach((label) => {
      const model = this.models[label];
      const N = model.params.states;
      model.alpha1 = new Array(N).fill(0);
      model.alpha2 = new Array(N).fill(0);

      // Compute Emission probability and initialize on the first state of
      // the primitive
      if (model.params.transitionMode === 'ergodic') {
        model.results.instantLikelihood = 0;
        for (let i = 0; i < N; i += 1) {
          model.alpha[i] = this.params.prior[modelIndex] * model.params.prior[i]
            * model.params.xStates[i].likelihood(observation);
          model.results.instantLikelihood += model.alpha[i];
        }
      } else {
        model.alpha[0] = this.params.prior[modelIndex]
          * model.params.xStates[0].likelihood(observation);
        [model.results.instantLikelihood] = model.alpha;
      }
      normConst += model.results.instantLikelihood;
      modelIndex += 1;
    });

    classes.forEach((label) => {
      const model = this.models[label];
      const N = model.params.states;
      for (let i = 0; i < N; i += 1) {
        model.alpha[i] /= normConst;
      }
    });


    this.frontierV1 = new Array(this.size).fill(0);
    this.frontierV2 = new Array(this.size).fill(0);
    this.forwardInitialized = true;
  },

  /**
   * Update the forward algorithm of the hierarchical HMM
   * @param  {Array<Number>} observation Observation vector
   * @private
   */
  updateForwardAlgorithm(observation) {
    let normConst = 0;

    // Frontier Algorithm: variables
    let tmp = 0;

    // Intermediate variables: compute the sum of probabilities of making a
    // transition to a new primitive
    this.frontierV1 = this.likelihoodAlpha(1);
    this.frontierV2 = this.likelihoodAlpha(2);

    // FORWARD UPDATE
    // --------------------------------------
    let dstModelIndex = 0;
    const classes = Object.keys(this.models).sort();
    classes.forEach((label) => {
      const dstModel = this.models[label];
      const N = dstModel.params.states;

      // 1) COMPUTE FRONTIER VARIABLE
      //    --------------------------------------
      // frontier variable : intermediate computation variable
      const front = new Array(N).fill(0);

      if (dstModel.params.transitionMode === 'ergodic') {
        for (let k = 0; k < N; k += 1) {
          for (let j = 0; j < N; j += 1) {
            front[k] += (dstModel.params.transition[j][k]
              / (1 - dstModel.params.exitProbabilities[j]))
              * dstModel.alpha[j];
          }

          for (
            let srcModelIndex = 0;
            srcModelIndex < this.size();
            srcModelIndex += 1
          ) {
            front[k] += dstModel.params.prior[k] * (
              (this.frontierV1[srcModelIndex]
                * this.params.transition[srcModelIndex][dstModelIndex])
              + (this.params.prior[dstModelIndex]
                * this.frontierV2[srcModelIndex])
            );
          }
        }
      } else {
        // k=0: first state of the primitive
        front[0] = dstModel.params.transition[0] * dstModel.alpha[0];

        for (
          let srcModelIndex = 0;
          srcModelIndex < this.size();
          srcModelIndex += 1
        ) {
          front[0] += (this.frontierV1[srcModelIndex]
              * this.params.transition[srcModelIndex][dstModelIndex])
            + (this.params.prior[dstModelIndex]
              * this.frontierV2[srcModelIndex]);
        }

        // k>0: rest of the primitive
        for (let k = 1; k < N; k += 1) {
          front[k] += (dstModel.params.transition[k * 2]
            / (1 - dstModel.params.exitProbabilities[k])) * dstModel.alpha[k];
          front[k] += (dstModel.params.transition[((k - 1) * 2) + 1]
            / (1 - dstModel.params.exitProbabilities[k - 1])) * dstModel.alpha[k - 1];
        }

        for (let k = 0; k < N; k += 1) {
          dstModel.alpha[k] = 0;
          dstModel.alpha1[k] = 0;
          dstModel.alpha2[k] = 0;
        }
      }

      // 2) UPDATE FORWARD VARIABLE
      //    --------------------------------------
      dstModel.results.exitLikelihood = 0.0;
      dstModel.results.instantLikelihood = 0.0;

      // end of the primitive: handle exit states
      for (let k = 0; k < N; k += 1) {
        tmp = dstModel.params.xStates[k].likelihood(observation) * front[k];
        dstModel.alpha2[k] = this.params.exitTransition[dstModelIndex]
          * dstModel.params.exitProbabilities[k] * tmp;
        dstModel.alpha1[k] = (1 - this.params.exitTransition[dstModelIndex])
          * dstModel.params.exitProbabilities[k] * tmp;
        dstModel.alpha[k] = (1 - dstModel.params.exitProbabilities[k]) * tmp;

        dstModel.results.exitLikelihood += dstModel.alpha1[k] + dstModel.alpha2[k];
        dstModel.results.instantLikelihood += dstModel.alpha[k]
          + dstModel.alpha1[k] + dstModel.alpha2[k];
        normConst += tmp;
      }

      dstModel.results.exitRatio = dstModel.results.exitLikelihood
        / dstModel.results.instantLikelihood;

      dstModelIndex += 1;
    });

    classes.forEach((label) => {
      const model = this.models[label];
      const N = model.params.states;
      for (let k = 0; k < N; k += 1) {
        model.alpha[k] /= normConst;
        model.alpha1[k] /= normConst;
        model.alpha2[k] /= normConst;
      }
    });
  },

  /**
   * Compute the likelihood of a given probability.
   * @param  {Number} exitNum Exit level number
   * @return {Array<Number>}
   */
  likelihoodAlpha(exitNum) {
    const likelihoodVector = new Array(this.size()).fill(0);
    if (exitNum < 0) {
      // Likelihood over all exit states
      let modelIndex = 0;
      Object.keys(this.models).sort().forEach((label) => {
        const model = this.models[label];
        likelihoodVector[modelIndex] = 0.0;
        for (let k = 0; k < model.params.states; k += 1) {
          likelihoodVector[modelIndex] += model.second.alpha[k]
            + model.second.alpha1[k] + model.second.alpha2[k];
        }
        modelIndex += 1;
      });
    } else {
      // Likelihood for exit state "exitNum"
      let modelIndex = 0;
      Object.keys(this.models).sort().forEach((label) => {
        const model = this.models[label];
        likelihoodVector[modelIndex] = 0;
        let { alpha } = model;
        if (exitNum === 1) {
          alpha = model.alpha1;
        }
        if (exitNum === 2) {
          alpha = model.alpha2;
        }
        for (let k = 0; k < model.params.states; k += 1) {
          likelihoodVector[modelIndex] += alpha[k];
        }
        modelIndex += 1;
      });
    }
    return likelihoodVector;
  },
};

/**
 * Add Hierarchical HMM prediction capabilities to a multi-class model.
 *
 * @todo algorithmic details
 * @todo validate parameters
 * @todo validate gaussian components
 *
 * @param  {MulticlassBaseModel} o Source Model
 * @return {HierarchicalHMM}
 *
 * @extends withMulticlassPrediction
 *
 * @throws {Error} is o is not a ModelBase
 */
export default function withHierarchicalHMMPrediction(o) {
  if (!isBaseModel(o)) {
    throw new Error('The base object must include a standard set of parameters (`params` key), @see `ModelBase`.');
  }
  // validateParameters(
  //   'Hierarchical HMM',
  //   hierarchicalHmmParameterSpec(o.params.states, o.params.transitionMode),
  //   o.params,
  // );
  return Object.assign(
    o,
    hierarchicalHmmPredictionPrototype,
    {
      // alpha: new Array(o.params.states).fill(0),
      // previous_alpha_: new Array(o.params.states).fill(0),
    },
  ).setup();
}
