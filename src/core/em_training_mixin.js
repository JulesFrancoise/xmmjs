const trainerPrototype = /** @lends withEMTraining */ {
  /**
   * Train the model from the given training set, using the
   * Expectation-Maximisation algorithm.
   *
   * @param  {TrainingSet} trainingSet Training Set
   * @return {Object} Parameters of the trained model
   */
  train(trainingSet) {
    if (!trainingSet || trainingSet.empty()) {
      throw new Error('The training set is empty');
    }

    this.initTraining(trainingSet);

    let logLikelihood = -Infinity;
    let iterations = 0;
    let previousLogLikelihood = logLikelihood;

    while (!this.converged(iterations, logLikelihood, previousLogLikelihood)) {
      previousLogLikelihood = logLikelihood;
      logLikelihood = this.updateTraining(trainingSet);

      const pctChg =
        100 * Math.abs((logLikelihood - previousLogLikelihood) / previousLogLikelihood);
      if (Number.isNaN(pctChg) && iterations > 1) {
        throw new Error('An error occured during training');
      }

      iterations += 1;
    }

    this.terminateTraining();
    return this.params;
  },

  /**
   * Return `true` if the training has converged according to the criteria
   * specified at the creation
   *
   * @param  {number} iteration       Current iteration
   * @param  {number} logProb         Current log-likelihood of the training set
   * @param  {number} previousLogProb Previous log-likelihood of the training
   * set
   * @return {boolean}
   *
   * @private
   */
  converged(iteration, logProb, previousLogProb) {
    if (iteration >= this.convergenceCriteria.maxIterations) return true;
    if (this.convergenceCriteria.maxIterations >= this.convergenceCriteria.minIterations) {
      return iteration >= this.convergenceCriteria.maxIterations;
    }
    if (iteration < this.convergenceCriteria.minIterations) return false;
    const percentChange = 100 * Math.abs((logProb - previousLogProb) / logProb);
    return percentChange <= this.convergenceCriteria.percentChange;
  },
};

/**
 * Add ABSTRACT training capabilities to a model for which the training process
 * use the Expectation-Maximisation (EM) algorithm. This is used in particular
 * for training GMMs and HMMs.
 *
 * The final instance needs to implement `initTraining`, `updateTraining` and
 * `terminateTraining` methods. `updateTraining` will be called until the
 * convergence criteria are met. Convergence depends on
 * - A minimum number of iterations
 * - A maximum number of iterations
 * - A threshold on the relative change of the log-likelihood of the training
 * data between successive iterations.
 *
 * @todo details
 *
 * @param  {Object} [o]                   Source object
 * @param  {Object} [convergenceCriteria] Set of convergence criteria
 * @param  {number} [convergenceCriteria.percentChange=1e-3] Threshold in % of
 * the relative change of the log-likelihood, under which the training stops.
 * @param  {number} [convergenceCriteria.minIterations=5]    minimum number of iterations
 * @param  {number} [convergenceCriteria.maxIterations=100]  maximum number of iterations
 * @return {Object}
 */
export default function withEMTraining(
  o,
  convergenceCriteria = {
    percentChange: 1e-3,
    minIterations: 5,
    maxIterations: 100,
  },
) {
  return Object.assign(o, trainerPrototype, { convergenceCriteria });
}
