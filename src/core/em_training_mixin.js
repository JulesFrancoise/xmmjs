const trainerPrototype = {
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
