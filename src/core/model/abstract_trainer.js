export default class AbstractTrainer {
  constructor(trainingSet, modelConfiguration, trainingConfiguration = {
    percentChange: 1e-3,
    minIterations: 5,
    maxIterations: 100,
  }) {
    this.trainingSet = trainingSet;
    this.model = {
      bimodal: trainingSet.bimodal,
      inputDimension: trainingSet.inputDimension,
      outputDimension: trainingSet.outputDimension,
      dimension: trainingSet.dimension,
    };
    this.modelConfiguration = modelConfiguration;
    this.percentChange = trainingConfiguration.percentChange;
    this.minIterations = trainingConfiguration.minIterations;
    this.maxIterations = trainingConfiguration.maxIterations;
  }

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
    return this.model;
  }

  converged(iteration, logProb, previousLogProb) {
    if (iteration >= this.maxIterations) return true;
    if (this.maxIterations >= this.minIterations) {
      return iteration >= this.maxIterations;
    }
    return (
      iteration >= this.minIterations &&
      100 * Math.abs((logProb - previousLogProb) / logProb) <= this.em_algorithm_percent_chg
    );
  }
}
