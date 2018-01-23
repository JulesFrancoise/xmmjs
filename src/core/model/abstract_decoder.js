import CircularBuffer from '../common/circular_buffer';

export default class AbstractDecoder {
  constructor(model, likelihoodWindow = 1) {
    this.bimodal = model.bimodal;
    this.inputDimension = model.inputDimension;
    this.outputDimension = model.outputDimension;
    this.dimension = model.dimension;
    this.results = {
      instantLikelihood: 0,
      logLikelihood: 0,
    };
    if (this.bimodal) {
      this.results.outputValues = [];
      this.results.outputCovariance = [];
    }

    this.likelihoodBuffer = new CircularBuffer(likelihoodWindow);
  }

  reset() {
    this.likelihoodBuffer.clear();
  }

  predict(observation) {
    const instantaneousLikelihood = this.likelihood(observation);
    if (this.bimodal) {
      this.regression(observation);
    }
    return instantaneousLikelihood;
  }

  updateResults() {
    this.likelihoodBuffer.push(Math.log(this.results.instantLikelihood));
    this.results.logLikelihood = 0;
    const bufSize = this.likelihoodBuffer.length;
    for (let i = 0; i < bufSize; i += 1) {
      this.results.logLikelihood += this.likelihoodBuffer(0, i);
    }
    this.results.logLikelihood /= bufSize;
  }
}
