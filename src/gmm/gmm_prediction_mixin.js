import { isBaseModel } from '../core/model_base_mixin';

const singleGmmPredictionPrototype = {
  allocate() {
    this.beta = new Array(this.params.gaussians).fill(0);
  },

  likelihood(observation) {
    let likelihood = 0;
    for (let c = 0; c < this.params.gaussians; c += 1) {
      this.beta[c] = this.componentLikelihood(observation, c);
      likelihood += this.beta[c];
    }
    for (let c = 0; c < this.params.gaussians; c += 1) {
      this.beta[c] /= likelihood;
    }

    return likelihood;
  },

  componentLikelihood(observation, mixtureComponent) {
    if (mixtureComponent >= this.params.gaussians) {
      throw new Error('The index of the Gaussian Mixture Component is out of bounds');
    }
    return this.params.mixtureCoeffs[mixtureComponent] *
        this.params.components[mixtureComponent].likelihood(observation);
  },
};

const singleGmmBimodalPredictionPrototype = {
  regression(inputObservation) {
    this.results.outputValues = Array(this.params.outputDimension).fill(0);
    this.results.outputCovariance = Array(this.covarianceMode === 'full' ? this.params.outputDimension ** 2 : this.params.outputDimension).fill(0);
    let tmpOutputValues;

    for (let c = 0; c < this.params.gaussians; c += 1) {
      tmpOutputValues = this.params.components[c].regression(inputObservation);
      for (let d = 0; d < this.params.outputDimension; d += 1) {
        this.results.outputValues[d] += this.beta[c] * tmpOutputValues[d];
        if (this.covarianceMode === 'full') {
          for (let d2 = 0; d2 < this.params.outputDimension; d2 += 1) {
            this.results.outputCovariance[(d * this.params.outputDimension) + d2] +=
              (this.beta[c] ** 2) *
              this.params.components[c].outputCovariance[(d * this.params.outputDimension) + d2];
          }
        } else {
          this.results.outputCovariance[d] +=
            (this.beta[c] ** 2) * this.params.components[c].outputCovariance[d];
        }
      }
    }
  },
};

export default function withGMMPrediction(o) {
  if (!isBaseModel(o)) {
    throw new Error('The base object must include a standard set of parameters (`params` key), @see `ModelBase`.');
  }
  return Object.assign(
    o,
    singleGmmPredictionPrototype,
    o.params.bimodal ? singleGmmBimodalPredictionPrototype : {},
  );
}
