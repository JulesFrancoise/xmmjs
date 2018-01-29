import { isBaseModel } from '../core/model_base_mixin';

const gmmBasePrototype = {
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

const gmmBimodalPrototype = {
  regression(inputObservation) {
    this.results.outputValues = Array(this.outputDimension).fill(0);
    this.results.outputCovariance = Array(this.covarianceMode === 'full' ? this.outputDimension ** 2 : this.outputDimension).fill(0);
    const tmpOutputValues = Array(this.outputDimension).fill(0);

    for (let c = 0; c < this.params.gaussians; c += 1) {
      this.params.components[c].regression(inputObservation, tmpOutputValues);
      for (let d = 0; d < this.outputDimension; d += 1) {
        this.results.outputValues[d] += this.beta[c] * tmpOutputValues[d];
        if (this.covarianceMode === 'full') {
          for (let d2 = 0; d2 < this.outputDimension; d2 += 1) {
            this.results.outputCovariance[(d * this.outputDimension) + d2] +=
              (this.beta[c] ** 2) *
              this.params.components[c].outputCovariance[(d * this.outputDimension) + d2];
          }
        } else {
          this.results.outputCovariance[d] +=
            (this.beta[c] ** 2) * this.params.components[c].outputCovariance[d];
        }
      }
    }
    return this.results.outputValues;
  },
};

export default function withGMMBase(o) {
  if (!isBaseModel(o)) {
    throw new Error('The base object must include a standard set of parameters (`params` key), @see `ModelBase`.');
  }
  return Object.assign(
    o,
    gmmBasePrototype,
    o.params.bimodal ? gmmBimodalPrototype : {},
  );
}
