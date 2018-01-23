import AbstractDecoder from './abstract_decoder';

export default class SingleClassGmmDecoder extends AbstractDecoder {
  constructor(
    model,
    likelihoodWindow = 1,
  ) {
    super(model, likelihoodWindow);
    this.gaussians = model.gaussians;
    this.covarianceMode = model.covarianceMode;
    this.components = model.components;
    this.mixtureCoeffs = model.mixtureCoeffs;

    this.beta = new Array(this.gaussians).fill(0);
  }

  likelihood(observation, outputObservation = null) {
    let likelihood = 0;
    for (let c = 0; c < this.gaussians; c += 1) {
      if (this.bimodal) {
        if (outputObservation === null) {
          this.beta[c] = this.obsProbInput(observation, c);
        } else {
          this.beta[c] = this.obsProbBimodal(observation, outputObservation, c);
        }
      } else {
        this.beta[c] = this.obsProb(observation, c);
      }
      likelihood += this.beta[c];
    }
    for (let c = 0; c < this.gaussians; c += 1) {
      this.beta[c] /= likelihood;
    }

    this.results.instantLikelihood = likelihood;
    this.updateResults();
    return likelihood;
  }

  regression(inputObservation) {
    this.results.outputValues = Array(this.outputDimension).fill(0);
    this.results.outputCovariance = Array(this.covarianceMode === 'full' ? this.outputDimension ** 2 : this.outputDimension).fill(0);
    const tmpOutputValues = Array(this.outputDimension).fill(0);

    for (let c = 0; c < this.gaussians.get(); c += 1) {
      this.components[c].regression(inputObservation, tmpOutputValues);
      for (let d = 0; d < this.outputDimension; d += 1) {
        this.results.outputValues[d] += this.beta[c] * tmpOutputValues[d];
        if (this.covarianceMode === 'full') {
          for (let d2 = 0; d2 < this.outputDimension; d2 += 1) {
            this.results.outputCovariance[(d * this.outputDimension) + d2] +=
              (this.beta[c] ** 2) *
              this.components[c].outputCovariance[(d * this.outputDimension) + d2];
          }
        } else {
          this.results.outputCovariance[d] +=
            (this.beta[c] ** 2) * this.components[c].outputCovariance[d];
        }
      }
    }
  }

  obsProb(observation, mixtureComponent = -1) {
    if (mixtureComponent >= this.gaussians) {
      throw new Error('The index of the Gaussian Mixture Component is out of bounds');
    }
    let p = 0;
    if (mixtureComponent < 0) {
      for (let m = 0; m < this.gaussians; m += 1) {
        p += this.obsProb(observation, m);
      }
    } else {
      p = this.mixtureCoeffs[mixtureComponent] *
        this.components[mixtureComponent].likelihood(observation);
    }
    return p;
  }

  obsProbInput(observation, mixtureComponent = -1) {
    if (!this.bimodal) {
      throw new Error('The model is not bimodal');
    }
    if (mixtureComponent >= this.gaussians) {
      throw new Error('The index of the Gaussian Mixture Component is out of bounds');
    }
    let p = 0;
    if (mixtureComponent < 0) {
      for (let m = 0; m < this.gaussians; m += 1) {
        p += this.obsProbInput(observation, m);
      }
    } else {
      p = this.mixtureCoeffs[mixtureComponent] *
        this.components[mixtureComponent].inputLikelihood(observation);
    }
    return p;
  }

  obsProbBimodal(inputObservation, outputObservation, mixtureComponent = -1) {
    if (!this.bimodal) {
      throw new Error('The model is not bimodal');
    }
    if (mixtureComponent >= this.gaussians) {
      throw new Error('The index of the Gaussian Mixture Component is out of bounds');
    }
    let p = 0;
    if (mixtureComponent < 0) {
      for (let m = 0; m < this.gaussians; m += 1) {
        p += this.obsProbBimodal(inputObservation, outputObservation, m);
      }
    } else {
      p = this.mixtureCoeffs[mixtureComponent] *
        this.components[mixtureComponent].bimodalLikelihood(inputObservation, outputObservation);
    }
    return p;
  }

  updateResults() {
    this.likelihoodBuffer.push(Math.log(this.results.instantLikelihood));
    this.results.logLikelihood = 0;
    const bufSize = this.likelihoodBuffer.length;
    for (let i = 0; i < bufSize; i += 1) {
      this.results.logLikelihood += this.likelihoodBuffer.get(i);
    }
    this.results.logLikelihood /= bufSize;
  }
}
