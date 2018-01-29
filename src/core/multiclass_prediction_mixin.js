import { isBaseModel } from './model_base_mixin';

const MulticlassPredictionBasePrototype = {
  reset() {
    Object.values(this.models).forEach(m => m.reset());
    this.results = {
      labels: [],
      instantLikelihoods: [],
      smoothedLikelihoods: [],
      smoothedLogLikelihoods: [],
      smoothedNormalizedLikelihoods: [],
      likeliest: null,
      classes: {},
    };
    if (this.params.bimodal) {
      this.resetBimodal();
    }
  },

  predict(observation) {
    Object.values(this.models).forEach(m => m.predict(observation));
    this.updateResults();
  },

  updateResults() {
    const labs = Object.keys(this.models).sort();
    this.results.labels = labs;
    let normInstant = 0;
    let normSmoothed = 0;
    let maxLogLikelihood = -Infinity;
    this.results.classes = labs
      .map((lab, i) => {
        this.results.instantLikelihoods[i] =
          this.models[lab].results.instantLikelihood;
        this.results.smoothedLogLikelihoods[i] =
          this.models[lab].results.logLikelihood;
        this.results.smoothedLikelihoods[i] =
          Math.exp(this.results.smoothedLogLikelihoods[i]);
        normInstant += this.results.instantLikelihoods[i];
        normSmoothed += this.results.smoothedLikelihoods[i];
        if (this.results.smoothedLogLikelihoods[i] > maxLogLikelihood) {
          maxLogLikelihood = this.results.smoothedLogLikelihoods[i];
          this.results.likeliest = lab;
        }
        return { [lab]: this.models[lab].results };
      })
      .reduce((o, x) => ({ ...o, ...x }), {});
    this.results.smoothedNormalizedLikelihoods =
      this.results.smoothedLikelihoods.map(x => x / normSmoothed);
    this.results.instantNormalizedLikelihoods =
      this.results.instantLikelihoods.map(x => x / normInstant);
    if (this.params.bimodal) {
      this.updateRegressionResults();
    }
  },
};

const MulticlassPredictionBimodalPrototype = {
  resetBimodal() {
    this.results.outputValues = [];
    this.results.outputCovariance = [];
  },

  updateRegressionResults() {
    if (this.params.multiClassRegressionEstimator === 'likeliest') {
      this.results.outputValues =
        this.models[this.results.likeliest].results.outputValues;
      this.results.outputCovariance =
        this.models[this.results.likeliest].results.outputCovariance;
    } else if (this.params.multiClassRegressionEstimator === 'mixture') {
      this.results.outputValues = Array(this.outputDimension).fill(0);
      this.results.outputCovariance = Array(this.outputDimension ** (this.configuration.covarianceMode === 'full' ? 2 : 1)).fill(0);
      this.results.labels.forEach((lab) => {
        this.results.outputValues.map((x, i) => x + (
          this.results.smoothedNormalizedLikelihoods[i] *
          this.models[lab].results.outputValues[i]
        ));
        this.results.outputCovariance.map((x, i) => x + (
          this.results.smoothedNormalizedLikelihoods[i] *
          this.models[lab].results.outputCovariance[i]
        ));
      });
    } else {
      throw new Error('Unknown regression estimator, use `likeliest` or `mixture`');
    }
  },
};

export default function withMulticlassPrediction(o, multiClassRegressionEstimator = 'likeliest') {
  if (!isBaseModel(o)) {
    throw new Error('The base object must include a standard set of parameters (`params` key), @see `ModelBase`.');
  }
  const m = Object.assign(
    o,
    MulticlassPredictionBasePrototype,
    o.params.bimodal ? MulticlassPredictionBimodalPrototype : {},
  );
  m.params.multiClassRegressionEstimator = multiClassRegressionEstimator;
  return m;
}
