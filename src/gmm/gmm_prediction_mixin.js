import validateParameters from '../common/validation';
import { isBaseModel } from '../core/model_base_mixin';

const gmmParameterSpec = gaussians => ({
  gaussians: {
    required: true,
    check: { min: 1 },
  },
  regularization: {
    required: true,
    check: ({ absolute, relative }) =>
      (absolute && relative && absolute > 0 && relative > 0),
  },
  covarianceMode: {
    required: true,
    check: ['full', 'diagonal'],
  },
  mixtureCoeffs: {
    required: true,
    check: m => m.length === gaussians,
  },
  components: {
    required: true,
    check: c => c.length === gaussians,
  },
});

/**
 * Add GMM prediction capabilities to a single-class model. Mostly, this checks
 * the validity of the model parameters
 *
 * @todo validate gaussian components
 *
 * @param  {GMMBaseModel} o Source Model
 * @return {GMMBaseModel}
 *
 * @throws {Error} is o is not a ModelBase
 */
export default function withGMMPrediction(o) {
  if (!isBaseModel(o)) {
    throw new Error('The base object must include a standard set of parameters (`params` key), @see `ModelBase`.');
  }
  validateParameters('GMM', gmmParameterSpec(o.params.gaussians), o.params);
  return Object.assign(
    o,
    { beta: new Array(o.params.gaussians).fill(0) },
  );
}
