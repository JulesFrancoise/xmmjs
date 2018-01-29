import ModelBase from './core/model_base_mixin';
import withAbtractPrediction from './core/prediction_mixin';
import withGMMBase from './gmm/gmm_base_mixin';
import withGMMPrediction from './gmm/gmm_prediction_mixin';
import MulticlassModelbase from './core/multiclass_mixin';
import withMulticlassPrediction from './core/multiclass_prediction_mixin';

export function GMMPredictor(
  params,
  likelihoodWindow = undefined,
) {
  const model = withGMMPrediction(withAbtractPrediction(
    withGMMBase(ModelBase(params)),
    likelihoodWindow,
  ));
  model.allocate();
  return model;
}

export function multiclassGMMPredictor(
  params,
  likelihoodWindow = undefined,
) {
  const model = withMulticlassPrediction(MulticlassModelbase(params));
  model.models = {};
  Object.keys(params.classes).forEach((label) => {
    model.models[label] = GMMPredictor(params.classes[label], likelihoodWindow);
  });
  model.reset();
  return model;
}
