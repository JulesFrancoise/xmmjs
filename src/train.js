import ModelBase from './core/model_base_mixin';
import withKMeansTraining from './kmeans/kmeans_training_mixin';
import withEMTraining from './core/em_training_mixin';
import withGMMBase from './gmm/gmm_base_mixin';
import withGMMTraining from './gmm/gmm_training_mixin';
import MulticlassModelbase from './core/multiclass_mixin';
import withMulticlassTraining from './core/multiclass_training_mixin';

export function trainKmeans(
  trainingSet,
  clusters,
  trainingConfig = undefined,
) {
  const { inputDimension, outputDimension } = trainingSet;
  const model = withKMeansTraining(
    ModelBase({
      inputDimension,
      outputDimension,
    }),
    clusters,
    trainingConfig,
  );
  return model.train(trainingSet);
}

export function trainGMM(
  trainingSet,
  configuration,
  convergenceCriteria = undefined,
) {
  const { inputDimension, outputDimension } = trainingSet;
  const { gaussians, regularization, covarianceMode } = configuration;
  const model = withGMMTraining(
    withEMTraining(
      withGMMBase(ModelBase({
        inputDimension,
        outputDimension,
        ...configuration,
      })),
      convergenceCriteria,
    ),
    gaussians,
    regularization,
    covarianceMode,
  );
  return model.train(trainingSet);
}

export function trainMulticlassGMM(
  trainingSet,
  configuration,
  convergenceCriteria = undefined,
) {
  const { inputDimension, outputDimension } = trainingSet;
  const model = withMulticlassTraining(
    MulticlassModelbase({ inputDimension, outputDimension, ...configuration }),
    ts => trainGMM(ts, configuration, convergenceCriteria),
  );
  return model.train(trainingSet);
}
