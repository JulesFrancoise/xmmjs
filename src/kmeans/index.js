import ModelBase from '../core/model_base_mixin';
import withKMeansTraining from './kmeans_training_mixin';

/**
 * Train a K-Means model.
 *
 * @todo K-Means details
 *
 * @param  {TrainingSet} trainingSet           training set
 * @param  {number} clusters                   Number of clusters
 * @param  {Object} [trainingConfig=undefined] Training configuration
 * @return {Object}                            K-Means parameters
 */
export default function trainKmeans(
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
