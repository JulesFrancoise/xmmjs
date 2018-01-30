import ModelBase from './core/model_base_mixin';
import withKMeansTraining from './kmeans/kmeans_training_mixin';
import withEMTraining from './core/em_training_mixin';
import withGMMBase from './gmm/gmm_base_mixin';
import withGMMTraining from './gmm/gmm_training_mixin';
import MulticlassModelbase from './core/multiclass_mixin';
import withMulticlassTraining from './core/multiclass_training_mixin';

/**
 * @typedef {Object} GMMParameters
 * @property {Boolean} bimodal Specifies if the model is bimodal
 * @property {Number} inputDimension Dimension of the input modality
 * @property {Number} outputDimension Dimension of the output modality
 * @property {Number} dimension Total dimension
 * @property {Number} gaussians Number of gaussian components in the mixture
 * @property {String} covarianceMode Covariance mode ('full' or 'diagonal')
 * @property {Array<Number>} mixtureCoeffs mixture coefficients ('weight' of
 * each gaussian component)
 * @property {Array<GaussianDistribution>} components Gaussian components
 */

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

/**
 * Train a single-class GMM Model.
 *
 * @todo GMM details
 *
 * @param  {TrainingSet} trainingSet                training set
 * @param  {Object} configuration                   Training configuration
 * @param  {Object} [convergenceCriteria=undefined] Convergence criteria of the
 * EM algorithm
 * @return {GMMParameters} Parameters of the trained GMM
 */
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

/**
 * Train a multi-class GMM Model.
 *
 * @todo GMM details
 *
 * @param  {TrainingSet} trainingSet                training set
 * @param  {Object} configuration                   Training configuration
 * @param  {Object} [convergenceCriteria=undefined] Convergence criteria of the
 * EM algorithm
 * @return {Object} Parameters of the trained GMM
 */
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
