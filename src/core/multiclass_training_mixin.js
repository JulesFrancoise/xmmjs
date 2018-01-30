/**
 * Add multiclass training capabilities to a model. It takes as argument
 * the training function called to train each class of the training set.
 *
 * @param  {MulticlassModelBase} o Source model
 * @param  {Function}  trainingFunction Training function for a single class
 * @return {MulticlassModelBase}
 */
export default function withMulticlassTraining(
  o,
  trainingFunction,
) {
  return Object.assign(
    o,
    /** @lends withMulticlassTraining */ {
      /**
       * Train the model, optionally specifying a set of classes to train
       *
       * @param  {TrainingSet} trainingSet   Training data set
       * @param  {undefined|Array<String>} [labels=undefined] Labels
       * corresponding to the classes to be trained (all if unspecified)
       * @return {Object} the parameters of the trained model
       *
       * @throws {Error} if the training set is empty
       * @throws {Error} if one of the specified class does not exist
       */
      train(trainingSet, labels = undefined) {
        if (!trainingSet || trainingSet.empty()) {
          throw new Error('The training set is empty');
        }
        if (labels) {
          labels.forEach((l) => {
            if (!this.includes(l)) {
              throw new Error(`Class labeled ${l} does not exist`);
            }
          });
        }

        this.params.classes = {};
        const labs = labels || trainingSet.labels();
        labs.forEach((label) => {
          const ts = trainingSet.getPhrasesOfClass(label);
          // console.log(ts);
          this.params.classes[label] = trainingFunction(ts);
        });
        return this.params;
      },
    },
  );
}
