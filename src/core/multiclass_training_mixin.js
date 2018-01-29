export default function withMulticlassTraining(
  o,
  trainingFunction,
) {
  return Object.assign(
    o,
    {
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
