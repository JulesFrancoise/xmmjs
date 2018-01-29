import ModelBase from './model_base_mixin';

const MulticlassBasePrototype = {
  size() {
    return this.models.size;
  },

  includes(label) {
    return Object.keys(this.models).includes(label);
  },

  remove(label) {
    if (this.includes(label)) {
      delete this.models[label];
    }
  },
};

export default function MulticlassModelbase({
  inputDimension,
  outputDimension,
  ...parameters
}) {
  return Object.assign(
    ModelBase({ inputDimension, outputDimension, ...parameters }),
    MulticlassBasePrototype,
  );
}
