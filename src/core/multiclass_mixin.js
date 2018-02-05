import ModelBase from './model_base_mixin';

/**
 * Multiclass Models Mixin
 * @type {Object}
 * @ignore
 */
const MulticlassBasePrototype = /** @lends MulticlassModelBase */{
  /**
   * Get the number of classes in the model
   * @return {number} number of classes
   */
  size() {
    return Object.keys(this.models).length;
  },

  /**
   * Check if a class with the given label exists
   * @param  {string} label Class label
   * @return {boolean}
   */
  includes(label) {
    return Object.keys(this.models).includes(label);
  },

  /**
   * Remove a class by label
   * @param  {string} label Class label
   */
  remove(label) {
    if (this.includes(label)) {
      delete this.models[label];
    }
  },
};

/**
 * Create an abstract Multiclass Model
 * @param       {number]} inputDimension  input dimension
 * @param       {number]} outputDimension output dimension
 * @param       {Object} parameters       additional parameters to copy
 * @function
 */
export default function MulticlassModelBase({
  inputDimension,
  outputDimension,
  ...parameters
}) {
  return Object.assign(
    ModelBase({ inputDimension, outputDimension, ...parameters }),
    MulticlassBasePrototype,
  );
}
