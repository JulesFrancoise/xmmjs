/**
 * Create the skeleton of a model
 *
 * @function
 * @param       {Number} inputDimension  input dimension
 * @param       {Number} outputDimension output dimension
 * @param       {Object} parameters      additional parameters to be copied
 * @constructor
 */
export default function ModelBase({
  inputDimension,
  outputDimension,
  ...parameters
}) {
  const p = parameters;
  delete p.bimodal;
  delete p.inputDimension;
  delete p.outputDimension;
  delete p.dimension;
  return /** @lends ModelBase */{
    params: {
      ...p,
      get bimodal() {
        return outputDimension > 0;
      },
      get inputDimension() {
        return inputDimension;
      },
      get outputDimension() {
        return outputDimension;
      },
      get dimension() {
        return inputDimension + outputDimension;
      },
    },
  };
}

/**
 * Check if an object is a base model (check for attribute existence)
 * @param  {Object}  o Source object
 * @return {Boolean}
 */
export function isBaseModel(o) {
  if (!Object.keys(o).includes('params')) return false;
  const keys = ['bimodal', 'inputDimension', 'outputDimension', 'dimension'];
  return keys.map(key => Object.keys(o.params).includes(key))
    .reduce((a, b) => a && b, true);
}
