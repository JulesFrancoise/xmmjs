/**
 * Check if the specification is respected for a given parameter and value,
 * and clip if relevant.
 *
 * @ignore
 *
 * @param  {String}        model      Stream Operator Name (for logging)
 * @param  {String}        parameter     Attribute name
 * @param  {Specification} specification Attribute specification
 * @param  {*}             value         Attribute value
 * @return {*}                           Type-checked parameter value
 */
function checkSpec(model, parameter, specification, value) {
  if (!specification) return;
  if (specification.constructor === Array && !specification.includes(value)) {
    throw new Error(`Attribute '${parameter}' (value: '${value}') is not allowed for model '${model}' (options: [${specification}]).`);
  } else if (specification.constructor === Object) {
    if (Object.keys(specification).includes('min') && value < specification.min) {
      throw new Error(`Attribute '${parameter}' (value: ${value}) is inferior to the minimum required value of ${specification.min} for model '${model}'.`);
    }
    if (Object.keys(specification).includes('max') && value > specification.max) {
      throw new Error(`Attribute '${parameter}' (value: ${value}) is superior to the maximum required value of ${specification.min} for model '${model}'.`);
    }
  } else if (typeof specification === 'function') {
    if (!specification(value)) {
      throw new Error(`Attribute '${parameter}' (value: ${value}) is incompatible with model '${model}'.`);
    }
  }
}

/**
 * Check the parameters of a model and return the parameters of the
 * output stream.
 *
 * The specification should be a structure of the form:
 * ```
 * const streamSpecification = {
 *   <parameter name>: {
 *     required: <boolean>,
 *     check: <null || Array || { min: <minimum value>, max: <maximum value>} || Function >,
 *     transform: Function,
 *   },
 * };
 * ```
 *
 * @param  {String} model      Name of the model for logging
 * @param  {Object} specification I/O Stream Specification
 * @param  {Object} values        Attributes of the input stream
 * @return {Object}               Attributes of the output stream
 *
 * @example
 * import setupStreamAttributes from 'stream';
 *
 * const specification = {
 *   type: {
 *     required: false,
 *     check: null,
 *     transform: x => x || null,
 *   },
 *   format: {
 *     required: true,
 *     check: ['scalar', 'vector'],
 *     transform: x => x,
 *   },
 *   size: {
 *     required: true,
 *     check: { min: 1 },
 *     transform: x => 2 * x,
 *   },
 *   stuff: {
 *     required: true,
 *     check: x => Math.log2(x) === Math.floor(Math.log2(x)),
 *     transform: x => Math.log2(x),
 *   },
 * };
 *
 * const values = {
 *   type: 'anything',
 *   format: 'vector',
 *   size: 3,
 *   stuff: 8,
 *   another: 'one',
 * };
 *
 * setupStreamAttributes('module name', specification, values);
 * // Returns:
 * // {
 * //   type: 'anything',
 * //   format: 'vector',
 * //   size: 6,
 * //   stuff: 3,
 * //   another: 'one',
 * // }
 */
export default function validateParameters(model, specification, values) {
  const parameters = Object.assign({}, values);
  Object.keys(specification).forEach((attr) => {
    const spec = specification[attr];

    // Check for required parameters
    if (spec.required && !Object.keys(values).includes(attr)) {
      throw new Error(`Stream parameter '${attr}' is required for model '${model}'.`);
    }

    // Check the validity of the input parameters
    checkSpec(model, attr, spec.check, values[attr]);

    parameters[attr] = spec.transform ?
      spec.transform(values[attr]) :
      values[attr];
  });
  return parameters;
}
