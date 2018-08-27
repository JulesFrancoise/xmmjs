/**
 * Data Phrase Prototype
 * @ignore
 */
const phrasePrototype = /** @lends Phrase */ {
  /**
   * Get the value at a given index and dimension
   * @param  {Number} index index
   * @param  {Number} dim   dimension
   * @return {Number}
   */
  get(index, dim) {
    if (typeof index !== 'number' || Math.floor(index) !== index) {
      throw new Error('The index must be an integer');
    }
    if (dim >= this.dimension) {
      throw new Error('Phrase: dimension out of bounds');
    }
    if (this.bimodal) {
      if (dim < this.inputDimension) {
        if (index >= this.inputData.length) {
          throw new Error('Phrase: index out of bounds');
        }
        return this.inputData[index][dim];
      }
      if (index >= this.outputData.length) {
        throw new Error('Phrase: index out of bounds');
      }
      return this.outputData[index][dim - this.inputDimension];
    }
    if (index >= this.length) {
      throw new Error('Phrase: index out of bounds');
    }
    if (!this.inputData[index]) {
      throw new Error('WTF?');
    }
    return this.inputData[index][dim];
  },

  /**
   * Get the data frame at a given index
   * @param  {Number} index index
   * @return {Array<number>}
   * @throws {Error} if the index is out of bounds
   */
  getFrame(index) {
    if (index >= this.length) {
      throw new Error('Phrase: index out of bounds');
    }
    if (this.bimodal) {
      return this.inputData[index].concat(this.outputData[index]);
    }
    return this.inputData[index];
  },

  /**
   * Push an observation vector to the phrase
   * @param  {Array<number>} observation observation data
   * @throws {Error} if the observation's dimension does not match the
   * dimension of the training set
   */
  push(observation) {
    // console.log('push:', observation);
    if (observation.length !== this.dimension) {
      throw new Error('Observation has wrong dimension');
    }

    if (this.bimodal) {
      this.inputData.push(observation.slice(0, this.inputDimension));
      this.outputData.push(observation.slice(this.inputDimension, this.dimension));
    } else {
      this.inputData.push(observation);
    }

    this.length += 1;
  },

  /**
   * Push an observation to the input modality only
   * @param  {Array<number>} observation observation data
   * @throws {Error} if the phrase is not bimodal
   * @throws {Error} if the observation's dimension does not match the
   * input dimension of the training set
   */
  pushInput(observation) {
    if (!this.bimodal) {
      throw new Error('this phrase is unimodal, use `push`');
    }
    if (observation.length !== this.inputDimension) {
      throw new Error('Observation has wrong dimension');
    }

    this.inputData.push(observation);
    this.trim();
  },

  /**
   * Push an observation to the output modality only
   * @param  {Array<number>} observation observation data
   * @throws {Error} if the phrase is not bimodal
   * @throws {Error} if the observation's dimension does not match the
   * output dimension of the training set
   */
  pushOutput(observation) {
    if (!this.bimodal) {
      throw new Error('this phrase is unimodal, use `push`');
    }
    if (observation.length !== this.outputDimension) {
      throw new Error('Observation has wrong dimension');
    }

    this.outputData.push(observation);
    this.trim();
  },

  /**
   * Clear the phrase's data
   */
  clear() {
    this.length = 0;
    this.inputData = [];
    this.outputData = [];
  },

  /**
   * Clear the phrase's input data
   */
  clearInput() {
    this.inputData = [];
    this.trim();
  },

  /**
   * Clear the phrase's output data
   */
  clearOutput() {
    this.outputData = [];
    this.trim();
  },

  /**
   * Compute the mean of the phrase (across time)
   * @return {Array<number>} The mean vector (same dimension as the
   * training set)
   */
  mean() {
    const mean = Array(this.dimension).fill(0);
    for (let d = 0; d < this.dimension; d += 1) {
      for (let t = 0; t < this.length; t += 1) {
        mean[d] += this.get(t, d);
      }
      mean[d] /= this.length;
    }
    return mean;
  },

  /**
   * Compute the standard deviation of the phrase (across time)
   * @return {Array<number>} The standard deviation vector (same dimension as
   * the training set)
   */
  standardDeviation() {
    const stddev = Array(this.dimension).fill(0);
    const mean = this.mean();
    for (let d = 0; d < this.dimension; d += 1) {
      for (let t = 0; t < this.length; t += 1) {
        stddev[d] += (this.get(t, d) - mean[d]) * (this.get(t, d) - mean[d]);
      }
      stddev[d] /= this.length;
      stddev[d] = Math.sqrt(stddev[d]);
    }
    return stddev;
  },

  /**
   * Compute the minimum and maximum of the phrase (across time)
   * @return {Array<{ min: number, max: number }>} The min/max vector (same
   * dimension as the training set)
   */
  minmax() {
    const minmax = Array.from(
      Array(this.dimension),
      () => ({ min: +Infinity, max: -Infinity }),
    );
    for (let d = 0; d < this.dimension; d += 1) {
      for (let t = 0; t < this.length; t += 1) {
        minmax[d].min = Math.min(this.get(t, d), minmax[d].min);
        minmax[d].max = Math.max(this.get(t, d), minmax[d].max);
      }
    }
    return minmax;
  },

  /**
   * Trim the phrase length to the minimum of the input and output lengths
   * @private
   */
  trim() {
    if (this.bimodal) {
      this.length = Math.min(this.inputData.length, this.outputData.length);
    }
  },
};

/**
 * Create a data phrase, potentially bimodal. Phrases are data structures for
 * temporal data (e.g. gestures), used to constitute training sets.
 *
 * @param {Object} [params]                   Phrase parameters
 * @param {Number} [params.inputDimension=1]  Dimension of the input modality
 * @param {Number} [params.outputDimension=0] Dimension of the output modality
 * (optional)
 * @param {Array<String>} [params.columnNames=null] Data column names, e.g.
 * \['accX', 'accY', 'accZ'\] (optional)
 * @param {String} [params.label='']          Phrase label
 * @return {Phrase}
 * @function
 *
 * @property {Boolean} bimodal Specifies if the phrase is bimodal
 * @property {Number} inputDimension Dimension of the input modality
 * @property {Number} outputDimension Dimension of the output modality
 * @property {Number} dimension Total dimension
 * @property {Number} length Phrase length (number of frames)
 * @property {String} label Phrase label
 * @property {Array<String>} columnNames Columns names
 */
export default function Phrase({
  inputDimension = 1,
  outputDimension = 0,
  columnNames = null,
  label = '',
} = {}) {
  const dimension = inputDimension + outputDimension;
  return Object.assign(
    Object.create(phrasePrototype),
    {
      bimodal: outputDimension > 0,
      inputDimension,
      outputDimension,
      dimension,
      length: 0,
      label,
      inputData: [],
      outputData: [],
      columnNames: columnNames || Array(dimension).fill(''),
    },
  );
}
