import Phrase from './phrase';

/**
 * Training Set Prototype
 * @ignore
 */
const trainingSetPrototype = /** @lends TrainingSet */ {
  /**
   * Get the training set size (number of phrases)
   * @return {number}
   */
  size() {
    return Object.keys(this.phrases).length;
  },

  /**
   * Checks if the training set is empty
   * @return {boolean}
   */
  empty() {
    return this.length === 0;
  },

  /**
   * Get a reference to a phrase by index
   * @param  {number} phraseIndex phrase index
   * @return {Phrase}
   */
  getPhrase(phraseIndex) {
    if (Object.keys(this.phrases).includes(phraseIndex.toString())) {
      return this.phrases[phraseIndex.toString()];
    }
    return null;
  },

  /**
   * Iterate over all phrases in the training set. The callback function
   * should take 3 arguments: the phrase, its index in the training set,
   * and the phrases structure.
   *
   * @param  {Function} callback Callback function
   */
  forEach(callback) {
    Object.keys(this.phrases).forEach((phraseIndex) => {
      callback(this.phrases[phraseIndex], phraseIndex, this.phrases);
    });
  },

  /**
   * Add a phrase to the training set and return it.
   * @param  {number} phraseIndex        phrase index
   * @param  {string} [label=undefined]  phrase label (its index if undefined)
   * @param  {Phrase} [phrase=undefined] Phrase data. If unspecified, an empty
   * phrase is created.
   * @return {Phrase}
   */
  push(phraseIndex, label = undefined, phrase = undefined) {
    const p = (phrase !== undefined) ? phrase : Phrase({
      inputDimension: this.inputDimension,
      outputDimension: this.outputDimension,
      columnNames: this.columnNames,
      label: (label !== undefined) ? label : phraseIndex.toString(),
    });
    this.phrases[phraseIndex] = p;
    return p;
  },

  /**
   * Remove a phrase
   * @param  {number} phraseIndex phrase index
   */
  remove(phraseIndex) {
    delete this.phrases[phraseIndex];
  },

  /**
   * Remove all phrases with a given label
   * @param  {string} label class label
   */
  removeClass(label) {
    this.phrases = Object.keys(this.phrases)
      .filter(i => this.phrases[i].label !== label)
      .map(i => ({ i: this.phrases[i] }))
      .reduce((x, p) => ({ ...x, ...p }), {});
  },

  /**
   * Clear the training set (delete all phrases)
   */
  clear() {
    this.phrases = {};
  },

  /**
   * Get the sub-training set composed of all phrases of a given class
   * @param  {string} label class label
   * @return {TrainingSet}
   */
  getPhrasesOfClass(label) {
    const ts = TrainingSet(this); // eslint-disable-line no-use-before-define
    ts.phrases = Object.keys(this.phrases)
      .filter(i => this.phrases[i].label === label)
      .map(i => ({ [i]: this.phrases[i] }))
      .reduce((x, p) => ({ ...x, ...p }), {});
    return ts;
  },

  /**
   * Get the list of unique labels in the training set
   * @return {Array<string>}
   */
  labels() {
    return Object.keys(this.phrases)
      .map(i => this.phrases[i].label)
      .reduce((ll, x) => (ll.includes(x) ? ll : ll.concat([x])), []);
  },

  /**
   * Get the list of phrase indices
   * @return {Array<number>}
   */
  indices() {
    return Object.keys(this.phrases);
  },

  /**
   * Get the mean of the training set over all phrases
   * @return {Array<number>} mean (same dimension as the training set)
   */
  mean() {
    const sum = Array(this.dimension).fill(0);
    let totalLength = 0;
    Object.keys(this.phrases).forEach((i) => {
      for (let d = 0; d < this.dimension; d += 1) {
        for (let t = 0; t < this.phrases[i].length; t += 1) {
          sum[d] += this.phrases[i].get(t, d);
        }
      }
      totalLength += this.phrases[i].length;
    });

    return sum.map(x => x / totalLength);
  },

  /**
   * Get the standard deviation of the training set over all phrases
   * @return {Array<number>} standard deviation (same dimension as the training set)
   */
  standardDeviation() {
    const stddev = Array(this.dimension).fill(0);
    const mean = this.mean();
    let totalLength = 0;
    Object.keys(this.phrases).forEach((i) => {
      for (let d = 0; d < this.dimension; d += 1) {
        for (let t = 0; t < this.phrases[i].length; t += 1) {
          stddev[d] += (this.phrases[i].get(t, d) - mean[d]) ** 2;
        }
      }
      totalLength += this.phrases[i].length;
    });

    return stddev.map(x => Math.sqrt(x / totalLength));
  },

  /**
   * Get the min and max of the training set over all phrases
   * @return {Array<{ min: number, max: number }>} min/max (same dimension as the training set)
   */
  minmax() {
    const minmax = Array.from(
      Array(this.dimension),
      () => ({ min: +Infinity, max: -Infinity }),
    );
    Object.keys(this.phrases).forEach((i) => {
      for (let d = 0; d < this.dimension; d += 1) {
        for (let t = 0; t < this.phrases[i].length; t += 1) {
          minmax[d].min += Math.min(minmax[d].min, this.phrases[i].get(t, d));
          minmax[d].max += Math.max(minmax[d].max, this.phrases[i].get(t, d));
        }
      }
    });
    return minmax;
  },
};

/**
 * Create a Training set, composed of a set of indexed data phrases
 * @param {Object} [params]                   Training set parameters
 * @param {Number} [params.inputDimension=1]  Dimension of the input modality
 * @param {Number} [params.outputDimension=0] Dimension of the output modality
 * (optional)
 * @param {Array<String>} [params.columnNames=null] Data column names, e.g.
 * \['accX', 'accY', 'accZ'\] (optional)
 * @return {TrainingSet}
 * @function
 *
 * @property {Boolean} bimodal Specifies if the training set is bimodal
 * @property {Number}  inputDimension Dimension of the input modality
 * @property {Number}  outputDimension Dimension of the output modality
 * @property {Number}  dimension Total dimension
 * @property {Array<String>} columnNames Columns names
 */
export default function TrainingSet({
  inputDimension = 1,
  outputDimension = 0,
  columnNames = null,
} = {}) {
  const dimension = inputDimension + outputDimension;
  return Object.assign(
    Object.create(trainingSetPrototype),
    {
      bimodal: outputDimension > 0,
      inputDimension,
      outputDimension,
      dimension,
      columnNames: columnNames || Array(dimension).fill(''),
      phrases: {},
    },
  );
}
