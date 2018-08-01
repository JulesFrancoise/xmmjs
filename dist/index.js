(function (global, factory) {
  typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports) :
  typeof define === 'function' && define.amd ? define(['exports'], factory) :
  (factory((global.xmm = {})));
}(this, (function (exports) { 'use strict';

  /**
   * Data Phrase Prototype
   * @ignore
   */
  const phrasePrototype = /** @lends Phrase */{
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
      if (observation.size() !== this.inputDimension) {
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
      if (observation.size() !== this.outputDimension) {
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
      const minmax = Array.from(Array(this.dimension), () => ({ min: +Infinity, max: -Infinity }));
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
    }
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
  function Phrase({
    inputDimension = 1,
    outputDimension = 0,
    columnNames = null,
    label = ''
  } = {}) {
    const dimension = inputDimension + outputDimension;
    return Object.assign(Object.create(phrasePrototype), {
      bimodal: outputDimension > 0,
      inputDimension,
      outputDimension,
      dimension,
      length: 0,
      label,
      inputData: [],
      outputData: [],
      columnNames: columnNames || Array(dimension).fill('')
    });
  }

  var _extends = Object.assign || function (target) {
    for (var i = 1; i < arguments.length; i++) {
      var source = arguments[i];

      for (var key in source) {
        if (Object.prototype.hasOwnProperty.call(source, key)) {
          target[key] = source[key];
        }
      }
    }

    return target;
  };

  var objectWithoutProperties = function (obj, keys) {
    var target = {};

    for (var i in obj) {
      if (keys.indexOf(i) >= 0) continue;
      if (!Object.prototype.hasOwnProperty.call(obj, i)) continue;
      target[i] = obj[i];
    }

    return target;
  };

  /**
   * Training Set Prototype
   * @ignore
   */
  const trainingSetPrototype = /** @lends TrainingSet */{
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
      Object.keys(this.phrases).forEach(phraseIndex => {
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
      const p = phrase !== undefined ? phrase : Phrase({
        inputDimension: this.inputDimension,
        outputDimension: this.outputDimension,
        columnNames: this.columnNames,
        label: label !== undefined ? label : phraseIndex.toString()
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
      this.phrases = Object.keys(this.phrases).filter(i => this.phrases[i].label !== label).map(i => ({ i: this.phrases[i] })).reduce((x, p) => _extends({}, x, p), {});
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
      ts.phrases = Object.keys(this.phrases).filter(i => this.phrases[i].label === label).map(i => ({ i: this.phrases[i] })).reduce((x, p) => _extends({}, x, p), {});
      return ts;
    },

    /**
     * Get the list of unique labels in the training set
     * @return {Array<string>}
     */
    labels() {
      return Object.keys(this.phrases).map(i => this.phrases[i].label).reduce((ll, x) => ll.includes(x) ? ll : ll.concat([x]), []);
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
      Object.keys(this.phrases).forEach(i => {
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
      Object.keys(this.phrases).forEach(i => {
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
      const minmax = Array.from(Array(this.dimension), () => ({ min: +Infinity, max: -Infinity }));
      Object.keys(this.phrases).forEach(i => {
        for (let d = 0; d < this.dimension; d += 1) {
          for (let t = 0; t < this.phrases[i].length; t += 1) {
            minmax[d].min += Math.min(minmax[d].min, this.phrases[i].get(t, d));
            minmax[d].max += Math.max(minmax[d].max, this.phrases[i].get(t, d));
          }
        }
      });
      return minmax;
    }
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
  function TrainingSet({
    inputDimension = 1,
    outputDimension = 0,
    columnNames = null
  } = {}) {
    const dimension = inputDimension + outputDimension;
    return Object.assign(Object.create(trainingSetPrototype), {
      bimodal: outputDimension > 0,
      inputDimension,
      outputDimension,
      dimension,
      columnNames: columnNames || Array(dimension).fill(''),
      phrases: {}
    });
  }

  /**
   * Create the skeleton of a model
   *
   * @function
   * @param       {Number} inputDimension  input dimension
   * @param       {Number} outputDimension output dimension
   * @param       {Object} parameters      additional parameters to be copied
   * @constructor
   */
  function ModelBase(_ref) {
    let {
      inputDimension,
      outputDimension
    } = _ref,
        parameters = objectWithoutProperties(_ref, ['inputDimension', 'outputDimension']);

    const p = parameters;
    delete p.bimodal;
    delete p.inputDimension;
    delete p.outputDimension;
    delete p.dimension;
    return (/** @lends ModelBase */{
        params: _extends({}, p, {
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
          }
        })
      }
    );
  }

  /**
   * Check if an object is a base model (check for attribute existence)
   * @param  {Object}  o Source object
   * @return {Boolean}
   */
  function isBaseModel(o) {
    if (!Object.keys(o).includes('params')) return false;
    const keys = ['bimodal', 'inputDimension', 'outputDimension', 'dimension'];
    return keys.map(key => Object.keys(o.params).includes(key)).reduce((a, b) => a && b, true);
  }

  /**
   * Compute the euclidean distance between to vectors
   * @param  {Array} v1
   * @param  {Array} v2
   * @return {number}
   */
  function euclidean(v1, v2) {
    return Math.sqrt(v1.map((x1, i) => (x1 - v2[i]) ** 2).reduce((a, x) => a + x, 0));
  }

  const kMeansTrainingPrototype = {
    train(trainingSet) {
      if (!trainingSet || trainingSet.empty()) {
        throw new Error('The training set is empty');
      }

      this.params.centers = Array.from(Array(this.params.clusters), () => new Array(this.params.dimension).fill(0));

      // TODO: improve initialization =>
      // https://www.slideshare.net/djempol/kmeans-initialization-15041920
      //
      if (this.trainingConfig.initialization === 'random') {
        this.initializeClustersRandom(trainingSet);
      } else if (this.trainingConfig.initialization === 'forgy') {
        this.initializeClustersForgy(trainingSet);
      } else if (this.trainingConfig.initialization === 'data') {
        this.initClustersWithFirstPhrase(trainingSet);
      } else {
        throw new Error('Unknown K-Means initialization, must be `random`, `forgy` or `data`');
      }

      for (let trainingNbIterations = 0; trainingNbIterations < this.trainingConfig.maxIterations; trainingNbIterations += 1) {
        const previousCenters = this.params.centers;

        this.updateCenters(previousCenters, trainingSet);

        let meanClusterDistance = 0;
        let maxRelativeCenterVariation = 0;
        for (let k = 0; k < this.params.clusters; k += 1) {
          for (let l = 0; l < this.params.clusters; l += 1) {
            if (k !== l) {
              meanClusterDistance += euclidean(this.params.centers[k], this.params.centers[l]);
            }
          }
          maxRelativeCenterVariation = Math.max(euclidean(previousCenters[k], this.params.centers[k]), maxRelativeCenterVariation);
        }
        meanClusterDistance /= this.params.clusters * (this.params.clusters - 1);
        maxRelativeCenterVariation /= this.params.clusters;
        maxRelativeCenterVariation /= meanClusterDistance;
        if (maxRelativeCenterVariation < this.trainingConfig.relativeDistanceThreshold) break;
      }
      return this.params;
    },

    initClustersWithFirstPhrase(trainingSet) {
      const phrase = trainingSet.getPhrase(trainingSet.indices()[0]);
      const step = Math.floor(phrase.length / this.params.clusters);

      let offset = 0;
      for (let c = 0; c < this.params.clusters; c += 1) {
        this.params.centers[c] = new Array(this.params.dimension).fill(0);
        for (let t = 0; t < step; t += 1) {
          for (let d = 0; d < this.params.dimension; d += 1) {
            this.params.centers[c][d] += phrase.get(offset + t, d) / step;
          }
        }
        offset += step;
      }
    },

    initializeClustersRandom(trainingSet) {
      const phrase = trainingSet.getPhrase(trainingSet.indices()[0]);
      const indices = Array.from(Array(phrase.length), () => Math.floor(Math.random() * this.params.clusters));
      const pointsPerCluster = indices.reduce((ppc, i) => {
        const p = ppc;
        p[i] += 1;
        return p;
      }, Array(this.params.clusters).fill(0));
      for (let i = 0; i < indices.length; i += 1) {
        const clustIdx = indices[i];
        for (let d = 0; d < this.params.dimension; d += 1) {
          this.params.centers[clustIdx][d] += phrase.get(i, d);
        }
      }
      this.params.centers.forEach((_, c) => {
        this.params.centers[c] = this.params.centers[c].map(x => x / pointsPerCluster[c]);
      });
    },

    initializeClustersForgy(trainingSet) {
      const phrase = trainingSet.getPhrase(trainingSet.indices()[0]);
      const indices = Array.from(Array(this.params.clusters), () => Math.floor(Math.random() * phrase.length));
      this.params.centers = indices.map(i => phrase.getFrame(i));
    },

    updateCenters(previousCenters, trainingSet) {
      this.params.centers = Array.from(Array(this.params.clusters), () => new Array(this.params.dimension).fill(0));
      const numFramesPerCluster = Array(this.params.clusters).fill(0);
      trainingSet.forEach(phrase => {
        for (let t = 0; t < phrase.length; t += 1) {
          const frame = phrase.getFrame(t);
          let minDistance = euclidean(frame, previousCenters[0]);
          let clusterMembership = 0;
          for (let k = 1; k < this.params.clusters; k += 1) {
            const distance = euclidean(frame, previousCenters[k], this.params.dimension);
            if (distance < minDistance) {
              clusterMembership = k;
              minDistance = distance;
            }
          }
          numFramesPerCluster[clusterMembership] += 1;
          for (let d = 0; d < this.params.dimension; d += 1) {
            this.params.centers[clusterMembership][d] += phrase.get(t, d);
          }
        }
      });
      for (let k = 0; k < this.params.clusters; k += 1) {
        if (numFramesPerCluster[k] > 0) {
          for (let d = 0; d < this.params.dimension; d += 1) {
            this.params.centers[k][d] /= numFramesPerCluster[k];
          }
        }
      }
    }
  };

  function withKMeansTraining(o, clusters, trainingConfiguration = {}) {
    if (!isBaseModel(o)) {
      throw new Error('The base object must include a standard set of parameters (`params` key), @see `ModelBase`.');
    }
    const trainingConfig = Object.assign({
      initialization: 'random',
      relativeDistanceThreshold: 1e-3,
      minIterations: 5,
      maxIterations: 100
    }, trainingConfiguration);
    const model = Object.assign(o, kMeansTrainingPrototype, {
      trainingConfig
    });
    model.params.clusters = clusters;
    return model;
  }

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
  function trainKmeans(trainingSet, clusters, trainingConfig = undefined) {
    const { inputDimension, outputDimension } = trainingSet;
    const model = withKMeansTraining(ModelBase({
      inputDimension,
      outputDimension
    }), clusters, trainingConfig);
    return model.train(trainingSet);
  }

  /* eslint-disable no-use-before-define */
  const kEpsilonPseudoInverse = 1.0e-9;

  /**
   * Matrix Prototype
   * @type {Object}
   * @property {Array} data Matrix data
   * @property {Number} ncols Number of columns
   * @property {Number} nrows Number of rows
   *
   * @ignore
   */
  const matrixPrototype = /** @lends Matrix */{
    /**
     * Compute the Sum of the matrix
     * @return {Number} Sum of all elements in the matrix
     */
    sum() {
      return this.data.reduce((a, b) => a + b, 0);
    },

    /**
     * Compute the transpose matrix
     * @return {Matrix}
     */
    transpose() {
      const out = Matrix(this.ncols, this.nrows);
      for (let i = 0; i < this.ncols; i += 1) {
        for (let j = 0; j < this.nrows; j += 1) {
          out.data[i * this.nrows + j] = this.data[j * this.ncols + i];
        }
      }
      return out;
    },

    /**
     * Compute the product of matrices
     * @param  {Matrix} mat Second matrix
     * @return {Matrix}     Product of the current matrix by `mat`
     */
    product(mat) {
      if (this.ncols !== mat.nrows) {
        throw new Error('Wrong dimensions for matrix product');
      }
      const out = Matrix(this.nrows, mat.ncols);
      for (let i = 0; i < this.nrows; i += 1) {
        for (let j = 0; j < mat.ncols; j += 1) {
          out.data[i * mat.ncols + j] = 0;
          for (let k = 0; k < this.ncols; k += 1) {
            out.data[i * mat.ncols + j] += this.data[i * this.ncols + k] * mat.data[k * mat.ncols + j];
          }
        }
      }
      return out;
    },

    /**
     * Compute the Pseudo-Inverse of a Matrix
     * @param  {Number} determinant Determinant (computed with the inversion)
     * @return {Matrix}             Pseudo-inverse of the matrix
     */
    pinv() {
      if (this.nrows === this.ncols) {
        return this.gaussJordanInverse();
      }

      const transp = this.transpose();
      if (this.nrows >= this.ncols) {
        const prod = transp.product(this);
        const { determinant, matrix: dst } = prod.gaussJordanInverse();
        return { determinant, matrix: dst.product(transp) };
      }
      const prod = this.product(transp);
      const { determinant, matrix: dst } = prod.gaussJordanInverse();
      return { determinant, matrix: transp.product(dst) };
    },

    /**
     * Compute the Gauss-Jordan Inverse of a Square Matrix
     * !!! Determinant (computed with the inversion
     * @private
     */
    gaussJordanInverse() {
      if (this.nrows !== this.ncols) {
        throw new Error('Gauss-Jordan inversion: Cannot invert Non-square matrix');
      }
      let determinant = 1;
      const mat = Matrix(this.nrows, this.ncols * 2);
      const newMat = Matrix(this.nrows, this.ncols * 2);
      const n = this.nrows;

      // Create matrix
      for (let i = 0; i < n; i += 1) {
        for (let j = 0; j < n; j += 1) {
          mat.data[i * 2 * n + j] = this.data[i * n + j];
        }
        mat.data[i * 2 * n + n + i] = 1;
      }

      for (let k = 0; k < n; k += 1) {
        let i = k;
        while (Math.abs(mat.data[i * 2 * n + k]) < kEpsilonPseudoInverse) {
          i += 1;
          if (i === n) {
            throw new Error('Non-invertible matrix');
          }
        }
        determinant *= mat.data[i * 2 * n + k];

        // if found > Exchange lines
        if (i !== k) {
          mat.swapLines(i, k);
        }

        newMat.data = mat.data.slice();

        for (let j = 0; j < 2 * n; j += 1) {
          newMat.data[k * 2 * n + j] /= mat.data[k * 2 * n + k];
        }
        for (let ii = 0; ii < n; ii += 1) {
          if (ii !== k) {
            for (let j = 0; j < 2 * n; j += 1) {
              newMat.data[ii * 2 * n + j] -= mat.data[ii * 2 * n + k] * newMat.data[k * 2 * n + j];
            }
          }
        }
        mat.data = newMat.data.slice();
      }

      const dst = Matrix(this.nrows, this.ncols);
      for (let i = 0; i < n; i += 1) {
        for (let j = 0; j < n; j += 1) {
          dst.data[i * n + j] = mat.data[i * 2 * n + n + j];
        }
      }
      return { determinant, matrix: dst };
    },

    /**
     * Swap 2 lines of the matrix
     * @param  {[type]} i index of the first line
     * @param  {[type]} j index of the second line
     * @private
     */
    swapLines(i, j) {
      for (let k = 0; k < this.ncols; k += 1) {
        const tmp = this.data[i * this.ncols + k];
        this.data[i * this.ncols + k] = this.data[j * this.ncols + k];
        this.data[j * this.ncols + k] = tmp;
      }
    },

    /**
     * Swap 2 columns of the matrix
     * @param  {[type]} i index of the first column
     * @param  {[type]} j index of the second column
     * @private
     */
    swapColumns(i, j) {
      for (let k = 0; k < this.nrows; k += 1) {
        const tmp = this.data[k * this.ncols + i];
        this.data[k * this.ncols + i] = this.data[k * this.ncols + j];
        this.data[k * this.ncols + j] = tmp;
      }
    }
  };

  /**
   * Create a matrix
   *
   * @function
   * @param       {Number} [nrows=0]  Number of rows
   * @param       {Number} [ncols=-1] Number of columns
   * @return {matrixPrototype}
   *
   * @property {Array} data Matrix data
   * @property {Number} ncols Number of columns
   * @property {Number} nrows Number of rows
   */
  function Matrix(nrows = 0, ncols = -1) {
    const nc = ncols < 0 ? nrows : ncols;
    return Object.assign(Object.create(matrixPrototype), //
    {
      nrows,
      ncols: nc,
      data: Array(nrows * nc).fill(0)
    });
  }

  /**
   * Gaussian Distribution Prototype
   *
   * @type {Object}
   * @property {boolean} bimodal           Specifies if the distribution is
   * bimodal (for regression use)
   * @property {number}  inputDimension    input dimension
   * @property {number}  outputDimension   output dimension
   * @property {number}  dimension         Total dimension
   * @property {Array}   mean              Distribution mean
   * @property {Array}   covariance        Distribution covariance
   * @property {Array}   inverseCovariance Inverse covariance
   *
   * @ignore
   */
  const baseGaussianPrototype = /** @lends GaussianDistribution */{
    /**
     * Allocate the distribution
     * @private
     */
    allocate() {
      this.mean = new Array(this.dimension).fill(0);
      if (this.covarianceMode === 'full') {
        this.covariance = new Array(this.dimension ** 2).fill(0);
        this.inverseCovariance = new Array(this.dimension ** 2).fill(0);
      } else {
        this.covariance = new Array(this.dimension).fill(0);
        this.inverseCovariance = new Array(this.dimension).fill(0);
      }
      if (this.bimodal) {
        this.allocateBimodal();
      }
    },

    /**
     * @brief Estimate the likelihood of an observation vector.
     *
     * If the distribution is bimodal an the observation is a vector of the size
     * of the input modality, the likelihood is computed only on the
     * distribution for the input modality
     *
     * @param  {array} observation data observation
     * @return {number}
     */
    likelihood(observation) {
      if (this.covarianceDeterminant === 0) {
        throw new Error('Covariance Matrix is not invertible');
      }
      if (this.bimodal && observation.length === this.inputDimension) {
        return this.inputLikelihood(observation);
      }
      if (observation.length !== this.dimension) {
        throw new Error(`GaussianDistribution: observation has wrong dimension. Expected \`${this.dimension}\`, got \`${observation.length}\``);
      }

      let euclideanDistance = 0;
      if (this.covarianceMode === 'full') {
        for (let l = 0; l < this.dimension; l += 1) {
          let tmp = 0;
          for (let k = 0; k < this.dimension; k += 1) {
            tmp += this.inverseCovariance[l * this.dimension + k] * (observation[k] - this.mean[k]);
          }
          euclideanDistance += (observation[l] - this.mean[l]) * tmp;
        }
      } else {
        for (let l = 0; l < this.dimension; l += 1) {
          euclideanDistance += this.inverseCovariance[l] * (observation[l] - this.mean[l]) * (observation[l] - this.mean[l]);
        }
      }

      let p = Math.exp(-0.5 * euclideanDistance) / Math.sqrt(this.covarianceDeterminant * (2 * Math.PI) ** this.dimension);

      if (p < 1e-180 || Number.isNaN(p) || Math.abs(p) === +Infinity) {
        p = 1e-180;
      }

      return p;
    },

    /**
     * Regularize the distribution, given a regularization vector of the same
     * dimension. Regularization adds the vector to the variance of the
     * distribution.
     *
     * @param  {Array} regularization regularization vector
     */
    regularize(regularization) {
      if (this.covarianceMode === 'full') {
        for (let d = 0; d < this.dimension; d += 1) {
          this.covariance[d * this.dimension + d] += regularization[d];
        }
      } else {
        for (let d = 0; d < this.dimension; d += 1) {
          this.covariance[d] += regularization[d];
        }
      }
    },

    /**
     * Update the inverse covariance of the distribution
     * @private
     */
    updateInverseCovariance() {
      if (this.covarianceMode === 'full') {
        const covMatrix = Matrix(this.dimension, this.dimension);

        covMatrix.data = this.covariance.slice();
        const inv = covMatrix.pinv();
        this.covarianceDeterminant = inv.determinant;
        this.inverseCovariance = inv.matrix.data;
      } else {
        // DIAGONAL COVARIANCE
        this.covarianceDeterminant = 1;
        for (let d = 0; d < this.dimension; d += 1) {
          if (this.covariance[d] <= 0) {
            throw new Error('Non-invertible matrix');
          }
          this.inverseCovariance[d] = 1 / this.covariance[d];
          this.covarianceDeterminant *= this.covariance[d];
        }
      }
      if (this.bimodal) {
        this.updateInverseCovarianceBimodal();
      }
    },

    /**
     * Convert to an ellipse allong two dimensions
     *
     * @param  {number} dimension1 first dimension
     * @param  {number} dimension2 second dimension
     * @return {Ellipse}
     */
    toEllipse(dimension1, dimension2) {
      if (dimension1 >= this.dimension || dimension2 >= this.dimension) {
        throw new Error('dimensions out of range');
      }

      const gaussianEllipse = {
        x: 0,
        y: 0,
        width: 0,
        height: 0,
        angle: 0
      };
      gaussianEllipse.x = this.mean[dimension1];
      gaussianEllipse.y = this.mean[dimension2];

      // Represent 2D covariance with square matrix
      // |a b|
      // |b c|
      let a;
      let b;
      let c;
      if (this.covarianceMode === 'full') {
        a = this.covariance[dimension1 * this.dimension + dimension1];
        b = this.covariance[dimension1 * this.dimension + dimension2];
        c = this.covariance[dimension2 * this.dimension + dimension2];
      } else {
        a = this.covariance[dimension1];
        b = 0;
        c = this.covariance[dimension2];
      }

      // Compute Eigen Values to get width, height and angle
      const trace = a + c;
      const determinant = a * c - b * b;
      const eigenVal1 = 0.5 * (trace + Math.sqrt(trace ** 2 - 4 * determinant));
      const eigenVal2 = 0.5 * (trace - Math.sqrt(trace ** 2 - 4 * determinant));
      gaussianEllipse.width = Math.sqrt(5.991 * eigenVal1);
      gaussianEllipse.height = Math.sqrt(5.991 * eigenVal2);
      gaussianEllipse.angle = Math.atan(b / (eigenVal1 - c));
      if (Number.isNaN(gaussianEllipse.angle)) {
        gaussianEllipse.angle = Math.PI / 2;
      }

      return gaussianEllipse;
    },

    /**
     * Modify the distribution along two dimensions given the equivalent values
     * as an Ellipse representation.
     *
     * @param  {Ellipse} gaussianEllipse The Ellipse corresponding to the 2D
     * covariance along the two target dimensions
     * @param  {number} dimension1      first dimension
     * @param  {number} dimension2      second dimension
     */
    fromEllipse(gaussianEllipse, dimension1, dimension2) {
      if (dimension1 >= this.dimension || dimension2 >= this.dimension) {
        throw new Error('dimensions out of range');
      }

      this.mean[dimension1] = gaussianEllipse.x;
      this.mean[dimension2] = gaussianEllipse.y;

      const eigenVal1 = gaussianEllipse.width * gaussianEllipse.width / 5.991;
      const eigenVal2 = gaussianEllipse.height * gaussianEllipse.height / 5.991;
      const tantheta = Math.tan(gaussianEllipse.angle);
      const b = (eigenVal1 - eigenVal2) * tantheta / (tantheta ** 2 + 1);
      const c = eigenVal1 - b / tantheta;
      const a = eigenVal2 + b / tantheta;

      if (this.covarianceMode === 'full') {
        this.covariance[dimension1 * this.dimension + dimension1] = a;
        this.covariance[dimension1 * this.dimension + dimension2] = b;
        this.covariance[dimension2 * this.dimension + dimension1] = b;
        this.covariance[dimension2 * this.dimension + dimension2] = c;
      } else {
        this.covariance[dimension1] = a;
        this.covariance[dimension2] = c;
      }
      this.updateInverseCovariance();
    }
  };

  /**
   * Bimodal Gaussian Distribution Prototype, for Regression purposes
   *
   * @type {Object}
   * @property {boolean} bimodal           Specifies if the distribution is
   * bimodal (for regression use)
   * @property {number}  inputDimension    input dimension
   * @property {number}  outputDimension   output dimension
   * @property {number}  dimension         Total dimension
   * @property {Array}   mean              Distribution mean
   * @property {Array}   covariance        Distribution covariance
   * @property {Array}   inverseCovariance Inverse covariance
   * @property {Array}   inverseCovarianceInput Inverse covariance of the input
   * modality
   *
   * @ignore
   */
  const bimodalGaussianPrototype = /** @lends GaussianDistribution */{
    /**
     * Allocate the distribution
     * @private
     */
    allocateBimodal() {
      if (this.covarianceMode === 'full') {
        this.inverseCovarianceInput = new Array(this.inputDimension ** 2).fill(0);
      } else {
        this.inverseCovarianceInput = new Array(this.inputDimension).fill(0);
      }
    },

    /**
     * Estimate the likelihood of an observation for the input modality only.
     * Called by `likelihood` when relevant.
     * @param  {Array} inputObservation observation (input modality only)
     * @return {number}
     * @private
     */
    inputLikelihood(inputObservation) {
      if (this.covarianceDeterminantInput === 0) {
        throw new Error('Covariance Matrix of input modality is not invertible');
      }

      let euclideanDistance = 0;
      if (this.covarianceMode === 'full') {
        for (let l = 0; l < this.inputDimension; l += 1) {
          let tmp = 0;
          for (let k = 0; k < this.inputDimension; k += 1) {
            tmp += this.inverseCovarianceInput[l * this.inputDimension + k] * (inputObservation[k] - this.mean[k]);
          }
          euclideanDistance += (inputObservation[l] - this.mean[l]) * tmp;
        }
      } else {
        for (let l = 0; l < this.inputDimension; l += 1) {
          euclideanDistance += this.inverseCovariance[l] * (inputObservation[l] - this.mean[l]) * (inputObservation[l] - this.mean[l]);
        }
      }

      let p = Math.exp(-0.5 * euclideanDistance) / Math.sqrt(this.covarianceDeterminantInput * (2 * Math.PI) ** this.inputDimension);

      if (p < 1e-180 || Number.isNaN(p) || Math.abs(p) === +Infinity) p = 1e-180;

      return p;
    },

    /**
     * Estimate the output values associated with an input observation by
     * regression, given the distribution parameters.
     *
     * @todo Clarify the maths here.
     *
     * @param  {Array} inputObservation observation (input modality only)
     * @return {Array} Output values
     */
    regression(inputObservation) {
      const outputDimension = this.dimension - this.inputDimension;
      const prediction = Array(outputDimension).fill(0);

      if (this.covarianceMode === 'full') {
        for (let d = 0; d < outputDimension; d += 1) {
          prediction[d] = this.mean[this.inputDimension + d];
          for (let e = 0; e < this.inputDimension; e += 1) {
            let tmp = 0;
            for (let f = 0; f < this.inputDimension; f += 1) {
              tmp += this.inverseCovarianceInput[e * this.inputDimension + f] * (inputObservation[f] - this.mean[f]);
            }
            prediction[d] += tmp * this.covariance[(d + this.inputDimension) * this.dimension + e];
          }
        }
      } else {
        for (let d = 0; d < outputDimension; d += 1) {
          prediction[d] = this.mean[this.inputDimension + d];
        }
      }
      return prediction;
    },

    /**
     * Update the inverse covariance
     * @private
     */
    updateInverseCovarianceBimodal() {
      if (this.covarianceMode === 'full') {
        const covMatrixInput = Matrix(this.inputDimension, this.inputDimension);
        for (let d1 = 0; d1 < this.inputDimension; d1 += 1) {
          for (let d2 = 0; d2 < this.inputDimension; d2 += 1) {
            covMatrixInput.data[d1 * this.inputDimension + d2] = this.covariance[d1 * this.dimension + d2];
          }
        }
        const invInput = covMatrixInput.pinv();
        this.covarianceDeterminantInput = invInput.determinant;
        this.inverseCovarianceInput = invInput.matrix.data;
      } else {
        // DIAGONAL COVARIANCE
        this.covarianceDeterminantInput = 1;
        for (let d = 0; d < this.inputDimension; d += 1) {
          if (this.covariance[d] <= 0) {
            throw new Error('Non-invertible matrix');
          }
          this.inverseCovarianceInput[d] = 1 / this.covariance[d];
          this.covarianceDeterminantInput *= this.covariance[d];
        }
      }
      this.updateOutputCovariance();
    },

    /**
     * Update the output covariance
     * @private
     */
    updateOutputCovariance() {
      if (this.covarianceMode === 'diagonal') {
        this.outputCovariance = this.covariance.slice(0, this.inputDimension);
        return;
      }

      // CASE: FULL COVARIANCE
      const covMatrixInput = Matrix(this.inputDimension, this.inputDimension);
      for (let d1 = 0; d1 < this.inputDimension; d1 += 1) {
        for (let d2 = 0; d2 < this.inputDimension; d2 += 1) {
          covMatrixInput.data[d1 * this.inputDimension + d2] = this.covariance[d1 * this.dimension + d2];
        }
      }
      const inv = covMatrixInput.pinv();
      const covarianceGS = Matrix(this.inputDimension, this.outputDimension);
      for (let d1 = 0; d1 < this.inputDimension; d1 += 1) {
        for (let d2 = 0; d2 < this.outputDimension; d2 += 1) {
          covarianceGS.data[d1 * this.outputDimension + d2] = this.covariance[d1 * this.dimension + this.inputDimension + d2];
        }
      }
      const covarianceSG = Matrix(this.outputDimension, this.inputDimension);
      for (let d1 = 0; d1 < this.outputDimension; d1 += 1) {
        for (let d2 = 0; d2 < this.inputDimension; d2 += 1) {
          covarianceSG.data[d1 * this.inputDimension + d2] = this.covariance[(this.inputDimension + d1) * this.dimension + d2];
        }
      }
      const tmptmptmp = inv.matrix.product(covarianceGS);
      const covarianceMod = covarianceSG.product(tmptmptmp);
      this.outputCovariance = Array(this.outputDimension ** 2).fill(0);
      for (let d1 = 0; d1 < this.outputDimension; d1 += 1) {
        for (let d2 = 0; d2 < this.outputDimension; d2 += 1) {
          this.outputCovariance[d1 * this.outputDimension + d2] = this.covariance[(this.inputDimension + d1) * this.dimension + this.inputDimension + d2] - covarianceMod.data[d1 * this.outputDimension + d2];
        }
      }
    }
  };

  /**
   * Multivariate Gaussian Distribution factory function.
   * Full covariance, optionally multimodal with support for regression.
   *
   * @function
   * @param {Number} [inputDimension=1]      Dimension of the input modality
   * @param {Number} [outputDimension=0]     Dimension of the output
   * modality (positive for regression, otherwise 0 for recognition).
   * @param {String} [covarianceMode='full'] covariance mode (full vs
   * diagonal)
   * @return {baseGaussianPrototype|bimodalGaussianPrototype}
   *
   * @property {boolean} bimodal           Specifies if the distribution is
   * bimodal (for regression use)
   * @property {number}  inputDimension    input dimension
   * @property {number}  outputDimension   output dimension
   * @property {number}  dimension         Total dimension
   * @property {Array}   mean              Distribution mean
   * @property {Array}   covariance        Distribution covariance
   * @property {Array}   inverseCovariance Inverse covariance
   */
  function GaussianDistribution(inputDimension = 1, outputDimension = 0, covarianceMode = 'full') {
    const bimodal = outputDimension > 0;
    const dimension = inputDimension + outputDimension;
    const proto = bimodal ? Object.assign({}, baseGaussianPrototype, bimodalGaussianPrototype) : baseGaussianPrototype;
    const data = Object.assign({
      bimodal,
      dimension,
      inputDimension,
      outputDimension,
      covarianceMode,
      covarianceDeterminant: 0
    }, bimodal ? { covarianceDeterminantInput: 0 } : {});
    const dist = Object.assign(Object.create(proto), data);
    dist.allocate();
    return dist;
  }

  const trainerPrototype = /** @lends withEMTraining */{
    /**
     * Train the model from the given training set, using the
     * Expectation-Maximisation algorithm.
     *
     * @param  {TrainingSet} trainingSet Training Set
     * @return {Object} Parameters of the trained model
     */
    train(trainingSet) {
      if (!trainingSet || trainingSet.empty()) {
        throw new Error('The training set is empty');
      }

      this.initTraining(trainingSet);

      let logLikelihood = -Infinity;
      let iterations = 0;
      let previousLogLikelihood = logLikelihood;

      while (!this.converged(iterations, logLikelihood, previousLogLikelihood)) {
        previousLogLikelihood = logLikelihood;
        logLikelihood = this.updateTraining(trainingSet);

        const pctChg = 100 * Math.abs((logLikelihood - previousLogLikelihood) / previousLogLikelihood);
        if (Number.isNaN(pctChg) && iterations > 1) {
          throw new Error('An error occured during training');
        }

        iterations += 1;
      }

      this.terminateTraining();
      return this.params;
    },

    /**
     * Return `true` if the training has converged according to the criteria
     * specified at the creation
     *
     * @param  {number} iteration       Current iteration
     * @param  {number} logProb         Current log-likelihood of the training set
     * @param  {number} previousLogProb Previous log-likelihood of the training
     * set
     * @return {boolean}
     *
     * @private
     */
    converged(iteration, logProb, previousLogProb) {
      if (iteration >= this.convergenceCriteria.maxIterations) return true;
      if (this.convergenceCriteria.maxIterations >= this.convergenceCriteria.minIterations) {
        return iteration >= this.convergenceCriteria.maxIterations;
      }
      if (iteration < this.convergenceCriteria.minIterations) return false;
      const percentChange = 100 * Math.abs((logProb - previousLogProb) / logProb);
      return percentChange <= this.convergenceCriteria.percentChange;
    }
  };

  /**
   * Add ABSTRACT training capabilities to a model for which the training process
   * use the Expectation-Maximisation (EM) algorithm. This is used in particular
   * for training GMMs and HMMs.
   *
   * The final instance needs to implement `initTraining`, `updateTraining` and
   * `terminateTraining` methods. `updateTraining` will be called until the
   * convergence criteria are met. Convergence depends on
   * - A minimum number of iterations
   * - A maximum number of iterations
   * - A threshold on the relative change of the log-likelihood of the training
   * data between successive iterations.
   *
   * @todo details
   *
   * @param  {Object} [o]                   Source object
   * @param  {Object} [convergenceCriteria] Set of convergence criteria
   * @param  {number} [convergenceCriteria.percentChange=1e-3] Threshold in % of
   * the relative change of the log-likelihood, under which the training stops.
   * @param  {number} [convergenceCriteria.minIterations=5]    minimum number of iterations
   * @param  {number} [convergenceCriteria.maxIterations=100]  maximum number of iterations
   * @return {Object}
   */
  function withEMTraining(o, convergenceCriteria = {
    percentChange: 1e-3,
    minIterations: 5,
    maxIterations: 100
  }) {
    return Object.assign(o, trainerPrototype, { convergenceCriteria });
  }

  /**
   * GMM Base prototype
   * @type {Object}
   * @ignore
   */
  const gmmBasePrototype = /** @lends withGMMBase */{
    /**
     * Allocate the training variables
     * @private
     */
    allocate() {
      this.params.components = Array.from(Array(this.params.gaussians), () => new GaussianDistribution(this.params.inputDimension, this.params.outputDimension, this.params.covarianceMode));
      this.params.mixtureCoeffs = Array(this.params.gaussians).fill(0);
      this.beta = new Array(this.params.gaussians).fill(0);
    },

    /**
     * Compute the likelihood of an observation given the GMM's parameters
     * @param  {Array<Number>} observation Observation vector
     * @return {Number}
     */
    likelihood(observation) {
      let likelihood = 0;
      for (let c = 0; c < this.params.gaussians; c += 1) {
        this.beta[c] = this.componentLikelihood(observation, c);
        likelihood += this.beta[c];
      }
      for (let c = 0; c < this.params.gaussians; c += 1) {
        this.beta[c] /= likelihood;
      }

      return likelihood;
    },

    /**
     * Compute the likelihood of an observation for a single component
     * @param  {Array<Number>} observation Observation vector
     * @param  {Number} mixtureComponent Component index
     * @return {Number}
     * @private
     */
    componentLikelihood(observation, mixtureComponent) {
      if (mixtureComponent >= this.params.gaussians) {
        throw new Error('The index of the Gaussian Mixture Component is out of bounds');
      }
      return this.params.mixtureCoeffs[mixtureComponent] * this.params.components[mixtureComponent].likelihood(observation);
    },

    /**
     * Update the inverse covariance of each Gaussian component
     * @private
     */
    updateInverseCovariances() {
      this.params.components.forEach(c => {
        c.updateInverseCovariance();
      });
      try {
        this.params.components.forEach(c => {
          c.updateInverseCovariance();
        });
      } catch (e) {
        throw new Error('Matrix inversion error: varianceoffset must be too small');
      }
    },

    /**
     * Normalize the mixing coefficients of the Gaussian mixture
     * @private
     */
    normalizeMixtureCoeffs() {
      let normConst = 0;
      for (let c = 0; c < this.params.gaussians; c += 1) {
        normConst += this.params.mixtureCoeffs[c];
      }
      if (normConst > 0) {
        for (let c = 0; c < this.params.gaussians; c += 1) {
          this.params.mixtureCoeffs[c] /= normConst;
        }
      } else {
        for (let c = 0; c < this.params.gaussians; c += 1) {
          this.params.mixtureCoeffs[c] = 1 / this.params.gaussians;
        }
      }
    },

    /**
     * Regularize the covariances
     * @private
     */
    regularize() {
      this.params.components.forEach(c => {
        c.regularize(this.currentRegularization);
      });
    }
  };

  /**
   * Bimodal (regression) GMM Prototype
   * @type {Object}
   * @ignore
   */
  const gmmBimodalPrototype = /** @lends withGMMBase */{
    /**
     * Estimate the output values corresponding to the input observation, by
     * regression given the GMM's parameters. This method is called Gaussian
     * Mixture Regression (GMR).
     *
     * @param  {Array<Number>} inputObservation Observation on the input modality
     * @return {Array<Number>} Output values (length = outputDimension)
     */
    regression(inputObservation) {
      this.results.outputValues = Array(this.params.outputDimension).fill(0);
      this.results.outputCovariance = Array(this.params.covarianceMode === 'full' ? this.params.outputDimension ** 2 : this.params.outputDimension).fill(0);
      let tmpOutputValues;

      for (let c = 0; c < this.params.gaussians; c += 1) {
        tmpOutputValues = this.params.components[c].regression(inputObservation);
        for (let d = 0; d < this.params.outputDimension; d += 1) {
          this.results.outputValues[d] += this.beta[c] * tmpOutputValues[d];
          if (this.params.covarianceMode === 'full') {
            for (let d2 = 0; d2 < this.params.outputDimension; d2 += 1) {
              this.results.outputCovariance[d * this.params.outputDimension + d2] += this.beta[c] ** 2 * this.params.components[c].outputCovariance[d * this.params.outputDimension + d2];
            }
          } else {
            this.results.outputCovariance[d] += this.beta[c] ** 2 * this.params.components[c].outputCovariance[d];
          }
        }
      }
      return this.results.outputValues;
    }
  };

  /**
   * Add basic GMM capabilities to a single-class model. This enables the
   * computation of the likelihoods and regression operations common to
   * training and prediction
   *
   * @see withGMMTraining
   * @see withGMMPrediction
   *
   * @param  {ModelBase} o Source Model
   * @return {GMMBaseModel}
   *
   * @throws {Error} is o is not a ModelBase
   */
  function withGMMBase(o) {
    if (!isBaseModel(o)) {
      throw new Error('The base object must include a standard set of parameters (`params` key), @see `ModelBase`.');
    }
    return Object.assign(o, gmmBasePrototype, o.params.bimodal ? gmmBimodalPrototype : {});
  }

  /**
   * GMM Training Prototype
   * @type {Object}
   * @ignore
   */
  const gmmTrainerPrototype = /** @lends withGMMTraining */{
    /**
     * Initialize the EM Training process
     * @param  {TrainingSet} trainingSet Training set
     */
    initTraining(trainingSet) {
      this.allocate();
      this.initParametersToDefault(trainingSet.standardDeviation());
      this.initMeansWithKMeans(trainingSet);
      this.initCovariances(trainingSet);
      this.regularize();
      this.updateInverseCovariances();
    },

    /**
     * Initialize the model parameters to their default values
     * @param  {Array<Number>} dataStddev Standard deviation of the training data
     * @private
     */
    initParametersToDefault(dataStddev) {
      let normCoeffs = 0;
      this.currentRegularization = dataStddev.map(std => Math.max(this.params.regularization.absolute, this.params.regularization.relative * std));
      for (let c = 0; c < this.params.gaussians; c += 1) {
        if (this.params.covarianceMode === 'full') {
          this.params.components[c].covariance = Array(this.params.dimension ** 2).fill(this.params.regularization.absolute / 2);
        } else {
          this.params.components[c].covariance = Array(this.params.dimension).fill(0);
        }
        this.params.components[c].regularize(this.currentRegularization);
        this.params.mixtureCoeffs[c] = 1 / this.params.gaussians;
        normCoeffs += this.params.mixtureCoeffs[c];
      }
      for (let c = 0; c < this.params.gaussians; c += 1) {
        this.params.mixtureCoeffs[c] /= normCoeffs;
      }
    },

    /**
     * Initialize the means of the model using a K-Means algorithm
     *
     * @see withKMeansTraining
     *
     * @param  {TrainingSet} trainingSet training set
     * @private
     */
    initMeansWithKMeans(trainingSet) {
      if (!trainingSet || trainingSet.empty()) return;
      const kmeans = withKMeansTraining(ModelBase({
        inputDimension: this.params.inputDimension,
        outputDimension: this.params.outputDimension
      }), this.params.gaussians, { initialization: 'data' });
      const kmeansParams = kmeans.train(trainingSet);
      for (let c = 0; c < this.params.gaussians; c += 1) {
        this.params.components[c].mean = kmeansParams.centers[c];
      }
    },

    /**
     * Initialize the covariances of the model from the training set
     *
     * @param  {TrainingSet} trainingSet training set
     * @private
     */
    initCovariances(trainingSet) {
      // TODO: simplify with covariance symmetricity
      // TODO: If Kmeans, covariances from cluster members
      if (!trainingSet || trainingSet.empty()) return;

      for (let n = 0; n < this.params.gaussians; n += 1) {
        this.params.components[n].covariance = Array(this.params.covarianceMode === 'full' ? this.params.dimension ** 2 : this.params.dimension).fill(0);
      }

      const gmeans = Array(this.params.gaussians * this.params.dimension).fill(0);
      const factor = Array(this.params.gaussians).fill(0);
      trainingSet.forEach(phrase => {
        const step = Math.floor(phrase.length / this.params.gaussians);
        let offset = 0;
        for (let n = 0; n < this.params.gaussians; n += 1) {
          for (let t = 0; t < step; t += 1) {
            for (let d1 = 0; d1 < this.params.dimension; d1 += 1) {
              gmeans[n * this.params.dimension + d1] += phrase.get(offset + t, d1);
              if (this.params.covarianceMode === 'full') {
                for (let d2 = 0; d2 < this.params.dimension; d2 += 1) {
                  this.params.components[n].covariance[d1 * this.params.dimension + d2] += phrase.get(offset + t, d1) * phrase.get(offset + t, d2);
                }
              } else {
                this.params.components[n].covariance[d1] += phrase.get(offset + t, d1) ** 2;
              }
            }
          }
          offset += step;
          factor[n] += step;
        }
      });

      for (let n = 0; n < this.params.gaussians; n += 1) {
        for (let d1 = 0; d1 < this.params.dimension; d1 += 1) {
          gmeans[n * this.params.dimension + d1] /= factor[n];
          if (this.params.covarianceMode === 'full') {
            for (let d2 = 0; d2 < this.params.dimension; d2 += 1) {
              this.params.components[n].covariance[d1 * this.params.dimension + d2] /= factor[n];
            }
          } else {
            this.params.components[n].covariance[d1] /= factor[n];
          }
        }
      }

      for (let n = 0; n < this.params.gaussians; n += 1) {
        for (let d1 = 0; d1 < this.params.dimension; d1 += 1) {
          if (this.params.covarianceMode === 'full') {
            for (let d2 = 0; d2 < this.params.dimension; d2 += 1) {
              this.params.components[n].covariance[d1 * this.params.dimension + d2] -= gmeans[n * this.params.dimension + d1] * gmeans[n * this.params.dimension + d2];
            }
          } else {
            this.params.components[n].covariance[d1] -= gmeans[n * this.params.dimension + d1] ** 2;
          }
        }
      }
    },

    /**
     * Update the EM Training process (1 EM iteration).
     * @param  TrainingSet trainingSet training set
     */
    updateTraining(trainingSet) {
      let logProb = 0;
      let totalLength = 0;
      trainingSet.forEach(phrase => {
        totalLength += phrase.length;
      });
      const phraseIndices = Object.keys(trainingSet.phrases);

      const p = Array.from(Array(this.params.gaussians), () => new Array(totalLength).fill(0));
      const E = Array(this.params.gaussians).fill(0);
      let tbase = 0;

      trainingSet.forEach(phrase => {
        for (let t = 0; t < phrase.length; t += 1) {
          let normConst = 0;
          for (let c = 0; c < this.params.gaussians; c += 1) {
            p[c][tbase + t] = this.componentLikelihood(phrase.getFrame(t), c);

            if (p[c][tbase + t] === 0 || Number.isNaN(p[c][tbase + t]) || p[c][tbase + t] === +Infinity) {
              p[c][tbase + t] = 1e-100;
            }
            normConst += p[c][tbase + t];
          }
          for (let c = 0; c < this.params.gaussians; c += 1) {
            p[c][tbase + t] /= normConst;
            E[c] += p[c][tbase + t];
          }
          logProb += Math.log(normConst);
        }
        tbase += phrase.length;
      });

      // Estimate Mixture coefficients
      for (let c = 0; c < this.params.gaussians; c += 1) {
        this.params.mixtureCoeffs[c] = E[c] / totalLength;
      }

      // Estimate means
      for (let c = 0; c < this.params.gaussians; c += 1) {
        for (let d = 0; d < this.params.dimension; d += 1) {
          this.params.components[c].mean[d] = 0;
          tbase = 0;
          for (let pix = 0; pix < phraseIndices.length; pix += 1) {
            const phrase = trainingSet.phrases[phraseIndices[pix]];
            for (let t = 0; t < phrase.length; t += 1) {
              this.params.components[c].mean[d] += p[c][tbase + t] * phrase.get(t, d);
            }
            tbase += phrase.length;
          }
          this.params.components[c].mean[d] /= E[c];
        }
      }

      // estimate covariances
      if (this.params.covarianceMode === 'full') {
        for (let c = 0; c < this.params.gaussians; c += 1) {
          for (let d1 = 0; d1 < this.params.dimension; d1 += 1) {
            for (let d2 = d1; d2 < this.params.dimension; d2 += 1) {
              this.params.components[c].covariance[d1 * this.params.dimension + d2] = 0;
              tbase = 0;
              for (let pix = 0; pix < phraseIndices.length; pix += 1) {
                const phrase = trainingSet.phrases[phraseIndices[pix]];
                for (let t = 0; t < phrase.length; t += 1) {
                  this.params.components[c].covariance[d1 * this.params.dimension + d2] += p[c][tbase + t] * (phrase.get(t, d1) - this.params.components[c].mean[d1]) * (phrase.get(t, d2) - this.params.components[c].mean[d2]);
                }
                tbase += phrase.length;
              }
              this.params.components[c].covariance[d1 * this.params.dimension + d2] /= E[c];
              if (d1 !== d2) {
                this.params.components[c].covariance[d2 * this.params.dimension + d1] = this.params.components[c].covariance[d1 * this.params.dimension + d2];
              }
            }
          }
        }
      } else {
        for (let c = 0; c < this.params.gaussians; c += 1) {
          for (let d1 = 0; d1 < this.params.dimension; d1 += 1) {
            this.params.components[c].covariance[d1] = 0;
            tbase = 0;
            for (let pix = 0; pix < phraseIndices.length; pix += 1) {
              const phrase = trainingSet.phrases[phraseIndices[pix]];
              for (let t = 0; t < phrase.length; t += 1) {
                const value = phrase.get(t, d1) - this.params.components[c].mean[d1];
                this.params.components[c].covariance[d1] += p[c][tbase + t] * value * value;
              }
              tbase += phrase.length;
            }
            this.params.components[c].covariance[d1] /= E[c];
          }
        }
      }

      this.regularize();
      this.updateInverseCovariances();

      return logProb;
    },

    /**
     * Terminate the EM Training process
     */
    terminateTraining() {}
  };

  /**
   * Add GMM Training capabilities to a GMM Model
   * @param  {GMMBase} o               Source GMM Model
   * @param  {Number} [gaussians=1]    Number of Gaussian components
   * @param  {Object} [regularization] Regularization parameters
   * @param  {Number} [regularization.absolute=1e-3] Absolute regularization
   * @param  {Number} [regularization.relative=1e-2] Relative Regularization
   (relative to the training set's variance along each dimension)
   * @param  {String} [covarianceMode='full'] Covariance mode ('full' or diagonal)
   * @return {BMMBase}
   */
  function withGMMTraining(o, gaussians = 1, regularization = { absolute: 1e-3, relative: 1e-2 }, covarianceMode = 'full') {
    if (!Object.keys(o).includes('params')) {
      throw new Error('The base object must include a standard set of parameters (`params` key), @see `ModelBase`.');
    }
    return Object.assign(o, gmmTrainerPrototype, {
      params: _extends({}, o.params, {
        gaussians,
        regularization,
        covarianceMode
      })
    });
  }

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
    }
  };

  /**
   * Create an abstract Multiclass Model
   * @param       {number]} inputDimension  input dimension
   * @param       {number]} outputDimension output dimension
   * @param       {Object} parameters       additional parameters to copy
   * @function
   */
  function MulticlassModelBase(_ref) {
    let {
      inputDimension,
      outputDimension
    } = _ref,
        parameters = objectWithoutProperties(_ref, ['inputDimension', 'outputDimension']);

    return Object.assign(ModelBase(_extends({ inputDimension, outputDimension }, parameters)), MulticlassBasePrototype);
  }

  /**
   * Add multiclass training capabilities to a model. It takes as argument
   * the training function called to train each class of the training set.
   *
   * @param  {MulticlassModelBase} o Source model
   * @param  {Function}  trainingFunction Training function for a single class
   * @return {MulticlassModelBase}
   */
  function withMulticlassTraining(o, trainingFunction) {
    return Object.assign(o,
    /** @lends withMulticlassTraining */{
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
          labels.forEach(l => {
            if (!this.includes(l)) {
              throw new Error(`Class labeled ${l} does not exist`);
            }
          });
        }

        this.params.classes = {};
        const labs = labels || trainingSet.labels();
        labs.forEach(label => {
          const ts = trainingSet.getPhrasesOfClass(label);
          // console.log(ts);
          this.params.classes[label] = trainingFunction(ts);
        });
        return this.params;
      }
    });
  }

  /**
   * Circular Buffer prototype
   *
   * @property {number}  capacity Buffer capacity
   * @property {number}  length Current buffer length
   * @property {boolean} full Specifies if the buffer is full
   *
   * @ignore
   */
  const circularBufferPrototype = /** @lends CircularBuffer */{
    /**
     * Clear the buffer contents
     */
    clear() {
      this.length = 0;
      this.index = 0;
      this.full = false;
      this.buffer = [];
    },

    /**
     * Push a value to the buffer
     * @param  {*} value data value (any type)
     */
    push(value) {
      if (this.full) {
        this.buffer[this.index] = value;
        this.index = (this.index + 1) % this.capacity;
      } else {
        this.buffer.push(value);
        this.length += 1;
        this.full = this.length === this.capacity;
      }
    },

    /**
     * Get the value at a given index
     * @param  {number} idx data index
     * @return {anything}   value at index
     */
    get(idx) {
      return this.buffer[(idx + this.index) % this.capacity];
    },

    /**
     * Fill the buffer with a constant value
     * @param  {*} value data value (any type)
     */
    fill(value) {
      this.length = this.capacity;
      this.index = 0;
      this.full = true;
      this.buffer = Array(this.capacity).fill(value);
    },

    /**
     * Iterate over the buffer's data
     * @param  {Function} callback Callback function
     * (@see Array.prototype.forEach).
     */
    forEach(callback) {
      for (let i = 0; i < this.length; i += 1) {
        callback(this.buffer[(i + this.index) % this.capacity], i);
      }
    },

    /**
     * Get an array of the buffer current values (ordered)
     * @return {Array} Buffer contents
     */
    values() {
      return this.buffer.slice(this.index).concat(this.buffer.slice(0, this.index));
    }
  };

  /**
   * Circular Buffer Data Structure (any data type)
   * @param  {number} capacity Buffer capacity
   * @return {circularBufferPrototype}
   * @function
   *
   * @property {number}  capacity Buffer capacity
   * @property {number}  length Current buffer length
   * @property {boolean} full Specifies if the buffer is full
   */
  function CircularBuffer(capacity) {
    const buffer = Object.create(circularBufferPrototype);
    buffer.capacity = capacity;
    buffer.clear();
    return buffer;
  }

  /**
   * Prototype for models with prediction capabilities
   * @param  {Boolean} bimodal Specifies whether the model is bimodal
   * @return {Object}
   * @ignore
   */
  const predictionBasePrototype = bimodal => ( /** @lends withAbtractPrediction */{
    /**
     * Likelihood Buffer
     * @type {CircularBuffer}
     * @private
     */
    likelihoodBuffer: CircularBuffer(1),

    /**
     * Likelihood Window (used to smooth the log-likelihoods over several frames)
     * @param {Number} [lw] Size (in frames) of the likelihood smoothing window
     */
    setLikelihoodWindow(lw) {
      this.likelihoodWindow = lw;
      this.likelihoodBuffer = CircularBuffer(lw);
    },

    /**
     * Reset the prediction process
     * @return {Modelbase} the model
     */
    reset() {
      this.likelihoodBuffer.clear();
      return this;
    },

    /**
     * Update the predictions with a new observation
     * @param  {Array<Number>} observation Observation vector
     * @return {Object} Prediction results
     *
     * @todo document results data structure
     */
    predict(observation) {
      const likelihood = this.likelihood(observation);
      if (bimodal) {
        this.regression(observation);
      }
      this.updateResults(likelihood);
      return this.results;
    },

    /**
     * Update the prediction results
     * @param  {Number} instantLikelihood Instantaneous likelihood
     * @private
     */
    updateResults(instantLikelihood) {
      this.results.instantLikelihood = instantLikelihood;
      this.likelihoodBuffer.push(Math.log(instantLikelihood));
      this.results.logLikelihood = 0;
      const bufSize = this.likelihoodBuffer.length;
      for (let i = 0; i < bufSize; i += 1) {
        this.results.logLikelihood += this.likelihoodBuffer.get(i);
      }
      this.results.logLikelihood /= bufSize;
    }
  });

  /**
   * Add ABSTRACT prediction capabilities to an existing model
   * @param  {Modelbase} o                 Source model
   * @param  {Number} [likelihoodWindow=1] Size of the likelihood smoothing window
   * @return {Modelbase}
   */
  function withAbtractPrediction(o, likelihoodWindow = 1) {
    if (!isBaseModel(o)) {
      throw new Error('The base object must include a standard set of parameters (`params` key), @see `ModelBase`.');
    }
    const results = Object.assign({ instantLikelihood: 0, logLikelihood: 0 }, o.params.bimodal ? { outputValues: [], outputCovariance: [] } : {});
    return Object.assign(o, predictionBasePrototype(o.params.bimodal), { results, likelihoodBuffer: CircularBuffer(likelihoodWindow) });
  }

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
  function validateParameters(model, specification, values) {
    const parameters = Object.assign({}, values);
    Object.keys(specification).forEach(attr => {
      const spec = specification[attr];

      // Check for required parameters
      if (spec.required && !Object.keys(values).includes(attr)) {
        throw new Error(`Stream parameter '${attr}' is required for model '${model}'.`);
      }

      // Check the validity of the input parameters
      checkSpec(model, attr, spec.check, values[attr]);

      parameters[attr] = spec.transform ? spec.transform(values[attr]) : values[attr];
    });
    return parameters;
  }

  const gmmParameterSpec = gaussians => ({
    gaussians: {
      required: true,
      check: { min: 1 }
    },
    regularization: {
      required: true,
      check: ({ absolute, relative }) => absolute && relative && absolute > 0 && relative > 0
    },
    covarianceMode: {
      required: true,
      check: ['full', 'diagonal']
    },
    mixtureCoeffs: {
      required: true,
      check: m => m.length === gaussians
    },
    components: {
      required: true,
      check: c => c.length === gaussians
    }
  });

  /**
   * Add GMM prediction capabilities to a single-class model. Mostly, this checks
   * the validity of the model parameters
   *
   * @todo validate gaussian components
   *
   * @param  {GMMBaseModel} o Source Model
   * @return {GMMBaseModel}
   *
   * @throws {Error} is o is not a ModelBase
   */
  function withGMMPrediction(o) {
    if (!isBaseModel(o)) {
      throw new Error('The base object must include a standard set of parameters (`params` key), @see `ModelBase`.');
    }
    validateParameters('GMM', gmmParameterSpec(o.params.gaussians), o.params);
    return Object.assign(o, { beta: new Array(o.params.gaussians).fill(0) });
  }

  /**
   * Multiclass prediction mixin
   * @type {Object}
   * @ignore
   */
  const MulticlassPredictionBasePrototype = /** @lends withMulticlassPrediction */{
    /**
     * Likelihood Window (used to smooth the log-likelihoods over several frames)
     * @return {Number}
     */
    getLikelihoodWindow() {
      return this.likelihoodWindow;
    },

    /**
     * Likelihood Window (used to smooth the log-likelihoods over several frames)
     * @param {Number} [lw] Size (in frames) of the likelihood smoothing window
     */
    setLikelihoodWindow(lw) {
      this.likelihoodWindow = lw;
      Object.keys(this.models).forEach(label => {
        this.models[label].setLikelihoodWindow(lw);
      });
    },

    /**
     * Reset the prediction process. This is particularly important for temporal
     * models such as HMMs, that depends on previous observations.
     */
    reset() {
      Object.values(this.models).forEach(m => m.reset());
      this.results = {
        labels: [],
        instantLikelihoods: [],
        smoothedLikelihoods: [],
        smoothedLogLikelihoods: [],
        smoothedNormalizedLikelihoods: [],
        likeliest: null,
        classes: {}
      };
      if (this.params.bimodal) {
        this.resetBimodal();
      }
    },

    /**
     * Make a prediction from a new observation (updates the results member)
     * @param  {Array<Number>} observation Observation vector
     */
    predict(observation) {
      Object.values(this.models).forEach(m => m.predict(observation));
      this.updateResults();
    },

    updateResults() {
      const labs = Object.keys(this.models).sort();
      this.results.labels = labs;
      let normInstant = 0;
      let normSmoothed = 0;
      let maxLogLikelihood = -Infinity;
      this.results.classes = labs.map((lab, i) => {
        this.results.instantLikelihoods[i] = this.models[lab].results.instantLikelihood;
        this.results.smoothedLogLikelihoods[i] = this.models[lab].results.logLikelihood;
        this.results.smoothedLikelihoods[i] = Math.exp(this.results.smoothedLogLikelihoods[i]);
        normInstant += this.results.instantLikelihoods[i];
        normSmoothed += this.results.smoothedLikelihoods[i];
        if (this.results.smoothedLogLikelihoods[i] > maxLogLikelihood) {
          maxLogLikelihood = this.results.smoothedLogLikelihoods[i];
          this.results.likeliest = lab;
        }
        return { [lab]: this.models[lab].results };
      }).reduce((o, x) => _extends({}, o, x), {});
      this.results.smoothedNormalizedLikelihoods = this.results.smoothedLikelihoods.map(x => x / normSmoothed);
      this.results.instantNormalizedLikelihoods = this.results.instantLikelihoods.map(x => x / normInstant);
      if (this.params.bimodal) {
        this.updateRegressionResults();
      }
    }
  };

  const MulticlassPredictionBimodalPrototype = {
    resetBimodal() {
      this.results.outputValues = [];
      this.results.outputCovariance = [];
    },

    updateRegressionResults() {
      if (this.params.multiClassRegressionEstimator === 'likeliest') {
        this.results.outputValues = this.models[this.results.likeliest].results.outputValues;
        this.results.outputCovariance = this.models[this.results.likeliest].results.outputCovariance;
      } else if (this.params.multiClassRegressionEstimator === 'mixture') {
        this.results.outputValues = Array(this.outputDimension).fill(0);
        this.results.outputCovariance = Array(this.outputDimension ** (this.configuration.covarianceMode === 'full' ? 2 : 1)).fill(0);
        this.results.labels.forEach(lab => {
          this.results.outputValues.map((x, i) => x + this.results.smoothedNormalizedLikelihoods[i] * this.models[lab].results.outputValues[i]);
          this.results.outputCovariance.map((x, i) => x + this.results.smoothedNormalizedLikelihoods[i] * this.models[lab].results.outputCovariance[i]);
        });
      } else {
        throw new Error('Unknown regression estimator, use `likeliest` or `mixture`');
      }
    }
  };

  /**
   * Add multiclass prediction capabilities to a multiclass model
   * @param  {MulticlassModelBase} o Source model
   * @param  {String} [multiClassRegressionEstimator='likeliest'] Type of
   * regression estimator:
   * - `likeliest` selects the output values from the likeliest class
   * - `mixture` computes the output values as the weighted sum of the
   * contributions of each class, weighed by their normalized likelihood
   * @return {MulticlassPredictionBasePrototype}
   * @function
   */
  function withMulticlassPrediction(o, multiClassRegressionEstimator = 'likeliest') {
    if (!isBaseModel(o)) {
      throw new Error('The base object must include a standard set of parameters (`params` key), @see `ModelBase`.');
    }
    const m = Object.assign(o, MulticlassPredictionBasePrototype, o.params.bimodal ? MulticlassPredictionBimodalPrototype : {});
    m.params.multiClassRegressionEstimator = multiClassRegressionEstimator;
    return m;
  }

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
  function trainGMM(trainingSet, configuration, convergenceCriteria = undefined) {
    const { inputDimension, outputDimension } = trainingSet;
    const { gaussians, regularization, covarianceMode } = configuration;
    const model = withGMMTraining(withEMTraining(withGMMBase(ModelBase(_extends({
      inputDimension,
      outputDimension
    }, configuration))), convergenceCriteria), gaussians, regularization, covarianceMode);
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
  function trainMulticlassGMM(trainingSet, configuration, convergenceCriteria = undefined) {
    const { inputDimension, outputDimension } = trainingSet;
    const model = withMulticlassTraining(MulticlassModelBase(_extends({ inputDimension, outputDimension }, configuration)), ts => trainGMM(ts, configuration, convergenceCriteria));
    return model.train(trainingSet);
  }

  /**
   * Create a GMM Predictor from a full set of parameters (generated by trainGMM).
   * @param       {Object} params                       Model parameters
   * @param       {number} [likelihoodWindow=undefined] Likelihoow window size
   * @function
   */
  function GMMPredictor(params, likelihoodWindow = undefined) {
    const model = withGMMPrediction(withAbtractPrediction(withGMMBase(ModelBase(params)), likelihoodWindow));
    params.components.forEach((c, i) => {
      model.params.components[i] = Object.assign(GaussianDistribution(params.inputDimension, params.outputDimension, params.covarianceMode), c);
    });
    model.reset();
    return model;
  }

  /**
   * Create a Multiclass GMM Predictor from a full set of parameters
   * (generated by trainMulticlassGMM).
   * @param       {Object} params                       Model parameters
   * @param       {number} [likelihoodWindow=undefined] Likelihoow window size
   * @function
   */
  function MulticlassGMMPredictor(params, likelihoodWindow = undefined) {
    const model = withMulticlassPrediction(MulticlassModelBase(params));
    model.models = {};
    Object.keys(params.classes).forEach(label => {
      model.models[label] = GMMPredictor(params.classes[label], likelihoodWindow);
    });
    model.reset();
    return model;
  }

  //
  // TODO: hierarchical + exit probabilities methods.
  //

  /**
   * HMM Base prototype
   * @type {Object}
   * @ignore
   */
  const hmmBasePrototype = /** @lends withHMMBase */{
    /**
     * Specifies if the forward algorithm has been initialized
     * @type {Boolean}
     * @private
     */
    forwardInitialized: false,

    /**
     * Specifies if the containing multiclass model is isHierarchical
     * @todo check that
     * @type {Boolean}
     * @private
     */
    isHierarchical: false,

    /**
     * Initialize the forward algorithm (See rabiner, 1989)
     * @param  {Array<Number>} observation Observation vector
     * @return {Number}                    `ct` (inverse likelihood)
     */
    initializeForwardAlgorithm(observation) {
      let normConst = 0;
      if (this.params.transitionMode === 'ergodic') {
        for (let i = 0; i < this.params.states; i += 1) {
          this.alpha[i] = this.params.prior[i] * this.params.xStates[i].likelihood(observation);
          normConst += this.alpha[i];
        }
      } else {
        this.alpha = new Array(this.params.states).fill(0);
        this.alpha[0] = this.params.xStates[0].likelihood(observation);
        normConst += this.alpha[0];
      }
      this.forwardInitialized = true;
      if (normConst > 0) {
        for (let i = 0; i < this.params.states; i += 1) {
          this.alpha[i] /= normConst;
        }
        return 1 / normConst;
      }
      for (let j = 0; j < this.params.states; j += 1) {
        this.alpha[j] = 1 / this.params.states;
      }
      return 1;
    },

    /**
     * Update the forward algorithm (See rabiner, 1989)
     * @param  {Array<Number>} observation Observation vector
     * @return {Number}                    `ct` (inverse likelihood)
     */
    updateForwardAlgorithm(observation) {
      let normConst = 0;
      this.previousAlpha = this.alpha.slice();
      for (let j = 0; j < this.params.states; j += 1) {
        this.alpha[j] = 0;
        if (this.params.transitionMode === 'ergodic') {
          for (let i = 0; i < this.params.states; i += 1) {
            this.alpha[j] += this.previousAlpha[i] * this.params.transition[i][j];
          }
        } else {
          this.alpha[j] += this.previousAlpha[j] * this.params.transition[j * 2];
          if (j > 0) {
            this.alpha[j] += this.previousAlpha[j - 1] * this.params.transition[(j - 1) * 2 + 1];
          } else {
            this.alpha[0] += this.previousAlpha[this.params.states - 1] * this.params.transition[this.params.states * 2 - 1];
          }
        }
        this.alpha[j] *= this.params.xStates[j].likelihood(observation);
        normConst += this.alpha[j];
      }
      if (normConst > 1e-300) {
        for (let j = 0; j < this.params.states; j += 1) {
          this.alpha[j] /= normConst;
        }
        return 1 / normConst;
      }
      return 0;
    }
  };

  /**
   * Add basic HMM capabilities to a single-class model. This enables the
   * computation of the likelihoods and regression operations common to
   * training and prediction
   *
   * @see withHMMTraining
   * @see withHMMPrediction
   *
   * @param  {ModelBase} o Source Model
   * @return {HMMBaseModel}
   *
   * @throws {Error} is o is not a ModelBase
   */
  function withHMMBase(o) {
    if (!isBaseModel(o)) {
      throw new Error('The base object must include a standard set of parameters (`params` key), @see `ModelBase`.');
    }
    return Object.assign(o, hmmBasePrototype);
  }

  const TRANSITION_REGULARIZATION = 1e-5;

  /**
   * HMM Training Prototype
   * @type {Object}
   * @ignore
   */
  const hmmTrainerPrototype = /** @lends withHMMTraining */{
    /**
     * Initialize the EM Training process
     * @param  {TrainingSet} trainingSet Training set
     */
    initTraining(trainingSet) {
      if (!trainingSet || trainingSet.empty()) return;

      this.allocate(trainingSet);
      this.initParametersToDefault(trainingSet.standardDeviation());
      if (this.params.gaussians > 1) {
        this.initMeansCovariancesWithGMMEM(trainingSet);
      } else {
        this.initMeansWithAllPhrases(trainingSet);
        this.initCovariancesFullyObserved(trainingSet);
      }
    },

    /**
     * Allocate the model's parameters and training variables
     * @param  {TrainingSet} trainingSet The training set
     * @private
     */
    allocate(trainingSet) {
      const {
        inputDimension,
        outputDimension,
        gaussians,
        regularization,
        covarianceMode
      } = this.params;
      this.params.xStates = Array.from(new Array(this.params.states), () => withGMMBase(ModelBase({
        inputDimension,
        outputDimension,
        gaussians,
        regularization,
        covarianceMode
      })));
      this.params.xStates.forEach(s => s.allocate());
      this.alpha = new Array(this.params.states).fill(0);
      this.previousAlpha = new Array(this.params.states).fill(0);
      this.beta = new Array(this.params.states).fill(0);
      this.previousBeta = new Array(this.params.states).fill(0);

      // Initialize Algorithm variables
      // ---------------------------------------
      const nbPhrases = trainingSet.size();
      this.gammaSequence = new Array(nbPhrases).fill(null);
      this.epsilonSequence = new Array(nbPhrases).fill(null);
      this.gammaSequenceperMixture = new Array(nbPhrases).fill(null);
      let i = 0;
      trainingSet.forEach(phrase => {
        const T = phrase.length;
        this.gammaSequence[i] = Array.from(new Array(T), () => new Array(this.params.states).fill(0));
        if (this.params.transitionMode === 'ergodic') {
          this.epsilonSequence[i] = Array.from(new Array(T), () => Array.from(new Array(this.params.states), () => new Array(this.params.states).fill(0)));
        } else {
          this.epsilonSequence[i] = Array.from(new Array(T), () => new Array(this.params.states * 2).fill(0));
        }
        this.gammaSequenceperMixture[i] = new Array(this.params.gaussians).fill(0);
        for (let c = 0; c < this.params.gaussians; c += 1) {
          this.gammaSequenceperMixture[i][c] = Array.from(new Array(T), () => new Array(this.params.states).fill(0));
        }
        i += 1;
      });

      this.gammaSum = new Array(this.params.states).fill(0);
      this.gammaSumPerMixture = new Array(this.params.states * this.params.gaussians).fill(0);
    },

    /**
     * Update the EM Training process (1 EM iteration).
     * @param  TrainingSet trainingSet training set
     */
    updateTraining(trainingSet) {
      let logProb = 0;

      // Forward-backward for each phrase
      // =================================================
      let phraseIndex = 0;
      trainingSet.forEach(phrase => {
        if (phrase.length > 0) {
          logProb += this.baumWelchForwardBackward(phrase, phraseIndex);
        }
        phraseIndex += 1;
      });
      this.baumWelchGammaSum(trainingSet);

      // Re-estimate model parameters
      // =================================================

      // set covariance and mixture coefficients to zero for each state
      for (let i = 0; i < this.params.states; i += 1) {
        for (let c = 0; c < this.params.gaussians; c += 1) {
          this.params.xStates[i].params.mixtureCoeffs[c] = 0;
          if (this.params.covarianceMode === 'full') {
            this.params.xStates[i].params.components[c].covariance = new Array(this.params.dimension ** 2).fill(0);
          } else {
            this.params.xStates[i].params.components[c].covariance = new Array(this.params.dimension).fill(0);
          }
        }
      }

      this.baumWelchEstimateMixtureCoefficients(trainingSet);
      this.baumWelchEstimateMeans(trainingSet);
      this.baumWelchEstimateCovariances(trainingSet);
      if (this.params.transitionMode === 'ergodic') {
        this.baumWelchEstimatePrior(trainingSet);
      }
      this.baumWelchEstimateTransitions(trainingSet);
      return logProb;
    },

    /**
     * terminate the EM Training process
     * @param  TrainingSet trainingSet training set
     */
    terminateTraining() {
      this.normalizeTransitions();
      this.gammaSequence = null;
      this.epsilonSequence = null;
      this.gammaSequenceperMixture = null;
      this.alphaSeq = null;
      this.betaSeq = null;
      this.gammaSum = null;
      this.gammaSumPerMixture = null;
      this.params.xStates = this.params.xStates.map(s => s.params);
    },

    /**
     * Initialize the model parameters to their default values
     * @param  {Array<Number>} dataStddev Standard deviation of the training data
     * @private
     */
    initParametersToDefault(dataStddev) {
      if (this.params.transitionMode === 'ergodic') {
        this.setErgodic();
      } else {
        this.setLeftRight();
      }
      const currentRegularization = dataStddev.map(std => Math.max(this.params.regularization.absolute, this.params.regularization.relative * std));
      const initCovariance = this.params.covarianceMode === 'full' ? () => new Array(this.params.dimension ** 2).fill(this.params.regularization.absolute / 2) : () => new Array(this.params.dimension).fill(0);
      for (let i = 0; i < this.params.states; i += 1) {
        // this.params.xStates[i].initParametersToDefault(dataStddev);
        const s = this.params.xStates[i];
        s.currentRegularization = currentRegularization;
        for (let c = 0; c < this.params.gaussians; c += 1) {
          s.params.components[c].covariance = initCovariance();
          s.params.components[c].regularize(currentRegularization);
          s.params.mixtureCoeffs[c] = 1 / this.params.gaussians;
        }
      }
    },

    /**
     * Initialize the means of each state using all available phrases in the
     * training set
     * @param  {TrainingSet} trainingSet Training set
     * @private
     */
    initMeansWithAllPhrases(trainingSet) {
      if (!trainingSet || trainingSet.empty()) return;

      for (let n = 0; n < this.params.states; n += 1) {
        for (let d = 0; d < this.params.dimension; d += 1) {
          this.params.xStates[n].params.components[0].mean[d] = 0.0;
        }
      }

      const factor = new Array(this.params.states).fill(0);
      trainingSet.forEach(phrase => {
        const step = Math.floor(phrase.length / this.params.states);
        let offset = 0;
        for (let n = 0; n < this.params.states; n += 1) {
          for (let t = 0; t < step; t += 1) {
            for (let d = 0; d < this.params.dimension; d += 1) {
              this.params.xStates[n].params.components[0].mean[d] += phrase.get(offset + t, d);
            }
          }
          offset += step;
          factor[n] += step;
        }
      });
      for (let n = 0; n < this.params.states; n += 1) {
        for (let d = 0; d < this.params.dimension; d += 1) {
          this.params.xStates[n].params.components[0].mean[d] /= factor[n];
        }
      }
    },

    /**
     * Initialize the covariance by direct (fully-observed) estimation from the
     * training data.
     * @param  {[type]} trainingSet [description]
     * @private
     */
    initCovariancesFullyObserved(trainingSet) {
      if (!trainingSet || trainingSet.empty()) return;

      for (let n = 0; n < this.params.states; n += 1) {
        this.params.xStates[n].params.components[0].covariance = new Array(this.params.dimension ** (this.params.covarianceMode === 'full' ? 2 : 1)).fill(0);
      }

      const factor = new Array(this.params.states).fill(0);
      const othermeans = new Array(this.params.states * this.params.dimension).fill(0);
      trainingSet.forEach(phrase => {
        const step = Math.floor(phrase.length / this.params.states);
        let offset = 0;
        for (let n = 0; n < this.params.states; n += 1) {
          for (let t = 0; t < step; t += 1) {
            for (let d1 = 0; d1 < this.params.dimension; d1 += 1) {
              othermeans[n * this.params.dimension + d1] += phrase.get(offset + t, d1);
              if (this.params.covarianceMode === 'full') {
                for (let d2 = 0; d2 < this.params.dimension; d2 += 1) {
                  this.params.xStates[n].params.components[0].covariance[d1 * this.params.dimension + d2] += phrase.get(offset + t, d1) * phrase.get(offset + t, d2);
                }
              } else {
                this.params.xStates[n].params.components[0].covariance[d1] += phrase.get(offset + t, d1) ** 2;
              }
            }
          }
          offset += step;
          factor[n] += step;
        }
      });

      for (let n = 0; n < this.params.states; n += 1) {
        for (let d1 = 0; d1 < this.params.dimension; d1 += 1) {
          othermeans[n * this.params.dimension + d1] /= factor[n];
          if (this.params.covarianceMode === 'full') {
            for (let d2 = 0; d2 < this.params.dimension; d2 += 1) {
              this.params.xStates[n].params.components[0].covariance[d1 * this.params.dimension + d2] /= factor[n];
            }
          } else {
            this.params.xStates[n].params.components[0].covariance[d1] /= factor[n];
          }
        }
      }

      for (let n = 0; n < this.params.states; n += 1) {
        for (let d1 = 0; d1 < this.params.dimension; d1 += 1) {
          if (this.params.covarianceMode === 'full') {
            for (let d2 = 0; d2 < this.params.dimension; d2 += 1) {
              this.params.xStates[n].params.components[0].covariance[d1 * this.params.dimension + d2] -= othermeans[n * this.params.dimension + d1] * othermeans[n * this.params.dimension + d2];
            }
          } else {
            this.params.xStates[n].params.components[0].covariance[d1] -= othermeans[n * this.params.dimension + d1] * othermeans[n * this.params.dimension + d1];
          }
        }
        this.params.xStates[n].regularize();
        this.params.xStates[n].updateInverseCovariances();
      }
    },

    /**
     * Initialize the means and covariance of each state's observation probability
     * distribution using the Expectation-Maximization algorithm for GMMs
     * @param  {[type]} trainingSet [description]
     * @private
     */
    initMeansCovariancesWithGMMEM(trainingSet) {
      for (let n = 0; n < this.params.states; n += 1) {
        const ts = TrainingSet(this.params);
        // eslint-disable-next-line no-loop-func
        trainingSet.forEach((phrase, phraseIndex) => {
          const step = Math.floor(phrase.length / this.params.states);
          if (step > 0) {
            ts.push(phraseIndex, phrase.label);
            for (let t = n * step; t < (n + 1) * step; t += 1) {
              ts.getPhrase(phraseIndex).push(phrase.getFrame(t));
            }
          }
        });
        if (!ts.empty()) {
          const gmmParams = trainGMM(ts, this.params);
          for (let c = 0; c < this.params.gaussians; c += 1) {
            this.params.xStates[n].params.components[c].mean = gmmParams.components[c].mean;
            this.params.xStates[n].params.components[c].covariance = gmmParams.components[c].covariance;
            this.params.xStates[n].updateInverseCovariances();
          }
        }
      }
    },

    /**
     * Initialize the transition matrix to an ergodic transition matrix
     * @private
     */
    setErgodic() {
      const p = 1 / this.params.states;
      this.params.prior = new Array(this.params.states).fill(p);
      this.params.transition = Array.from(new Array(this.params.states), () => new Array(this.params.states).fill(p));
    },

    /**
     * Initialize the transition matrix to a left-right transition matrix
     * @private
     */
    setLeftRight() {
      this.params.prior = new Array(this.params.states).fill(0);
      this.params.prior[0] = 1;
      this.params.transition = new Array(this.params.states * 2).fill(0.5);
      this.params.transition[(this.params.states - 1) * 2] = 1;
      this.params.transition[(this.params.states - 1) * 2 + 1] = 0;
    },

    /**
     * Normalize the hidden state transition parameters
     * (prior + transition matrix)
     * @private
     */
    normalizeTransitions() {
      if (this.params.transitionMode === 'ergodic') {
        const normPrior = this.params.prior.reduce((a, b) => a + b, 0);
        for (let i = 0; i < this.params.states; i += 1) {
          this.params.prior[i] /= normPrior;
          let transitionNorm = 0;
          for (let j = 0; j < this.params.states; j += 1) {
            transitionNorm += this.params.transition[i][j];
          }
          for (let j = 0; j < this.params.states; j += 1) {
            this.params.transition[i][j] /= transitionNorm;
          }
        }
      } else {
        for (let i = 0; i < this.params.states; i += 1) {
          const transitionNorm = this.params.transition[i * 2] + this.params.transition[i * 2 + 1];
          this.params.transition[i * 2] /= transitionNorm;
          this.params.transition[i * 2 + 1] /= transitionNorm;
        }
      }
    },

    /**
     * Initialize the backward algorithm (see rabiner, 1989)
     * @param  {Number} ct Inverse probability at time T - 1 (last observation of
     * the sequence)
     * @private
     */
    initializeBackwardAlgorithm(ct) {
      for (let i = 0; i < this.params.states; i += 1) {
        this.beta[i] = ct;
      }
    },

    /**
     * Initialize the backward algorithm (see rabiner, 1989)
     * @param  {Number} ct Inverse probability at time t
     * @param  {Array<Number>} observation Observation vector
     * @private
     */
    updateBackwardAlgorithm(ct, observation) {
      this.previousBeta = this.beta.slice();
      for (let i = 0; i < this.params.states; i += 1) {
        this.beta[i] = 0;
        if (this.params.transitionMode === 'ergodic') {
          for (let j = 0; j < this.params.states; j += 1) {
            this.beta[i] += this.params.transition[i][j] * this.previousBeta[j] * this.params.xStates[j].likelihood(observation);
          }
        } else {
          this.beta[i] += this.params.transition[i * 2] * this.previousBeta[i] * this.params.xStates[i].likelihood(observation);
          if (i < this.params.states - 1) {
            this.beta[i] += this.params.transition[i * 2 + 1] * this.previousBeta[i + 1] * this.params.xStates[i + 1].likelihood(observation);
          }
        }
        this.beta[i] *= ct;
        if (Number.isNaN(this.beta[i]) || Math.abs(this.beta[i]) === +Infinity) {
          this.beta[i] = 1e100;
        }
      }
    },

    /**
     * Forward algorithm update step for the Baum-Welch algorithms. It is similar
     * to `updateForwardAlgorithm` except it takes precomputed observation
     * likelihoods as argument.
     * @param  {Array<Number>} observationLikelihoods observation likelihoods
     * @private
     */
    baumWelchForwardUpdate(observationLikelihoods) {
      let normConst = 0;
      this.previousAlpha = this.alpha.slice();
      for (let j = 0; j < this.params.states; j += 1) {
        this.alpha[j] = 0;
        if (this.params.transitionMode === 'ergodic') {
          for (let i = 0; i < this.params.states; i += 1) {
            this.alpha[j] += this.previousAlpha[i] * this.params.transition[i][j];
          }
        } else {
          this.alpha[j] += this.previousAlpha[j] * this.params.transition[j * 2];
          if (j > 0) {
            this.alpha[j] += this.previousAlpha[j - 1] * this.params.transition[(j - 1) * 2 + 1];
          } else {
            this.alpha[0] += this.previousAlpha[this.params.states - 1] * this.params.transition[this.params.states * 2 - 1];
          }
        }
        this.alpha[j] *= observationLikelihoods[j];
        normConst += this.alpha[j];
      }
      if (Number.isNaN(normConst)) {
        throw new Error('Holy molly');
      }
      if (normConst > 1e-300) {
        for (let j = 0; j < this.params.states; j += 1) {
          this.alpha[j] /= normConst;
        }
        return 1 / normConst;
      }
      return 0;
    },

    /**
     * Backward algorithm update step for the Baum-Welch algorithms. It is similar
     * to `updatebackwardAlgorithm` except it takes precomputed observation
     * likelihoods as argument.
     * @param  {Number} ct Inverse probability at time t
     * @param  {Array<Number>} observationLikelihoods observation likelihoods
     * @private
     */
    baumWelchBackwardUpdate(ct, observationLikelihoods) {
      this.previousBeta = this.beta.slice();
      for (let i = 0; i < this.params.states; i += 1) {
        this.beta[i] = 0;
        if (this.params.transitionMode === 'ergodic') {
          for (let j = 0; j < this.params.states; j += 1) {
            this.beta[i] += this.params.transition[i][j] * this.previousBeta[j] * observationLikelihoods[j];
          }
        } else {
          this.beta[i] += this.params.transition[i * 2] * this.previousBeta[i] * observationLikelihoods[i];
          if (i < this.params.states - 1) {
            this.beta[i] += this.params.transition[i * 2 + 1] * this.previousBeta[i + 1] * observationLikelihoods[i + 1];
          }
        }
        this.beta[i] *= ct;
        if (Number.isNaN(this.beta[i]) || Math.abs(this.beta[i]) === +Infinity) {
          this.beta[i] = 1e100;
        }
      }
    },

    /**
     * Forward-Backward algorithm for the Baum-Welch training algorithm
     * @param  {Phrase} currentPhrase Current data phrase
     * @param  {Number} phraseIndex   Current phrase index
     * @return {Number} Log-likelihood
     * @private
     */
    baumWelchForwardBackward(currentPhrase, phraseIndex) {
      const T = currentPhrase.length;

      const ct = new Array(T).fill(0);
      let logProb;
      this.alphaSeq = [];
      this.betaSeq = [];

      const observationProbabilities = Array.from(new Array(T), () => new Array(this.params.states).fill(0));
      for (let t = 0; t < T; t += 1) {
        for (let i = 0; i < this.params.states; i += 1) {
          observationProbabilities[t][i] = this.params.xStates[i].likelihood(currentPhrase.getFrame(t));
        }
      }

      // Forward algorithm
      ct[0] = this.initializeForwardAlgorithm(currentPhrase.getFrame(0));
      logProb = -Math.log(ct[0]);
      this.alphaSeq.push(this.alpha.slice());

      for (let t = 1; t < T; t += 1) {
        ct[t] = this.baumWelchForwardUpdate(observationProbabilities[t]);
        logProb -= Math.log(ct[t]);
        this.alphaSeq.push(this.alpha.slice());
      }

      // Backward algorithm
      this.initializeBackwardAlgorithm(ct[T - 1]);
      this.betaSeq.push(this.beta.slice());

      for (let t = T - 2; t >= 0; t -= 1) {
        this.baumWelchBackwardUpdate(ct[t], observationProbabilities[t + 1]);
        this.betaSeq.push(this.beta.slice());
      }
      this.betaSeq.reverse();

      // Compute Gamma Variable
      for (let t = 0; t < T; t += 1) {
        for (let i = 0; i < this.params.states; i += 1) {
          this.gammaSequence[phraseIndex][t][i] = this.alphaSeq[t][i] * this.betaSeq[t][i] / ct[t];
        }
      }

      // Compute Gamma variable for each mixture component
      let normConst;

      for (let t = 0; t < T; t += 1) {
        for (let i = 0; i < this.params.states; i += 1) {
          normConst = 0;
          if (this.params.gaussians === 1) {
            const oo = observationProbabilities[t][i];
            this.gammaSequenceperMixture[phraseIndex][0][t][i] = this.gammaSequence[phraseIndex][t][i] * oo;
            normConst += oo;
          } else {
            for (let c = 0; c < this.params.gaussians; c += 1) {
              const oo = this.params.xStates[i].componentLikelihood(currentPhrase.getFrame(t), c);
              this.gammaSequenceperMixture[phraseIndex][c][t][i] = this.gammaSequence[phraseIndex][t][i] * oo;
              normConst += oo;
            }
          }
          if (normConst > 0) {
            for (let c = 0; c < this.params.gaussians; c += 1) {
              this.gammaSequenceperMixture[phraseIndex][c][t][i] /= normConst;
            }
          }
        }
      }

      // Compute Epsilon Variable
      if (this.params.transitionMode === 'ergodic') {
        for (let t = 0; t < T - 1; t += 1) {
          for (let i = 0; i < this.params.states; i += 1) {
            for (let j = 0; j < this.params.states; j += 1) {
              this.epsilonSequence[phraseIndex][t][i][j] = this.alphaSeq[t][i] * this.params.transition[i][j] * this.betaSeq[t + 1][j];
              this.epsilonSequence[phraseIndex][t][i][j] *= observationProbabilities[t + 1][j];
            }
          }
        }
      } else {
        for (let t = 0; t < T - 1; t += 1) {
          for (let i = 0; i < this.params.states; i += 1) {
            this.epsilonSequence[phraseIndex][t][i * 2] = this.alphaSeq[t][i] * this.params.transition[i * 2] * this.betaSeq[t + 1][i];
            this.epsilonSequence[phraseIndex][t][i * 2] *= observationProbabilities[t + 1][i];
            if (i < this.params.states - 1) {
              this.epsilonSequence[phraseIndex][t][i * 2 + 1] = this.alphaSeq[t][i] * this.params.transition[i * 2 + 1] * this.betaSeq[t + 1][i + 1];
              this.epsilonSequence[phraseIndex][t][i * 2 + 1] *= observationProbabilities[t + 1][i + 1];
            }
          }
        }
      }

      return logProb;
    },

    /**
     * Sums the Gamma variables used for parameter estimation during training
     * @param  {TrainingSet} trainingSet Training Set
     * @private
     */
    baumWelchGammaSum(trainingSet) {
      for (let i = 0; i < this.params.states; i += 1) {
        this.gammaSum[i] = 0;
        for (let c = 0; c < this.params.gaussians; c += 1) {
          this.gammaSumPerMixture[i * this.params.gaussians + c] = 0;
        }
      }

      let phraseIndex = 0;
      trainingSet.forEach(phrase => {
        for (let i = 0; i < this.params.states; i += 1) {
          for (let t = 0; t < phrase.length; t += 1) {
            this.gammaSum[i] += this.gammaSequence[phraseIndex][t][i];
            for (let c = 0; c < this.params.gaussians; c += 1) {
              this.gammaSumPerMixture[i * this.params.gaussians + c] += this.gammaSequenceperMixture[phraseIndex][c][t][i];
            }
          }
        }
        phraseIndex += 1;
      });
    },

    /**
     * Estimate the mixture coefficients of the GMM observation probability
     * distribution at each state.
     * @param  {TrainingSet} trainingSet Training Set
     * @private
     */
    baumWelchEstimateMixtureCoefficients(trainingSet) {
      let phraseIndex = 0;
      trainingSet.forEach(phrase => {
        for (let i = 0; i < this.params.states; i += 1) {
          for (let t = 0; t < phrase.length; t += 1) {
            for (let c = 0; c < this.params.gaussians; c += 1) {
              this.params.xStates[i].params.mixtureCoeffs[c] += this.gammaSequenceperMixture[phraseIndex][c][t][i];
            }
          }
        }
        phraseIndex += 1;
      });

      // Scale mixture coefficients
      for (let i = 0; i < this.params.states; i += 1) {
        this.params.xStates[i].normalizeMixtureCoeffs();
      }
    },

    /**
     * Estimate the means of the GMM observation probability
     * distribution at each state.
     * @param  {TrainingSet} trainingSet Training Set
     * @private
     */
    baumWelchEstimateMeans(trainingSet) {
      for (let i = 0; i < this.params.states; i += 1) {
        for (let c = 0; c < this.params.gaussians; c += 1) {
          this.params.xStates[i].params.components[c].mean.fill(0);
        }
      }

      // Re-estimate Mean
      let phraseIndex = 0;
      trainingSet.forEach(phrase => {
        for (let i = 0; i < this.params.states; i += 1) {
          for (let t = 0; t < phrase.length; t += 1) {
            for (let c = 0; c < this.params.gaussians; c += 1) {
              for (let d = 0; d < this.params.dimension; d += 1) {
                this.params.xStates[i].params.components[c].mean[d] += this.gammaSequenceperMixture[phraseIndex][c][t][i] * phrase.get(t, d);
              }
            }
          }
        }
        phraseIndex += 1;
      });

      // Normalize mean
      for (let i = 0; i < this.params.states; i += 1) {
        for (let c = 0; c < this.params.gaussians; c += 1) {
          for (let d = 0; d < this.params.dimension; d += 1) {
            if (this.gammaSumPerMixture[i * this.params.gaussians + c] > 0) {
              this.params.xStates[i].params.components[c].mean[d] /= this.gammaSumPerMixture[i * this.params.gaussians + c];
            }
            if (Number.isNaN(this.params.xStates[i].params.components[c].mean[d])) {
              throw new Error('Convergence Error');
            }
          }
        }
      }
    },

    /**
     * Estimate the covariances of the GMM observation probability
     * distribution at each state.
     * @param  {TrainingSet} trainingSet Training Set
     * @private
     */
    baumWelchEstimateCovariances(trainingSet) {
      let phraseIndex = 0;
      trainingSet.forEach(phrase => {
        for (let i = 0; i < this.params.states; i += 1) {
          for (let t = 0; t < phrase.length; t += 1) {
            for (let c = 0; c < this.params.gaussians; c += 1) {
              for (let d1 = 0; d1 < this.params.dimension; d1 += 1) {
                if (this.params.covarianceMode === 'full') {
                  for (let d2 = d1; d2 < this.params.dimension; d2 += 1) {
                    this.params.xStates[i].params.components[c].covariance[d1 * this.params.dimension + d2] += this.gammaSequenceperMixture[phraseIndex][c][t][i] * (phrase.get(t, d1) - this.params.xStates[i].params.components[c].mean[d1]) * (phrase.get(t, d2) - this.params.xStates[i].params.components[c].mean[d2]);
                  }
                } else {
                  const value = phrase.get(t, d1) - this.params.xStates[i].params.components[c].mean[d1];
                  this.params.xStates[i].params.components[c].covariance[d1] += this.gammaSequenceperMixture[phraseIndex][c][t][i] * value ** 2;
                }
              }
            }
          }
        }
        phraseIndex += 1;
      });

      // Scale covariance
      for (let i = 0; i < this.params.states; i += 1) {
        for (let c = 0; c < this.params.gaussians; c += 1) {
          if (this.gammaSumPerMixture[i * this.params.gaussians + c] > 0) {
            for (let d1 = 0; d1 < this.params.dimension; d1 += 1) {
              if (this.params.covarianceMode === 'full') {
                for (let d2 = d1; d2 < this.params.dimension; d2 += 1) {
                  this.params.xStates[i].params.components[c].covariance[d1 * this.params.dimension + d2] /= this.gammaSumPerMixture[i * this.params.gaussians + c];
                  if (d1 !== d2) {
                    this.params.xStates[i].params.components[c].covariance[d2 * this.params.dimension + d1] = this.params.xStates[i].params.components[c].covariance[d1 * this.params.dimension + d2];
                  }
                }
              } else {
                this.params.xStates[i].params.components[c].covariance[d1] /= this.gammaSumPerMixture[i * this.params.gaussians + c];
              }
            }
          }
        }
        this.params.xStates[i].regularize();
        this.params.xStates[i].updateInverseCovariances();
      }
    },

    /**
     * Estimate the prior probabilities of the model
     * @param  {TrainingSet} trainingSet Training Set
     * @private
     */
    baumWelchEstimatePrior(trainingSet) {
      this.params.prior.fill(0);

      // Re-estimate Prior probabilities
      let sumprior = 0;
      for (let phraseIndex = 0; phraseIndex < trainingSet.size(); phraseIndex += 1) {
        for (let i = 0; i < this.params.states; i += 1) {
          this.params.prior[i] += this.gammaSequence[phraseIndex][0][i];
          sumprior += this.params.prior[i];
        }
      }

      // Scale Prior vector
      if (sumprior > 0) {
        for (let i = 0; i < this.params.states; i += 1) {
          this.params.prior[i] /= sumprior;
        }
      } else {
        throw new Error('The Prior is all ZERO.....');
      }
    },

    /**
     * Estimate the transition probabilities of the model
     * @param  {TrainingSet} trainingSet Training Set
     * @private
     */
    baumWelchEstimateTransitions(trainingSet) {
      // Set transition matrix to 0
      this.params.transition = this.params.transitionMode === 'ergodic' ? Array.from(new Array(this.params.states), () => new Array(this.params.states).fill(0)) : new Array(this.params.states * 2).fill(0);

      // Re-estimate Transition probabilities
      let phraseIndex = 0;
      trainingSet.forEach(phrase => {
        if (phrase.length > 0) {
          for (let i = 0; i < this.params.states; i += 1) {
            // Experimental: A bit of regularization (sometimes avoids
            // numerical errors)
            if (this.params.transitionMode === 'leftright') {
              this.params.transition[i * 2] += TRANSITION_REGULARIZATION;
              if (i < this.params.states - 1) {
                this.params.transition[i * 2 + 1] += TRANSITION_REGULARIZATION;
              } else {
                this.params.transition[i * 2] += TRANSITION_REGULARIZATION;
              }
            }
            // End Regularization
            if (this.params.transitionMode === 'ergodic') {
              for (let j = 0; j < this.params.states; j += 1) {
                for (let t = 0; t < phrase.length - 1; t += 1) {
                  this.params.transition[i][j] += this.epsilonSequence[phraseIndex][t][i][j];
                }
              }
            } else {
              for (let t = 0; t < phrase.length - 1; t += 1) {
                this.params.transition[i * 2] += this.epsilonSequence[phraseIndex][t][i * 2];
              }
              if (i < this.params.states - 1) {
                for (let t = 0; t < phrase.length - 1; t += 1) {
                  this.params.transition[i * 2 + 1] += this.epsilonSequence[phraseIndex][t][i * 2 + 1];
                }
              }
            }
          }
        }
        phraseIndex += 1;
      });

      // Scale transition matrix
      if (this.params.transitionMode === 'ergodic') {
        for (let i = 0; i < this.params.states; i += 1) {
          for (let j = 0; j < this.params.states; j += 1) {
            this.params.transition[i][j] /= this.gammaSum[i] + 2 * TRANSITION_REGULARIZATION;
            if (Number.isNaN(this.params.transition[i][j])) {
              throw new Error('Convergence Error. Check your training data or increase the variance offset');
            }
          }
        }
      } else {
        for (let i = 0; i < this.params.states; i += 1) {
          this.params.transition[i * 2] /= this.gammaSum[i] + 2 * TRANSITION_REGULARIZATION;
          if (Number.isNaN(this.params.transition[i * 2])) {
            throw new Error('Convergence Error. Check your training data or increase the variance offset');
          }
          if (i < this.params.states - 1) {
            this.params.transition[i * 2 + 1] /= this.gammaSum[i] + 2 * TRANSITION_REGULARIZATION;
            if (Number.isNaN(this.params.transition[i * 2 + 1])) {
              throw new Error('Convergence Error. Check your training data or increase the variance offset');
            }
          }
        }
      }
    }
  };

  /**
   * Add HMM Training capabilities to a HMM Model
   * @param  {HMMBase} o               Source HMM Model
   * @param  {Number} [states=1]       Number of hidden states
   * @param  {Number} [gaussians=1]    Number of Gaussian components
   * @param  {Object} [regularization] Regularization parameters
   * @param  {Number} [regularization.absolute=1e-3] Absolute regularization
   * @param  {Number} [regularization.relative=1e-2] Relative Regularization
   (relative to the training set's variance along each dimension)
   * @param  {String} [transitionMode='ergodic'] Structure of the transition
   * matrix ('ergodic' or 'left-right').
   * @param  {String} [covarianceMode='full'] Covariance mode ('full' or diagonal)
   * @return {BMMBase}
   */
  function withHMMTraining(o, states = 1, gaussians = 1, regularization = { absolute: 1e-3, relative: 1e-2 }, transitionMode = 'leftright', covarianceMode = 'full') {
    if (!Object.keys(o).includes('params')) {
      throw new Error('The base object must include a standard set of parameters (`params` key), @see `ModelBase`.');
    }
    return Object.assign(o, hmmTrainerPrototype, {
      params: _extends({}, o.params, {
        states,
        gaussians,
        regularization,
        transitionMode,
        covarianceMode
      })
    });
  }

  const hmmParameterSpec = (states, transitionMode) => ({
    states: {
      required: true,
      check: { min: 1 }
    },
    gaussians: {
      required: true,
      check: { min: 1 }
    },
    regularization: {
      required: true,
      check: ({ absolute, relative }) => absolute && relative && absolute > 0 && relative > 0
    },
    transitionMode: {
      required: true,
      check: ['ergodic', 'leftright']
    },
    covarianceMode: {
      required: true,
      check: ['full', 'diagonal']
    },
    prior: {
      required: true,
      check: m => transitionMode === 'leftright' || m.length === states
    },
    transition: {
      required: true,
      check: m => transitionMode === 'leftright' ? m.length === 2 * states : m.length === states
    },
    xStates: {
      required: true,
      check: m => m.length === states
    }
  });

  /**
   * HMM Base prototype
   * @type {Object}
   * @ignore
   */
  const hmmPredictionPrototype = /** @lends withHMMPrediction */{
    forwardInitialized: false,
    isHierarchical: false,

    /**
     * Setup the Model by allocating GMM predictors to each of the hidden states
     * @return {HMMBaseModel} the model
     * @private
     */
    setup() {
      this.params.xStates = this.params.xStates.map(s => GMMPredictor(s).reset());
      return this;
    },

    /**
     * Reset the prediction process
     * @return {HMMBaseModel} the model
     */
    reset() {
      this.likelihoodBuffer.clear();
      this.params.xStates.forEach(s => {
        s.reset();
      });
      return this;
    },

    /**
     * Compute the likelihood of an observation given the HMM's parameters
     * @param  {Array<Number>} observation Observation vector
     * @return {Number}
     */
    likelihood(observation) {
      const ct = this.forwardInitialized ? this.updateForwardAlgorithm(observation) : this.initializeForwardAlgorithm(observation);
      this.updateAlphaWindow();
      this.updateProgress();
      return 1 / ct;
    },

    updateProgress() {
      this.results.progress = 0.0;
      for (let i = this.windowMinindex; i < this.windowMaxindex; i += 1) {
        if (this.isHierarchical) {
          this.results.progress += (this.alpha[i] + this.alpha1[i] + this.alpha2[i]) * (i / this.windowNormalizationConstant);
        } else {
          this.results.progress += this.alpha[i] * i / this.windowNormalizationConstant;
        }
      }
      this.results.progress /= this.params.states - 1;
    },

    /**
     * Update the state probabilities filtering window (for multiclass
     * hierarchical HMM I think...)
     * @private
     */
    updateAlphaWindow() {
      this.results.likeliestState = 0;
      // Get likeliest State
      let bestAlpha = this.isHierarchical ? this.alpha[0] + this.alpha1[0] : this.alpha[0];
      for (let i = 1; i < this.params.states; i += 1) {
        if (this.isHierarchical) {
          if (this.alpha[i] + this.alpha1[i] > bestAlpha) {
            bestAlpha = this.alpha[i] + this.alpha1[i];
            this.results.likeliestState = i;
          }
        } else if (this.alpha[i] > bestAlpha) {
          bestAlpha = this.alpha[i];
          this.results.likeliestState = i;
        }
      }

      // Compute Window
      this.windowMinindex = this.results.likeliestState - Math.floor(this.params.states / 2);
      this.windowMaxindex = this.results.likeliestState + Math.floor(this.params.states / 2);
      this.windowMinindex = this.windowMinindex >= 0 ? this.windowMinindex : 0;
      this.windowMaxindex = this.windowMaxindex <= this.params.states ? this.windowMaxindex : this.params.states;
      this.windowNormalizationConstant = 0.0;
      for (let i = this.windowMinindex; i < this.windowMaxindex; i += 1) {
        this.windowNormalizationConstant += this.isHierarchical ? this.alpha[i] + this.alpha1[i] : this.alpha[i];
      }
    }
  };

  /**
   * Bimodal (regression) HMM Prototype
   * @type {Object}
   * @ignore
   */
  const hmmBimodalPredictionPrototype = /** @lends withHMMPrediction */{
    /**
     * Estimate the output values corresponding to the input observation, by
     * regression given the HMM's parameters. This method is called Hidden
     * Mixture Regression (GMR).
     *
     * @param  {Array<Number>} inputObservation Observation on the input modality
     * @return {Array<Number>} Output values (length = outputDimension)
     */
    regression(inputObservation) {
      this.results.outputValues = Array(this.params.outputDimension).fill(0);
      this.results.outputCovariance = Array(this.params.covarianceMode === 'full' ? this.params.outputDimension ** 2 : this.params.outputDimension).fill(0);

      if (this.params.regressionEstimator === 'likeliest') {
        this.params.xStates[this.results.likeliestState].predict(inputObservation);
        this.results.outputValues = this.params.xStates[this.results.likeliestState].results.outputValues;
        return this.results.outputValues;
      }

      const clipMinState = this.params.regressionEstimator === 'full' ? 0 : this.windowMinindex;
      const clipMaxState = this.params.regressionEstimator === 'full' ? this.params.states : this.windowMaxindex;
      let normalizationConstant = this.params.regressionEstimator === 'full' ? 1 : this.windowNormalizationConstant;

      if (normalizationConstant <= 0.0) normalizationConstant = 1;

      // Compute Regression
      for (let i = clipMinState; i < clipMaxState; i += 1) {
        this.params.xStates[i].likelihood(inputObservation);
        this.params.xStates[i].regression(inputObservation);
        const tmpPredictedOutput = this.params.xStates[i].results.outputValues;
        for (let d = 0; d < this.params.outputDimension; d += 1) {
          if (this.isHierarchical) {
            this.results.outputValues[d] += (this.alpha[i] + this.alpha1[i]) * (tmpPredictedOutput[d] / normalizationConstant);
            if (this.params.covarianceMode === 'full') {
              for (let d2 = 0; d2 < this.params.outputDimension; d2 += 1) {
                this.results.outputCovariance[d * this.params.outputDimension + d2] += (this.alpha[i] + this.alpha1[i]) * (this.alpha[i] + this.alpha1[i]) * (this.params.xStates[i].results.outputCovariance[d * this.params.outputDimension + d2] / normalizationConstant);
              }
            } else {
              this.results.outputCovariance[d] += (this.alpha[i] + this.alpha1[i]) * (this.alpha[i] + this.alpha1[i]) * (this.params.xStates[i].results.outputCovariance[d] / normalizationConstant);
            }
          } else {
            this.results.outputValues[d] += this.alpha[i] * (tmpPredictedOutput[d] / normalizationConstant);
            if (this.params.covarianceMode === 'full') {
              for (let d2 = 0; d2 < this.params.outputDimension; d2 += 1) {
                this.results.outputCovariance[d * this.params.outputDimension + d2] += this.alpha[i] ** 2 * (this.params.xStates[i].results.outputCovariance[d * this.params.outputDimension + d2] / normalizationConstant);
              }
            } else {
              this.results.outputCovariance[d] += this.alpha[i] ** 2 * this.params.xStates[i].results.outputCovariance[d] / normalizationConstant;
            }
          }
        }
      }
      return this.results.outputValues;
    }
  };

  /**
   * Add HMM prediction capabilities to a single-class model. Mostly, this checks
   * the validity of the model parameters
   *
   * @todo validate gaussian components
   *
   * @param  {HMMBaseModel} o Source Model
   * @return {HMMBaseModel}
   *
   * @throws {Error} is o is not a ModelBase
   */
  function withHMMPrediction(o) {
    if (!isBaseModel(o)) {
      throw new Error('The base object must include a standard set of parameters (`params` key), @see `ModelBase`.');
    }
    validateParameters('HMM', hmmParameterSpec(o.params.states, o.params.transitionMode), o.params);
    return Object.assign(o, hmmPredictionPrototype, o.params.bimodal ? hmmBimodalPredictionPrototype : {}, {
      alpha: new Array(o.params.states).fill(0),
      previous_alpha_: new Array(o.params.states).fill(0)
    }).setup();
  }

  const DEFAULT_EXITPROBABILITY_LAST_STATE = 0.1;

  /**
   * Hierarchical HMM Base prototype
   * @type {Object}
   * @ignore
   */
  const hierarchicalHmmPredictionPrototype =
  /** @lends withHierarchicalHMMPrediction */
  {
    /**
     * Specificies if the forward algorithm has been initialized
     * @type {Boolean}
     * @private
     */
    forwardInitialized: false,

    /**
     * Setup the model (allocate transition parameters)
     * @return {HierarchicalHMM} [description]
     * @private
     */
    setup() {
      const numClasses = this.size();
      this.params.prior = new Array(numClasses).fill(1 / numClasses);
      this.params.transition = Array.from(new Array(numClasses), () => new Array(numClasses).fill(1 / numClasses));
      this.params.exitTransition = new Array(numClasses).fill(0.1);
      Object.values(this.models).forEach(model => {
        const m = model;
        m.isHierarchical = true;
      });
      this.updateExitProbabilities();
      return this;
    },

    /**
     * Update the exit probabilities of each sub-Markov model
     * @param  {Array<Number>|undefined} [exitProbabilities=undefined] Vector of
     * exit probabilities (optional)
     * @private
     */
    updateExitProbabilities(exitProbabilities = undefined) {
      const exitProb = exitProbabilities !== undefined ? exitProbabilities : new Array(this.params.states - 1).fill(0).concat([DEFAULT_EXITPROBABILITY_LAST_STATE]);
      Object.keys(this.models).forEach(label => {
        this.models[label].params.exitProbabilities = exitProb.slice();
      });
    },

    /**
     * Reset the prediction process. This is particularly important for temporal
     * models such as HMMs, that depends on previous observations.
     */
    reset() {
      Object.values(this.models).forEach(m => m.reset());
      this.results = {
        labels: [],
        instantLikelihoods: [],
        smoothedLikelihoods: [],
        smoothedLogLikelihoods: [],
        smoothedNormalizedLikelihoods: [],
        exitLikelihood: [],
        likeliest: null,
        classes: {}
      };
      if (this.params.bimodal) {
        this.resetBimodal();
      }
      this.forwardInitialized = false;
    },

    /**
     * Make a prediction from a new observation (updates the results member)
     * @param  {Array<Number>} observation Observation vector
     */
    predict(observation) {
      if (this.forwardInitialized) {
        this.updateForwardAlgorithm(observation);
      } else {
        this.initializeForwardAlgorithm(observation);
      }
      Object.keys(this.models).sort().forEach(label => {
        const model = this.models[label];
        model.updateAlphaWindow();
        model.updateProgress();
        model.updateResults(model.results.instantLikelihood);
      });
      this.updateResults();

      if (this.params.bimodal) {
        Object.values(this.models).forEach(m => m.regression(observation));

        if (this.params.multiClassRegressionEstimator === 'likeliest') {
          this.results.outputValues = this.models[this.results.likeliest].results.outputValues;
          this.results.outputCovariance = this.models[this.results.likeliest].results.outputCovariance;
        } else {
          this.results.outputValues = new Array(this.outputDimension).fill(0);
          this.results.outputCovariance = new Array(this.params.covarianceMode === 'full' ? this.outputDimension ** 2 : this.outputDimension).fill(0);

          let modelIndex = 0;
          Object.values(this.models).forEach(model => {
            for (let d = 0; d < this.outputDimension; d += 1) {
              this.results.outputValues[d] += this.results.smoothedNormalizedLikelihoods[modelIndex] * model.second.results.outputValues[d];

              if (this.params.covarianceMode === 'full') {
                for (let d2 = 0; d2 < this.outputDimension; d2 += 1) {
                  this.results.outputCovariance[d * this.outputDimension + d2] += this.results.smoothedNormalizedLikelihoods[modelIndex] * model.results.outputCovariance[d * this.outputDimension + d2];
                }
              } else {
                this.results.outputCovariance[d] += this.results.smoothedNormalizedLikelihoods[modelIndex] * model.second.results.outputCovariance[d];
              }
            }
            modelIndex += 1;
          });
        }
      }
    },

    /**
     * Initialize the forward algorithm of the hierarchical HMM
     * @param  {Array<Number>} observation Observation vector
     * @private
     */
    initializeForwardAlgorithm(observation) {
      let normConst = 0;
      let modelIndex = 0;
      const classes = Object.keys(this.models).sort();
      classes.forEach(label => {
        const model = this.models[label];
        const N = model.params.states;
        model.alpha1 = new Array(N).fill(0);
        model.alpha2 = new Array(N).fill(0);

        // Compute Emission probability and initialize on the first state of
        // the primitive
        if (model.params.transitionMode === 'ergodic') {
          model.results.instantLikelihood = 0;
          for (let i = 0; i < N; i += 1) {
            model.alpha[i] = this.params.prior[modelIndex] * model.params.prior[i] * model.params.xStates[i].likelihood(observation);
            model.results.instantLikelihood += model.alpha[i];
          }
        } else {
          model.alpha[0] = this.params.prior[modelIndex] * model.params.xStates[0].likelihood(observation);
          [model.results.instantLikelihood] = model.alpha;
        }
        normConst += model.results.instantLikelihood;
        modelIndex += 1;
      });

      classes.forEach(label => {
        const model = this.models[label];
        const N = model.params.states;
        for (let i = 0; i < N; i += 1) {
          model.alpha[i] /= normConst;
        }
      });

      this.frontierV1 = new Array(this.size).fill(0);
      this.frontierV2 = new Array(this.size).fill(0);
      this.forwardInitialized = true;
    },

    /**
     * Update the forward algorithm of the hierarchical HMM
     * @param  {Array<Number>} observation Observation vector
     * @private
     */
    updateForwardAlgorithm(observation) {
      let normConst = 0;

      // Frontier Algorithm: variables
      let tmp = 0;

      // Intermediate variables: compute the sum of probabilities of making a
      // transition to a new primitive
      this.frontierV1 = this.likelihoodAlpha(1);
      this.frontierV2 = this.likelihoodAlpha(2);

      // FORWARD UPDATE
      // --------------------------------------
      let dstModelIndex = 0;
      const classes = Object.keys(this.models).sort();
      classes.forEach(label => {
        const dstModel = this.models[label];
        const N = dstModel.params.states;

        // 1) COMPUTE FRONTIER VARIABLE
        //    --------------------------------------
        // frontier variable : intermediate computation variable
        const front = new Array(N).fill(0);

        if (dstModel.params.transitionMode === 'ergodic') {
          for (let k = 0; k < N; k += 1) {
            for (let j = 0; j < N; j += 1) {
              front[k] += dstModel.params.transition[j][k] / (1 - dstModel.params.exitProbabilities[j]) * dstModel.alpha[j];
            }

            for (let srcModelIndex = 0; srcModelIndex < this.size(); srcModelIndex += 1) {
              front[k] += dstModel.params.prior[k] * (this.frontierV1[srcModelIndex] * this.params.transition[srcModelIndex][dstModelIndex] + this.params.prior[dstModelIndex] * this.frontierV2[srcModelIndex]);
            }
          }
        } else {
          // k=0: first state of the primitive
          front[0] = dstModel.params.transition[0] * dstModel.alpha[0];

          for (let srcModelIndex = 0; srcModelIndex < this.size(); srcModelIndex += 1) {
            front[0] += this.frontierV1[srcModelIndex] * this.params.transition[srcModelIndex][dstModelIndex] + this.params.prior[dstModelIndex] * this.frontierV2[srcModelIndex];
          }

          // k>0: rest of the primitive
          for (let k = 1; k < N; k += 1) {
            front[k] += dstModel.params.transition[k * 2] / (1 - dstModel.params.exitProbabilities[k]) * dstModel.alpha[k];
            front[k] += dstModel.params.transition[(k - 1) * 2 + 1] / (1 - dstModel.params.exitProbabilities[k - 1]) * dstModel.alpha[k - 1];
          }

          for (let k = 0; k < N; k += 1) {
            dstModel.alpha[k] = 0;
            dstModel.alpha1[k] = 0;
            dstModel.alpha2[k] = 0;
          }
        }

        // 2) UPDATE FORWARD VARIABLE
        //    --------------------------------------
        dstModel.results.exitLikelihood = 0.0;
        dstModel.results.instantLikelihood = 0.0;

        // end of the primitive: handle exit states
        for (let k = 0; k < N; k += 1) {
          tmp = dstModel.params.xStates[k].likelihood(observation) * front[k];
          dstModel.alpha2[k] = this.params.exitTransition[dstModelIndex] * dstModel.params.exitProbabilities[k] * tmp;
          dstModel.alpha1[k] = (1 - this.params.exitTransition[dstModelIndex]) * dstModel.params.exitProbabilities[k] * tmp;
          dstModel.alpha[k] = (1 - dstModel.params.exitProbabilities[k]) * tmp;

          dstModel.results.exitLikelihood += dstModel.alpha1[k] + dstModel.alpha2[k];
          dstModel.results.instantLikelihood += dstModel.alpha[k] + dstModel.alpha1[k] + dstModel.alpha2[k];
          normConst += tmp;
        }

        dstModel.results.exitRatio = dstModel.results.exitLikelihood / dstModel.results.instantLikelihood;

        dstModelIndex += 1;
      });

      classes.forEach(label => {
        const model = this.models[label];
        const N = model.params.states;
        for (let k = 0; k < N; k += 1) {
          model.alpha[k] /= normConst;
          model.alpha1[k] /= normConst;
          model.alpha2[k] /= normConst;
        }
      });
    },

    /**
     * Compute the likelihood of a given probability.
     * @param  {Number} exitNum Exit level number
     * @return {Array<Number>}
     */
    likelihoodAlpha(exitNum) {
      const likelihoodVector = new Array(this.size()).fill(0);
      if (exitNum < 0) {
        // Likelihood over all exit states
        let modelIndex = 0;
        Object.keys(this.models).sort().forEach(label => {
          const model = this.models[label];
          likelihoodVector[modelIndex] = 0.0;
          for (let k = 0; k < model.params.states; k += 1) {
            likelihoodVector[modelIndex] += model.second.alpha[k] + model.second.alpha1[k] + model.second.alpha2[k];
          }
          modelIndex += 1;
        });
      } else {
        // Likelihood for exit state "exitNum"
        let modelIndex = 0;
        Object.keys(this.models).sort().forEach(label => {
          const model = this.models[label];
          likelihoodVector[modelIndex] = 0;
          let { alpha } = model;
          if (exitNum === 1) {
            alpha = model.alpha1;
          }
          if (exitNum === 2) {
            alpha = model.alpha2;
          }
          for (let k = 0; k < model.params.states; k += 1) {
            likelihoodVector[modelIndex] += alpha[k];
          }
          modelIndex += 1;
        });
      }
      return likelihoodVector;
    }
  };

  /**
   * Add Hierarchical HMM prediction capabilities to a multi-class model.
   *
   * @todo algorithmic details
   * @todo validate parameters
   * @todo validate gaussian components
   *
   * @param  {MulticlassBaseModel} o Source Model
   * @return {HierarchicalHMM}
   *
   * @extends withMulticlassPrediction
   *
   * @throws {Error} is o is not a ModelBase
   */
  function withHierarchicalHMMPrediction(o) {
    if (!isBaseModel(o)) {
      throw new Error('The base object must include a standard set of parameters (`params` key), @see `ModelBase`.');
    }
    // validateParameters(
    //   'Hierarchical HMM',
    //   hierarchicalHmmParameterSpec(o.params.states, o.params.transitionMode),
    //   o.params,
    // );
    return Object.assign(o, hierarchicalHmmPredictionPrototype, {
      // alpha: new Array(o.params.states).fill(0),
      // previous_alpha_: new Array(o.params.states).fill(0),
    }).setup();
  }

  /**
   * @typedef {Object} HMMParameters
   * @property {Boolean} bimodal Specifies if the model is bimodal
   * @property {Number} inputDimension Dimension of the input modality
   * @property {Number} outputDimension Dimension of the output modality
   * @property {Number} dimension Total dimension
   * @property {Number} states Number of hidden states in the Markov model
   * @property {Number} gaussians Number of components in the Gaussian mixture
   * observation distribution of each state
   * @property {String} transitionMode Transition matrix mode ('ergodic' or 'leftright')
   * @property {String} covarianceMode Covariance mode ('full' or 'diagonal')
   * @property {Array<Number>} mixtureCoeffs mixture coefficients ('weight' of
   * each gaussian component)
   * @property {Array<GaussianDistribution>} components Gaussian components
   */

  /**
   * Train a single-class HMM Model.
   *
   * @todo HMM details
   *
   * @param  {TrainingSet} trainingSet                training set
   * @param  {Object} configuration                   Training configuration
   * @param  {Object} [convergenceCriteria=undefined] Convergence criteria of the
   * EM algorithm
   * @return {HMMParameters} Parameters of the trained HMM
   */
  function trainHMM(trainingSet, configuration, convergenceCriteria = undefined) {
    const { inputDimension, outputDimension } = trainingSet;
    const {
      states,
      gaussians,
      regularization,
      transitionMode,
      covarianceMode
    } = configuration;
    const model = withHMMTraining(withEMTraining(withHMMBase(ModelBase(_extends({
      inputDimension,
      outputDimension
    }, configuration))), convergenceCriteria), states, gaussians, regularization, transitionMode, covarianceMode);
    return model.train(trainingSet);
  }

  /**
   * Train a multi-class HMM Model.
   *
   * @todo HMM details
   *
   * @param  {TrainingSet} trainingSet                training set
   * @param  {Object} configuration                   Training configuration
   * @param  {Object} [convergenceCriteria=undefined] Convergence criteria of the
   * EM algorithm
   * @return {Object} Parameters of the trained HMM
   */
  function trainMulticlassHMM(trainingSet, configuration, convergenceCriteria = undefined) {
    const { inputDimension, outputDimension } = trainingSet;
    const model = withMulticlassTraining(MulticlassModelBase(_extends({ inputDimension, outputDimension }, configuration)), ts => trainHMM(ts, configuration, convergenceCriteria));
    return model.train(trainingSet);
  }

  /**
   * Create a HMM Predictor from a full set of parameters (generated by trainHMM).
   * @param       {Object} params                       Model parameters
   * @param       {number} [likelihoodWindow=undefined] Likelihoow window size
   * @function
   */
  function HMMPredictor(params, likelihoodWindow = undefined) {
    const model = withHMMPrediction(withAbtractPrediction(withHMMBase(ModelBase(params)), likelihoodWindow));
    model.reset();
    return model;
  }

  /**
   * Create a Multiclass HMM Predictor from a full set of parameters
   * (generated by trainMulticlassHMM).
   * @param       {Object} params                       Model parameters
   * @param       {number} [likelihoodWindow=undefined] Likelihoow window size
   * @function
   */
  function MulticlassHMMPredictor(params, likelihoodWindow = undefined) {
    const model = withMulticlassPrediction(MulticlassModelBase(params));
    model.models = {};
    Object.keys(params.classes).forEach(label => {
      model.models[label] = HMMPredictor(params.classes[label], likelihoodWindow);
    });
    model.reset();
    return model;
  }

  /**
   * Create a Multiclass HMM Predictor from a full set of parameters
   * (generated by trainMulticlassHMM).
   * @param       {Object} params                       Model parameters
   * @param       {number} [likelihoodWindow=undefined] Likelihoow window size
   * @function
   */
  function HierarchicalHMMPredictor(params, likelihoodWindow = undefined) {
    let model = MulticlassModelBase(params);
    model.models = {};
    Object.keys(params.classes).forEach(label => {
      model.models[label] = HMMPredictor(params.classes[label], likelihoodWindow);
    });
    model = withHierarchicalHMMPrediction(withMulticlassPrediction(model));
    model.reset();
    return model;
  }

  exports.TrainingSet = TrainingSet;
  exports.trainKmeans = trainKmeans;
  exports.trainGMM = trainGMM;
  exports.trainMulticlassGMM = trainMulticlassGMM;
  exports.GMMPredictor = GMMPredictor;
  exports.MulticlassGMMPredictor = MulticlassGMMPredictor;
  exports.trainHMM = trainHMM;
  exports.trainMulticlassHMM = trainMulticlassHMM;
  exports.HMMPredictor = HMMPredictor;
  exports.MulticlassHMMPredictor = MulticlassHMMPredictor;
  exports.HierarchicalHMMPredictor = HierarchicalHMMPredictor;

  Object.defineProperty(exports, '__esModule', { value: true });

})));
//# sourceMappingURL=index.js.map
