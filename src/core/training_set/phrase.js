export default class Phrase {
  constructor({
    inputDimension = 1,
    outputDimension = 0,
    columnNames = null,
    label = '',
  } = {}) {
    this.bimodal = outputDimension > 0;
    this.inputDimension = inputDimension;
    this.outputDimension = outputDimension;
    this.dimension = inputDimension + outputDimension;
    this.length = 0;
    this.label = label;
    this.inputData = [];
    this.outputData = [];
    this.columnNames = columnNames || Array(this.dimension).fill('');
  }

  get(index, dim) {
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
    return this.inputData[index][dim];
  }

  getFrame(index) {
    if (index >= this.length) {
      throw new Error('Phrase: index out of bounds');
    }
    if (this.bimodal) {
      return this.inputData[index].concat(this.outputData[index]);
    }
    return this.inputData[index];
  }

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
  }

  pushInput(observation) {
    if (!this.bimodal) {
      throw new Error('this phrase is unimodal, use `push`');
    }
    if (observation.size() !== this.inputDimension) {
      throw new Error('Observation has wrong dimension');
    }

    this.inputData.push(observation);
    this.trim();
  }

  pushOutput(observation) {
    if (!this.bimodal) {
      throw new Error('this phrase is unimodal, use `push`');
    }
    if (observation.size() !== this.outputDimension) {
      throw new Error('Observation has wrong dimension');
    }

    this.outputData.push(observation);
    this.trim();
  }

  clear() {
    this.length = 0;
    this.inputData = [];
    this.outputData = [];
  }

  clearInput() {
    this.inputData = [];
    this.trim();
  }

  clearOutput() {
    this.outputData = [];
    this.trim();
  }

  mean() {
    const mean = Array(this.dimension).fill(0);
    for (let d = 0; d < this.dimension; d += 1) {
      for (let t = 0; t < this.length; t += 1) {
        mean[d] += this.get(t, d);
      }
      mean[d] /= this.length;
    }
    return mean;
  }

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
  }

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
  }

  trim() {
    if (this.bimodal) {
      this.length = Math.min(this.inputData.length, this.outputData.length);
    }
  }
}
