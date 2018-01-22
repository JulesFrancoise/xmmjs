import Phrase from './phrase';

export default class TrainingSet {
  constructor({
    inputDimension = 1,
    outputDimension = 0,
    columnNames = null,
  } = {}) {
    this.bimodal = outputDimension > 0;
    this.inputDimension = inputDimension;
    this.outputDimension = outputDimension;
    this.dimension = inputDimension + outputDimension;
    this.columnNames = columnNames || Array(this.dimension).fill('');
    this.phrases = {};
  }

  size() {
    return Object.keys(this.phrases).length;
  }

  empty() {
    return this.size() === 0;
  }

  getPhrase(phraseIndex) {
    if (Object.keys(this.phrases).includes(phraseIndex)) {
      return this.phrases[phraseIndex];
    }
    return null;
  }

  push(phraseIndex, label = undefined, phrase = undefined) {
    this.phrases[phraseIndex] = (phrase !== undefined) ? phrase : new Phrase({
      inputDimension: this.inputDimension,
      outputDimension: this.outputDimension,
      columnNames: this.columnNames,
      label: (label !== undefined) ? label : phraseIndex.toString(),
    });
  }

  remove(phraseIndex) {
    delete this.phrases[phraseIndex];
  }

  removeClass(label) {
    this.phrases = Object.keys(this.phrases)
      .filter(i => this.phrases[i].label !== label)
      .map(i => ({ i: this.phrases[i] }))
      .reduce((x, p) => ({ ...x, ...p }), {});
  }

  clear() {
    this.phrases.clear();
    this.labels.clear();
  }

  getPhrasesOfClass(label) {
    return Object.keys(this.phrases)
      .filter(i => this.phrases[i].label === label)
      .map(i => ({ i: this.phrases[i] }))
      .reduce((x, p) => ({ ...x, ...p }), {});
  }

  labels() {
    return Object.keys(this.phrases)
      .map(i => this.phrases[i].label)
      .reduce((ll, x) => (ll.includes(x) ? ll : ll.concat([x])), []);
  }

  mean() {
    const sum = Array(this.dimension).fill(0);
    let totalLength = 0;
    Object.keys(this.phrases).forEach((i) => {
      for (let d = 0; d < this.dimension; d += 1) {
        for (let t = 0; t < this.phrases[i].size(); t += 1) {
          sum[d] += this.phrases[i].get(t, d);
        }
      }
      totalLength += this.phrases[i].size();
    });

    return sum.map(x => x / totalLength);
  }

  standardDeviation() {
    const stddev = Array(this.dimension).fill(0);
    const mean = this.mean();
    let totalLength = 0;
    Object.keys(this.phrases).forEach((i) => {
      for (let d = 0; d < this.dimension; d += 1) {
        for (let t = 0; t < this.phrases[i].size(); t += 1) {
          stddev[d] += (this.phrases[i].get(t, d) - mean[d]) ** 2;
        }
      }
      totalLength += this.phrases[i].size();
    });

    return stddev.map(x => Math.sqrt(x / totalLength));
  }

  minmax() {
    const minmax = Array.from(
      Array(this.dimension),
      () => ({ min: +Infinity, max: -Infinity }),
    );
    Object.keys(this.phrases).forEach((i) => {
      for (let d = 0; d < this.dimension; d += 1) {
        for (let t = 0; t < this.phrases[i].size(); t += 1) {
          minmax[d].min += Math.min(minmax[d].min, this.phrases[i].get(t, d));
          minmax[d].max += Math.max(minmax[d].max, this.phrases[i].get(t, d));
        }
      }
    });
    return minmax;
  }
}
