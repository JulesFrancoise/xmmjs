import { isBaseModel } from '../core/model_base_mixin';
import euclidean from '../common/euclidean';

const kMeansTrainingPrototype = {
  train(trainingSet) {
    if (!trainingSet || trainingSet.empty()) {
      throw new Error('The training set is empty');
    }

    this.params.centers = Array.from(
      Array(this.params.clusters),
      () => new Array(this.params.dimension).fill(0),
    );

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

    for (
      let trainingNbIterations = 0;
      trainingNbIterations < this.trainingConfig.maxIterations;
      trainingNbIterations += 1
    ) {
      const previousCenters = this.params.centers;

      this.updateCenters(previousCenters, trainingSet);

      let meanClusterDistance = 0;
      let maxRelativeCenterVariation = 0;
      for (let k = 0; k < this.params.clusters; k += 1) {
        for (let l = 0; l < this.params.clusters; l += 1) {
          if (k !== l) {
            meanClusterDistance += euclidean(
              this.params.centers[k],
              this.params.centers[l],
            );
          }
        }
        maxRelativeCenterVariation = Math.max(
          euclidean(
            previousCenters[k],
            this.params.centers[k],
          ),
          maxRelativeCenterVariation,
        );
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
    const indices = Array.from(
      Array(phrase.length),
      () => Math.floor(Math.random() * this.params.clusters),
    );
    const pointsPerCluster = indices.reduce(
      (ppc, i) => {
        const p = ppc;
        p[i] += 1;
        return p;
      },
      Array(this.params.clusters).fill(0),
    );
    for (let i = 0; i < indices.length; i += 1) {
      const clustIdx = indices[i];
      for (let d = 0; d < this.params.dimension; d += 1) {
        this.params.centers[clustIdx][d] += phrase.get(i, d);
      }
    }
    this.params.centers.forEach((_, c) => {
      this.params.centers[c] = this.params.centers[c]
        .map(x => x / pointsPerCluster[c]);
    });
  },

  initializeClustersForgy(trainingSet) {
    const phrase = trainingSet.getPhrase(trainingSet.indices()[0]);
    const indices = Array.from(
      Array(this.params.clusters),
      () => Math.floor(Math.random() * phrase.length),
    );
    this.params.centers = indices.map(i => phrase.getFrame(i));
  },

  updateCenters(previousCenters, trainingSet) {
    this.params.centers = Array.from(Array(this.params.clusters), () =>
      new Array(this.params.dimension).fill(0));
    const numFramesPerCluster = Array(this.params.clusters).fill(0);
    trainingSet.forEach((phrase) => {
      for (let t = 0; t < phrase.length; t += 1) {
        const frame = phrase.getFrame(t);
        let minDistance = euclidean(frame, previousCenters[0]);
        let clusterMembership = 0;
        for (let k = 1; k < this.params.clusters; k += 1) {
          const distance = euclidean(
            frame,
            previousCenters[k],
            this.params.dimension,
          );
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
  },
};

export default function withKMeansTraining(
  o,
  clusters,
  trainingConfiguration = {},
) {
  if (!isBaseModel(o)) {
    throw new Error('The base object must include a standard set of parameters (`params` key), @see `ModelBase`.');
  }
  const trainingConfig = Object.assign({
    initialization: 'random',
    relativeDistanceThreshold: 1e-3,
    minIterations: 5,
    maxIterations: 100,
  }, trainingConfiguration);
  const model = Object.assign(o, kMeansTrainingPrototype, {
    trainingConfig,
  });
  model.params.clusters = clusters;
  return model;
}
