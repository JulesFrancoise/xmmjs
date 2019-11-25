# xmmjs - A Javascript port of the XMM Library

From Ircam's XMM library (https://github.com/Ircam-RnD/xmm):

> XMM is a portable, cross-platform C++ library that implements Gaussian Mixture Models and Hidden Markov Models for recognition and regression. The XMM library was developed for movement interaction in creative applications and implements an interactive machine learning workflow with fast training and continuous, real-time inference.

## Installing

#### Node.js

```shell
yarn add xmmjs
# OR
npm install --save xmmjs
```

#### In the browser

```html
<script src="https://cdn.jsdelivr.net/gh/JulesFrancoise/xmmjs/dist/index.js"></script>
```

## Getting Started

Basic example of GMM-based recognition

```js
const xmm = require('xmm');

// Create a training set to host the training data
const ts = xmm.TrainingSet({ inputDimension: 3 });

// Add a new phrase to the training set, and record data frames
const phrase1 = ts.push(0, 'one');
for (let i = 0; i < 1000; i += 1) {
  const frame = Array.from(Array(3), () => Math.random()); // get data from somewhere
  phrase1.push(frame);
}
const phrase2 = ts.push(1, 'two');
for (let i = 0; i < 1000; i += 1) {
  const frame = Array.from(Array(3), () => 1 + Math.random()); // get data from somewhere
  phrase2.push(frame);
}

// Train the GMM with the given configuration
const configuration = {
  gaussians: 3,
  regularization: {
    absolute: 1e-1,
    relative: 1e-10,
  },
  covarianceMode: 'full',
};
const gmmParams = xmm.trainMulticlassGMM(ts, configuration);

// Create a predictor to perform real-time recognition
const predictor = xmm.MulticlassGMMPredictor(gmmParams);
predictor.reset();

predictor.predict([0.5, 0.5, 0.5]);
console.log('results (0.5)', predictor.results);

predictor.predict([1.5, 1.5, 1.5]);
console.log('results (1.5)', predictor.results);
```

## Credits

xmmjs has been developed at [LIMSI-CNRS](https://www.limsi.fr/en/) by [Jules Françoise](https://www.julesfrancoise.com), and is released under the MIT Licence.

`xmmjs` is based on the XMM C++ Library developed at Ircam-Centre Pompidou:
https://github.com/Ircam-RnD/xmm
