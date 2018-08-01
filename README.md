# xmmjs - A Javascript port of the XMM Library

From Ircam's XMM library (https://github.com/Ircam-RnD/xmm):

> XMM is a portable, cross-platform C++ library that implements Gaussian Mixture Models and Hidden Markov Models for recognition and regression. The XMM library was developed for movement interaction in creative applications and implements an interactive machine learning workflow with fast training and continuous, real-time inference.

## Installing

```shell
yarn add @JulesFrancoise/xmmjs
# OR
npm install --save @JulesFrancoise/xmmjs
```

## Getting Started

```js
// Create a training set to host the training data
const ts = TrainingSet({ inputDimension: 3 });

// Add a new phrase to the training set, and record data frames
const phrase = ts.push(0, 'default');
for (let i = 0; i < 1000; i += 1) {
  const frame = ...; // get data from somewhere
  phrase.push(frame);
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
const gmmParams = trainGMM(ts, configuration);

// Create a predictor to perform real-time recognition
const predictor = GMMPredictor(gmmParams);
predictor.reset();
predictor.predict([0, 0, 0]);
```

## Credits

xmmjs has been developed at [LIMSI-CNRS](https://www.limsi.fr/en/) by [Jules Françoise](https://www.julesfrancoise.com), and is released under the MIT Licence.

`xmmjs` is based on the XMM C++ Library developed at Ircam-Centre Pompidou:
https://github.com/Ircam-RnD/xmm

## Developing

TODO => see https://github.com/wearehive/project-guidelines/blob/master/README.sample.md
