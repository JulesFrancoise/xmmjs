import test from 'ava';
import GaussianDistribution from '../src/common/gaussian_distribution';

// test('Create a Gaussian Distribution with default parameters', (t) => {
//   const g = GaussianDistribution();
//   console.log(g);
//   // console.log(g.__proto__); // eslint-disable-line
//   t.pass();
// });

test('Create a Bimodal Gaussian Distribution', (t) => {
  const g = GaussianDistribution(2, 1);
  g.mean = [1, 2];
  g.regularize([1, 1, 2]);
  g.updateInverseCovariance();
  // console.log(g);
  const h = GaussianDistribution(3, 2);
  h.mean = [5, 4, 3, 2, 1];
  h.covariance = Array.from(Array(25), () => Math.random());
  h.regularize([5, 4, 3, 2, 1]);
  h.updateInverseCovariance();
  // console.log(h);
  // console.log(g.__proto__); // eslint-disable-line
  t.pass();
});
