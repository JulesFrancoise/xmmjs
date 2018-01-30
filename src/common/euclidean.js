/**
 * Compute the euclidean distance between to vectors
 * @param  {Array} v1
 * @param  {Array} v2
 * @return {number}
 */
export default function euclidean(v1, v2) {
  return Math.sqrt(v1
    .map((x1, i) => (x1 - v2[i]) ** 2)
    .reduce((a, x) => (a + x), 0));
}
