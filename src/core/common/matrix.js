const kEpsilonPseudoInverse = 1.0e-9;

export default class Matrix {
  constructor(nrows = 0, ncols = -1) {
    /**
     * Number of rows of the matrix
     * @type {Number}
     */
    this.nrows = nrows;
    /**
     * Number of columns of the matrix
     * @type {Number}
     */
    this.ncols = ncols < 0 ? nrows : ncols;
    /**
     * Matrix data
     * @type {Number}
     */
    this.data = Array(this.nrows * this.ncols).fill(0);
  }

  /**
   * Compute the Sum of the matrix
   * @return {Number} Sum of all elements in the matrix
   */
  sum() {
    return this.data.reduce((a, b) => a + b, 0);
  }

  /**
   * Pretty-Print the matrix
   */
  print() {
    for (let i = 0; i < this.nrows; i += 1) {
      let line = '';
      for (let j = 0; j < this.ncols; j += 1) {
        line += `${this.data[(i * this.ncols) + j]} `.padStart(10);
      }
      console.log(line); // eslint-disable-line no-console
    }
  }

  /**
   * Compute the transpose matrix
   * @return {Matrix}
   */
  transpose() {
    const out = new Matrix(this.ncols, this.nrows);
    for (let i = 0; i < this.ncols; i += 1) {
      for (let j = 0; j < this.nrows; j += 1) {
        out.data[(i * this.nrows) + j] = this.data[(j * this.ncols) + i];
      }
    }
    return out;
  }

  /**
   * Compute the product of matrices
   * @param  {Matrix} mat Second matrix
   * @return {Matrix}     Product of the current matrix by `mat`
   */
  product(mat) {
    if (this.ncols !== mat.nrows) {
      throw new Error('Wrong dimensions for matrix product');
    }
    const out = new Matrix(this.nrows, mat.ncols);
    for (let i = 0; i < this.nrows; i += 1) {
      for (let j = 0; j < mat.ncols; j += 1) {
        out.data[(i * mat.ncols) + j] = 0;
        for (let k = 0; k < this.ncols; k += 1) {
          out.data[(i * mat.ncols) + j] +=
            this.data[(i * this.ncols) + k] * mat.data[(k * mat.ncols) + j];
        }
      }
    }
    return out;
  }

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
  }

  /**
   * Compute the Gauss-Jordan Inverse of a Square Matrix
   * !!! Determinant (computed with the inversion
   */
  gaussJordanInverse() {
    if (this.nrows !== this.ncols) {
      throw new Error('Gauss-Jordan inversion: Cannot invert Non-square matrix');
    }
    let determinant = 1;
    const mat = new Matrix(this.nrows, this.ncols * 2);
    const newMat = new Matrix(this.nrows, this.ncols * 2);
    const n = this.nrows;

    // Create matrix
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) {
        mat.data[(i * 2 * n) + j] = this.data[(i * n) + j];
      }
      mat.data[(i * 2 * n) + n + i] = 1;
    }

    for (let k = 0; k < n; k += 1) {
      let i = k;
      while (Math.abs(mat.data[(i * 2 * n) + k]) < kEpsilonPseudoInverse()) {
        i += 1;
        if (i === n) {
          throw new Error('Non-invertible matrix');
        }
      }
      determinant *= mat.data[(i * 2 * n) + k];

      // if found > Exchange lines
      if (i !== k) {
        mat.swapLines(i, k);
      }

      newMat.data = mat.data;

      for (let j = 0; j < 2 * n; j += 1) {
        newMat.data[(k * 2 * n) + j] /= mat.data[(k * 2 * n) + k];
      }
      for (let ii = 0; ii < n; ii += 1) {
        if (ii !== k) {
          for (let j = 0; j < 2 * n; j += 1) {
            newMat.data[(ii * 2 * n) + j] -=
                mat.data[(ii * 2 * n) + k] *
                newMat.data[(k * 2 * n) + j];
          }
        }
      }
      mat.data = newMat.data;
    }

    const dst = new Matrix(this.nrows, this.ncols);
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) {
        dst.data[(i * n) + j] = mat.data[(i * 2 * n) + n + j];
      }
    }
    return { determinant, matrix: dst };
  }

  /**
   * Swap 2 lines of the matrix
   * @param  {[type]} i index of the first line
   * @param  {[type]} j index of the second line
   */
  swapLines(i, j) {
    for (let k = 0; k < this.ncols; k += 1) {
      const tmp = this.data[(i * this.ncols) + k];
      this.data[(i * this.ncols) + k] = this.data[(j * this.ncols) + k];
      this.data[(j * this.ncols) + k] = tmp;
    }
  }

  /**
   * Swap 2 columns of the matrix
   * @param  {[type]} i index of the first column
   * @param  {[type]} j index of the second column
   */
  swapColumns(i, j) {
    for (let k = 0; k < this.nrows; k += 1) {
      const tmp = this.data[(k * this.ncols) + i];
      this.data[(k * this.ncols) + i] = this.data[(k * this.ncols) + j];
      this.data[(k * this.ncols) + j] = tmp;
    }
  }
}
