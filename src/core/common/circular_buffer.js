/**
 * Circular Buffer Data Structure
 */
export default class CircularBuffer {
  /**
   * Constructor
   * @param  {number} capacity Buffer capacity
   */
  constructor(capacity) {
    /**
     * Buffer capacity
     * @type {number}
     */
    this.capacity = capacity;
    this.clear();
  }

  /**
   * Clear the buffer contents
   */
  clear() {
    /**
     * Current buffer length
     * @type {Number}
     */
    this.length = 0;
    /**
     * Current index
     * @type {Number}
     */
    this.index = 0;
    /**
     * Specifies if the buffer is already full
     * @type {Boolean}
     */
    this.full = false;
    /**
     * Buffer data
     * @type {Array}
     */
    this.buffer = [];
  }

  /**
   * Push a value to the buffer
   * @param  {any} value data value (any type)
   */
  push(value) {
    if (this.full) {
      this.buffer[this.index] = value;
      this.index = (this.index + 1) % this.capacity;
    } else {
      this.buffer.push(value);
      this.length += 1;
      this.full = (this.length === this.capacity);
    }
  }

  /**
   * Get the value at a given index
   * @param  {number} idx data index
   * @return {anything}   value at index
   */
  get(idx) {
    return this.buffer[(idx + this.index) % this.capacity];
  }

  /**
   * Fill the buffer with a constant value
   * @param  {any} value data value (any type)
   */
  fill(value) {
    this.length = this.capacity;
    this.index = 0;
    this.full = true;
    this.buffer = Array(this.capacity).fill(value);
  }

  /**
   * Iterate over the buffer's data
   * @param  {Function} callback Callback function
   * (@see Array.prototype.forEach).
   */
  forEach(callback) {
    for (let i = 0; i < this.length; i += 1) {
      callback(this.buffer[(i + this.index) % this.capacity], i);
    }
  }

  /**
   * Get an array of the buffer current values (ordered)
   * @return {Array} Buffer contents
   */
  values() {
    return this.buffer.slice(this.index)
      .concat(this.buffer.slice(0, this.index));
  }
}
