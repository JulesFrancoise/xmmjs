export default function ModelBase({
  inputDimension,
  outputDimension,
  ...parameters
}) {
  const p = parameters;
  delete p.bimodal;
  delete p.inputDimension;
  delete p.outputDimension;
  delete p.dimension;
  return {
    params: {
      ...p,
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
      },
    },
  };
}

export function isBaseModel(o) {
  if (!Object.keys(o).includes('params')) return false;
  const keys = ['bimodal', 'inputDimension', 'outputDimension', 'dimension'];
  return keys.map(key => Object.keys(o.params).includes(key))
    .reduce((a, b) => a && b, true);
}
