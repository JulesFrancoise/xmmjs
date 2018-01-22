export function createEllipse() {
  return {
    x: 0,
    y: 0,
    width: 0,
    height: 0,
    angle: 0,
  };
}

export function covariance2ellipse(cxx, cxy, cyy) {
  const gaussianEllipse = createEllipse();
  gaussianEllipse.x = 0;
  gaussianEllipse.y = 0;

  // Compute Eigen Values to get width, height and angle
  const trace = cxx + cyy;
  const determinant = (cxx * cyy) - (cxy * cxy);
  const eigenVal1 = 0.5 * (trace + Math.sqrt((trace ** 2) - (4 * determinant)));
  const eigenVal2 = 0.5 * (trace - Math.sqrt((trace ** 2) - (4 * determinant)));
  gaussianEllipse.width = Math.sqrt(5.991 * eigenVal1);
  gaussianEllipse.height = Math.sqrt(5.991 * eigenVal2);
  gaussianEllipse.angle = Math.atan(cxy / (eigenVal1 - cyy));
  if (Number.isNaN(gaussianEllipse.angle)) {
    gaussianEllipse.angle = Math.PI / 2;
  }

  return gaussianEllipse;
}
