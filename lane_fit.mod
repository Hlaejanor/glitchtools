set P;  # photon ids

param x{P};
param y{P};
param lamb{P};

var alpha >= 0.0001;
var lucretius >= -2, <= -1;
var x0;
var y0;
var theta;

# g[i] = alpha * exp(lucretius * lamb[i])
param pi := 3.14159265358979;

var g{P} = alpha * exp(lucretius * lamb[i]);
var cos_theta = cos(theta);
var sin_theta = sin(theta);

# Projection of (x[i]-x0, y[i]-y0) onto lane direction
var proj{P} = (x[i] - x0) * cos_theta + (y[i] - y0) * sin_theta;

# Relative position in lane cycle
var modpos{P} >= 0;
s.t. mod_constraint{i in P}:
  modpos[i] = proj[i] - floor(proj[i] / g[i]) * g[i];

var dist{P} >= 0;
s.t. dist_def{i in P}:
  dist[i] = abs(modpos[i] - 0.5 * g[i]);

minimize total_squared_error:
  sum{i in P} dist[i]^2;
