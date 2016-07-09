#%%
# Data from: http://iweb.tms.org/SUP/selectedreadings/01-4771-201.pdf

import numpy
import matplotlib.pyplot as plt
import seaborn
import pystan

freckles = [1,
            1,
            4,
            4,
            3,
            3,
            0,
            0,
            7,
            7,
            9,
            15,
            10,
            9,
            1,
            1,
            9,
            9,
            9,
            20,
            17,
            12,
            2,
            1];

rlabels = [1] * 8 + [2] * 8 + [3] * 8
clabels = range(1, 9) + range(1, 9) + range(1, 9)

comps = numpy.array([[0, 8.4, 0, 5.6, 4.5, 2.8],
                    [0, 8.1, 0, 5.5, 4.3, 2.8],
                    [0, 4, 0.28, 5.8, 4.6, 2.9],
                    [0, 3.9, 0.31, 5.8, 4.7, 3.1],
                    [0.139, 3.9, 0, 5.8, 4.8, 3],
                    [0.14, 4, 0, 5.8, 4.8, 3],
                    [0.13, 8.2, 0.27, 5.5, 4.5, 2.8],
                    [0.119, 8.4, 0.26, 5.4, 4.5, 2.9]])

#%%

model_code = """
data {
  int<lower=1> N; // Number of different freckle samples
  int<lower=1> R; // Number of different radii
  int<lower=1> C; // Number of different compositions -- N == R * C
  int<lower=1> E; // Number of different explanatory compositions
  int<lower=1> rlabels[N]; // Radius label
  int<lower=1> clabels[N]; // Composition label
  real<lower=0.0> comps[C, E];
  real<lower=0.0> y[N];
}

parameters {
  real mu;

  real alpha[R, E];

  real<lower=0.01> sigma;
}

model {
  //mu ~ normal(1.0, 2.0);

  for (r in 1:R) {
    for (e in 1:E) {
      alpha[e] ~ normal(0.0, 5.0);
    }
  }

  for (n in 1:N) {
    y[n] ~ normal(mu + alpha[rlabels[n], 1] * comps[clabels[n], 1]
                     + alpha[rlabels[n], 2] * comps[clabels[n], 2]
                     + alpha[rlabels[n], 3] * comps[clabels[n], 3], sigma);
  }
}

generated quantities {
  real yp[C, R];

  for (r in 1:R) {
    real tmp;

    for (c in 1:C) {
      tmp <- 0.0;
      for(e in 1:E) {
        tmp <- tmp + alpha[r, e] * comps[c, e];
      }

      yp[c, r] <- normal_rng(mu + tmp, sigma);
    }
  }
}
"""

m = pystan.StanModel(model_code = model_code)
#%%

fit = m.sampling(data = {
    'N' : len(freckles),
    'R' : 3,
    'C' : 8,
    'E' : 3,
    'rlabels' : rlabels,
    'clabels' : clabels,
    'comps' : comps[:, :3],
    'y' : freckles
})

print fit