#%%

import os
import numpy
import matplotlib.pyplot as plt
import pystan

os.chdir('/home/bbales2/old_paper_modeling/')

data = []
f = open('data.txt')
f.readline()

for line in f:
    x, y = line.split(',')

    x = float(x) / 1.0
    y = max(0.0, float(y))

    data.append((x, y))

data = sorted(data, key = lambda x : x[0])[3:]
#data[0] = (0.0, 0.0)

xs, ys = zip(*data)

f.close()

#%%
model_code = """
functions {
  real[] sho(real t,
             real[] y,
             real[] theta,
             real[] x_r,
             int[] x_i) {
    real dydt[1];
    dydt[1] <- (theta[1] / theta[2]) / pow(1 + t / theta[2], 2) + theta[3];
    return dydt;
  }
}

data {
  int<lower=1> T;
  real y[T];
  real ts[T];
  real t0;
}

transformed data {
  real x_r[0];
  int x_i[0];
}

parameters {
  real<lower=0.0> sigma;
  real<lower=0.0> Cp;
  real<lower=0.0> p;
  real<lower=0.0> em;
}

transformed parameters {
  real theta[3];

  theta[1] <- C;
  theta[2] <- p;
  theta[3] <- em;
}

model {
  real y0[1];
  real yhat[T, 1];

  sigma ~ cauchy(0, 2.5);

  C ~ normal(0.0, 10.0);
  p ~ normal(0.0, 10000.0);
  em ~ normal(0.0, 1.0);
  y0[1] <- 0.0;

  yhat <- integrate_ode(sho, y0, t0, ts, theta, x_r, x_i);

  for (t in 1:T)
    y[t] ~ normal(yhat[t], sigma);
}

generated quantities {
  real yhat[T, 1];
  real y0[1];

  y0[1] <- 0.0;

  yhat <- integrate_ode(sho, y0, t0, ts, theta, x_r, x_i);
}
"""

sm = pystan.StanModel(model_code = model_code)

#%%

model_code = """
data {
  int<lower=0> N;
  vector[N] y;
}

parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}

model {
  for (n in 2:N)
      y[n] ~ normal(alpha + beta * y[n-1], sigma);
}

generated quantities {
  real yhat[N];

  yhat[1] <- y[1];

  for (n in 2:N)
    yhat[n] <- normal_rng(alpha + beta * yhat[n - 1], sigma);
}
"""

sm = pystan.StanModel(model_code = model_code)

#%%

model_code = """
data {
  int<lower=3> T; // number of observations
  vector[T] y;
  // observation at time T
}

parameters {
  real mu; // mean
  real<lower=0> sigma; // error scale
  vector[2] theta; // lag coefficients
}

transformed parameters {
  vector[T] epsilon; // error terms
  epsilon[1] <- y[1] - mu;
  epsilon[2] <- y[2] - mu - theta[1] * epsilon[1];

  for (t in 3:T)
    epsilon[t] <- ( y[t] - mu
    - theta[1] * epsilon[t - 1]
    - theta[2] * epsilon[t - 2] );
}

model {
  mu ~ cauchy(0,2.5);
  theta ~ cauchy(0,2.5);
  sigma ~ cauchy(0,2.5);

  for (t in 3:T)
    y[t] ~ normal(mu
      + theta[1] * epsilon[t - 1]
      + theta[2] * epsilon[t - 2],
      sigma);
}

generated quantities {
  real yhat[T];

  yhat[1] <- y[1];
  yhat[2] <- y[2];

  for (t in 3:T)
    yhat[t] <- normal_rng(mu
      + theta[1] * epsilon[t - 1]
      + theta[2] * epsilon[t - 2],
      sigma);
}
"""

sm = pystan.StanModel(model_code = model_code)

#%%

fit = sm.sampling(data = {
    'T' : len(ys),
    'y' : ys,
    't0' : 0.0,
    'ts' : xs
})

print fit

#%%

fit = sm.sampling(data = {
    'N' : len(ys) + 1,
    'y' : [0.0] + list(ys),
})

print fit

#%%

fit = sm.sampling(data = {
    'T' : len(ys) + 1,
    'y' : [0.0] + list(ys),
})

print fit

#%%

y0s = [0] * 2000#fit.extract()['y0'][2000:]
y2s = fit.extract()['yhat'][2000:]

plt.plot(xs, ys, '*')

for i in numpy.random.choice(range(2000), 50):
    plt.plot(xs, y2s[i, 1:])

plt.show()
