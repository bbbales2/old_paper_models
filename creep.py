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
  real<lower=0.0> y[T];
  real ts[T];
  real t0;
}

transformed data {
  real x_r[0];
  int x_i[0];
}

parameters {
  real<lower=0.0> sigma;
  real<lower=0.0> C;
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
  p ~ normal(0.0, 100000.0);
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
  real beta1;
  real beta2;
  real<lower=0> sigma;
}

model {
  for (n in 3:N)
      y[n] ~ normal(alpha + beta1 * y[n - 1] + beta2 * y[n - 2], sigma);
}

generated quantities {
  real yhat[N];

  yhat[1] <- y[1];
  yhat[2] <- y[2];

  for (n in 3:N)
    yhat[n] <- alpha + beta1 * yhat[n - 1] + beta2 * yhat[n - 2];//normal_rng(, sigma);
}
"""

sm = pystan.StanModel(model_code = model_code)

#%%

model_code = """
data {
  int<lower=3> T; // number of observations
  real<lower=0.0> y[T]; // observation at time T
}

parameters {
  real mu; // mean
  real<lower=0> sigma; // error scale
  vector[1] theta; // lag coefficients
}

transformed parameters {
  vector[T] epsilon; // error terms
  epsilon[1] <- y[1] - mu;

  for (t in 2:T)
    epsilon[t] <- ( y[t] - mu
    - theta[1] * epsilon[t - 1] );
}

model {
  mu ~ cauchy(0, 2.5);
  theta ~ cauchy(0, 2.5);
  sigma ~ cauchy(0, 2.5);

  for (t in 2:T)
    y[t] ~ normal(mu
      + theta[1] * epsilon[t - 1],
      sigma);
}

generated quantities {
  real yhat[T];

  yhat[1] <- y[1];

  for (t in 2:T)
    yhat[t] <- mu
      + theta[1] * epsilon[t - 1];//normal_rng(, sigma);
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
    'N' : len(ys) + 2,
    'y' : [0.0, 0.0] + list(ys),
})

print fit

#%%

fit = sm.sampling(data = {
    'T' : len(ys) + 1,
    'y' : [0.0] + list(ys),
})

print fit

#%%

y0s = [0] * 500#fit.extract()['y0'][2000:]
y2s = fit.extract()['yhat'][3500:]

plt.plot(xs, ys, '*')

idxs = numpy.random.choice(range(500), 10)

for i in idxs:
    #print 'b1', fit.extract()['beta1'][i]
    #print 'b2', fit.extract()['beta2'][i]
    plt.plot(xs, y2s[i, :])

plt.ylabel('Elongation (%)', fontsize = 20)
plt.xlabel('Time (hrs)', fontsize = 20)
plt.gcf().set_size_inches((12, 9))
plt.show()

#%%

import matplotlib.pyplot as plt
import pandas
import seaborn

Cs = fit.extract()['C'][3500:]
ps = fit.extract()['p'][3500:]
ems = fit.extract()['em'][3500:]

labels = [['samples', 'plotted'][i in idxs] for i in range(500)]

df = pandas.DataFrame({ 'C' : Cs, 'p' : ps, 'em' : ems, 'labels' : labels })

ags = None
kags = None

def plot_scatter(*args, **kwargs):
    global ags
    global kags

    ags = args
    kags = kwargs

    plt.plot(*(list(ags) + ['*']), **kwargs)

g = seaborn.PairGrid(data = df, hue = 'labels', palette = reversed(seaborn.color_palette("Set1", 8)[0:2]))
g = g.map_diag(plt.hist)
g = g.map_offdiag(plot_scatter)
plt.gcf().set_size_inches((12, 9))
plt.show()

#%%
iris = seaborn.load_dataset("iris")
g = seaborn.PairGrid(iris, hue="species")
g = g.map_diag(plt.hist)
g = g.map_offdiag(seaborn.histplot)
g = g.add_legend()