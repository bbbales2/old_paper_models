#%%

import scipy
import os
import numpy
import matplotlib.pyplot as plt
import seaborn
import pystan
import pandas

os.chdir('/home/bbales2/old_paper_modeling')

data = scipy.io.loadmat('jackie/CreepInfo_75MPa_corr_NoBlip.mat')['CreepInfo_075corr_NoBlip']

data[:, 0] /= 1.0

plt.plot(data[:, 0], data[:, 1])
plt.show()

plt.semilogy(data[:, 0], data[:, 1])
plt.show()

#%%

model_code = """
data {
  int<lower=1> N; // Number of single samples
  vector<lower=0.0>[N] t;
  vector<lower=0.0>[N] y;
}

parameters {
  real<lower=0.0> sigma;
  real<lower=0.0> a;
  real<lower=0.0> b;
  real<lower=0.0> c;
}


model {
  vector[N] yhat;

  a ~ normal(0, 1.0);
  b ~ normal(0, 100000.0);
  c ~ normal(0, 1.0);

  yhat <- a * (t / b) ./ (1 + t / b) + c * t;//(1.0 - exp(-t / b))

  y ~ normal(yhat, sigma);
}

generated quantities {
  vector[N] yhat;

  yhat <- a * (t / b) ./ (1 + t / b) + c * t;
}
"""

sm = pystan.StanModel(model_code = model_code)

#%%

model_code = """
data {
  int<lower=1> N; // Number of single samples
  vector<lower=0.0>[N] t;
  vector<lower=0.0>[N] y;
}

parameters {
  real<lower=0.0> sigma1;
  real<lower=0.0> sigma2;
  real<lower=0.0> a;
  real<lower=0.0, upper=300000.0> b;
  real<lower=0.0> c;
  real<lower=0.0> d;
}


model {
  vector[N] yhat;

  a ~ normal(0, 1.0);
  b ~ normal(180000, 10000.0);
  c ~ normal(0, 1.0);
  d ~ normal(0, 1.0);

  sigma1 ~ cauchy(0.0, 1.0);
  sigma2 ~ normal(0.0, 0.1);

  for (n in 1:N)
  {
    if (t[n] < b)
    {
      y[n] ~ normal(a, sigma1);
    }
    else
    {
      y[n] ~ normal(c * t[n] + d, sigma2);
    }
  }

  //yhat <- a * (t / b) ./ (1 + t / b) + c * t;//(1.0 - exp(-t / b))

  //y ~ normal(yhat, sigma);
}

//generated quantities {
//  vector[N] yhat;
//
//  yhat <- a * (t / b) ./ (1 + t / b) + c * t;
//}
"""

sm = pystan.StanModel(model_code = model_code)

#%%

fit = sm.sampling(data = {
  'N' : len(data),
  't' : data[:, 0],
  'y' : data[:, 1]
  })

#%%
r = fit.extract()

df = pandas.DataFrame({ 'a' : r['a'][2000:], 'b' : r['b'][2000:], 'c' : r['c'][2000:], 'd' : r['d'][2000:], 'sigma1' : r['sigma1'][2000:], 'sigma2' : r['sigma2'][2000:] })

ags = None
kags = None

def plot_scatter(*args, **kwargs):
    global ags
    global kags

    ags = args
    kags = kwargs

    plt.plot(*(list(ags) + ['*']), **kwargs)

g = seaborn.PairGrid(data = df)
g = g.map_diag(plt.hist)
g = g.map_offdiag(plot_scatter)
plt.gcf().set_size_inches((12, 9))
plt.show()
#%%

yhat = fit.extract()['yhat']

idxs = numpy.random.choice(range(2000), 20)

plt.plot(data[:, 0], data[:, 1], 'r*')
for i in idxs:
    plt.plot(data[:, 0], yhat[2000 + i, :])
    print i

plt.show()

r = fit.extract()

labels = [['samples', 'plotted'][i in idxs] for i in range(2000)]

df = pandas.DataFrame({ 'a' : r['a'][2000:], 'b' : r['b'][2000:], 'c' : r['c'][2000:], 'labels' : labels })

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
