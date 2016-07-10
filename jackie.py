#%%

import scipy
import os
import numpy
import matplotlib.pyplot as plt
import seaborn
import pystan
import pandas

os.chdir('/home/bbales2/old_paper_modeling')

data1 = scipy.io.loadmat('jackie/CreepInfo_75MPa_corr.mat')['CreepInfo_075corr']
data2 = scipy.io.loadmat('jackie/CreepInfo_90MPa_corr_NoBlip.mat')['CreepInfo_090corr_NoBlip']
data3 = scipy.io.loadmat('jackie/CreepInfo_105MPa_corr.mat')['CreepInfo_105corr']

data = numpy.concatenate((data1, data2, data3), axis = 0)

data[:, 0] /= 1.0

plt.plot(data[:, 0], data[:, 1])
plt.show()

plt.semilogy(data[:, 0], data[:, 1])
plt.show()

Ns = [len(data1), len(data2), len(data3)]
Ss = [1, len(data1), len(data1) + len(data2)]
Mpa = [7.05, 90.0, 105.0]

#%%

model_code = """
data {
  int<lower=1> N; // Number of single samples
  int<lower=1> L;
  int<lower=1> Ss[L];
  int<lower=1> Ns[L];
  vector<lower=0.0>[N] t;
  vector<lower=0.0>[N] y;
  vector<lower=0.0>[L] Mpa;
}

parameters {
  real<lower=0.0> sigma;
  vector<lower=0.0>[L] a;
  vector<lower=0.0>[L] b;

  real<lower=0.0> a2;
  real<lower=0.0> b2;
  real<lower=0.0> sigma2;
}

model {
  vector[N] yhat;

  a ~ normal(0, 1.0);
  b ~ normal(0, 1.0);

  sigma ~ cauchy(0.0, 1.0);
  sigma2 ~ cauchy(0.0, 1.0);

  for (l in 1:L)
  {
      y[Ss[l] + Ns[l] / 2 : Ss[l] + Ns[l]] ~ normal(a[l] * t[Ss[l] + Ns[l] / 2 : Ss[l] + Ns[l]] + b[l], sigma);
  }

  Mpa ~ normal(a2 * a + b2, sigma2);
}

generated quantities {
  vector[N] yhat;
  vector[L] Mpahat;

  for (l in 1:L)
  {
    yhat[Ss[l]:Ss[l] + Ns[l]] <- a[l] * t[Ss[l]:Ss[l] + Ns[l]] + b[l];
  }

  Mpahat <- a2 * a + b2;
}
"""

sm = pystan.StanModel(model_code = model_code)

#%%

fit = sm.sampling(data = {
  'N' : len(data),
  'L' : len(Mpa),
  'Ss' : Ss,
  'Ns' : Ns,
  'Mpa' : Mpa,
  't' : data[:, 0],
  'y' : data[:, 1]
  })

#%%
r = fit.extract()
#%%

yhat = fit.extract()['yhat']

idxs = numpy.random.choice(range(2000), 20)

plt.plot(data[:, 0], data[:, 1], 'r*')
for i in idxs:
    plt.plot(data[:, 0], yhat[2000 + i, :])
    print i

plt.ylabel('Deformation')
plt.xlabel('Time')
plt.show()

Mpahat = fit.extract()['Mpahat']
a = fit.extract()['a']

#plt.plot(data[:, 0], data[:, 1], 'r*')
for i in idxs:
    plt.plot(a[2000 + i, :], Mpahat[2000 + i, :])
    plt.plot(a[2000 + i, :], Mpa, '*')

    print a[2000 + i, :]
    print ahat[2000 + i, :]
    print r['a2'][2000 + i]
    print r['b2'][2000 + i]
    print i

plt.show()

#%%

r = fit.extract()

#labels = [['samples', 'plotted'][i in idxs] for i in range(2000)]

df = pandas.DataFrame({ 'a0' : r['a'][2000:, 0], 'a1' : r['a'][2000:, 1], 'a2' : r['a'][2000:, 2], 'labels' : labels })

ags = None
kags = None

def plot_scatter(*args, **kwargs):
    global ags
    global kags

    ags = args
    kags = kwargs

    plt.plot(*(list(ags) + ['*']), **kwargs)

g = seaborn.PairGrid(data = df)#, hue = 'labels', palette = reversed(seaborn.color_palette("Set1", 8)[0:2])
g = g.map_diag(plt.hist)
g = g.map_offdiag(seaborn.kdeplot)#plot_scatter
plt.gcf().set_size_inches((12, 9))
plt.show()
