#%%

import scipy
import os
import numpy
import matplotlib.pyplot as plt
import seaborn
import pystan
import pandas

os.chdir('/home/bbales2/old_paper_modeling')

data1 = scipy.io.loadmat('jackie/CreepInfo_75MPa_corr.mat')['CreepInfo_075corr'][::20]
data2 = scipy.io.loadmat('jackie/CreepInfo_90MPa_corr_NoBlip.mat')['CreepInfo_090corr_NoBlip'][::20]
data3 = scipy.io.loadmat('jackie/CreepInfo_105MPa_corr.mat')['CreepInfo_105corr'][::20]

data1[:, 0] /= 100000.0
data2[:, 0] /= 100000.0
data3[:, 0] /= 100000.0

data = numpy.concatenate((data1, data2, data3), axis = 0)

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
model_code = """
data {
  int<lower=1> T;
  real<lower=0> y[T];
  vector<lower=0>[T] x;
}

transformed data {
  real log_unif;
  log_unif <- -log(T);
}

parameters {
  real<lower=0> C;
  real<lower=0> p;
  real<lower=0.0> sigma1;

  real<lower=0> a;
  real<lower=0> b;
  real<lower=0.0> sigma2;
}

transformed parameters {
  vector[T] lp;

  lp <- rep_vector(log_unif, T);

  for (s in 1:T)
  {
    //lp[s] <- lp[s] + normal_log(y[1:s], C * (x[1:s] / p) ./ (1 + x[1:s] / p), sigma1);
    //lp[s] <- lp[s] + normal_log(y[s:], a * x[s:] + b, sigma2);

    for (t in 1:T)
    {
      if(t < s)
      {
        lp[s] <- lp[s] + normal_log(y[t], C * (x[t] / p) / (1 + x[t] / p), sigma1);//C * x[t] + p
      }
      else
      {
        lp[s] <- lp[s] + normal_log(y[t], a * x[t] + b, sigma2);
      }
    }
  }
}

model {
  C ~ normal(0.0, 1.0);
  p ~ normal(0.0, 1.0);
  sigma1 ~ normal(0.0, 1.0);

  a ~ normal(0.0, 1.0);
  b ~ normal(0.0, 1.0);
  sigma2 ~ normal(0.0, 1.0);

  increment_log_prob(log_sum_exp(lp));
}

generated quantities {
  vector[T] yhat1;
  vector[T] yhat2;

  yhat1 <- C * (x / p) ./ (1 + x / p);
  yhat2 <- a * x + b;
}
"""

sm = pystan.StanModel(model_code = model_code)
#%%

fit = sm.sampling(data = {
  'T' : len(data1),
  'x' : data1[:, 0],
  'y' : data1[:, 1]
  })

print fit
#%%
plt.plot(data1[:, 0], data1[:, 1])
plt.plot(data1[:, 0], fit.extract()['yhat1'][2500, :], 'g')
plt.plot(data1[:, 0], fit.extract()['yhat2'][2500, :], 'r')
plt.show()

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
seaborn.distplot(r['a'][2000:, 0], kde = False, fit = scipy.stats.norm)
plt.xlabel('Slope of first section')
plt.show()

seaborn.distplot(r['a'][2000:, 1], kde = False, fit = scipy.stats.norm)
plt.xlabel('Slope of second section')
plt.show()

seaborn.distplot(r['a'][2000:, 2], kde = False, fit = scipy.stats.norm)
plt.xlabel('Slope of third section')
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
