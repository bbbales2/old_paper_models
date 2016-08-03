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

#data1[:, 0] /= 100000.0
#data2[:, 0] /= 100000.0
#data3[:, 0] /= 100000.0

data = data2[len(data2) / 2:]#numpy.concatenate((data1, data2, data3), axis = 0)

plt.plot(data[:, 0], data[:, 1])
plt.show()

plt.semilogy(data[:, 0], data[:, 1])
plt.show()

Ns = [len(data1), len(data2), len(data3)]
Ss = [1, len(data1), len(data1) + len(data2)]
Mpa = [70.5, 90.0, 105.0]

#%%

model_code = """
data {
  int<lower=1> N;
  vector<lower=0.0>[N] dt;
  vector[N] dy;
}

parameters {
  real<lower=0.0> sigma1;
  real a;
}

model {
  a ~ normal(0, 1.0);
  
  sigma1 ~ cauchy(0.0, 1.0);

  dy ./ dt ~ normal(a, sigma1);
}

generated quantities {
  real slope;
  
  slope <- normal_rng(a, sigma1);
}
"""

sm = pystan.StanModel(model_code = model_code)
#%%

fit = sm.sampling(data = {
  'N' : len(data) - 1,
  'dt' : data[1:, 0] - data[:-1, 0],
  'dy' : data[1:, 1] - data[:-1, 1]
})

plt.hist(fit.extract()['slope'][2000:])

#%%

model_code = """
data {
  int<lower=1> N; // Number of single samples
  //int<lower=1> K; // Number of mixture components
  vector<lower=0.0>[N] t;
  vector<lower=0.0>[N] y;
}

parameters {
  real<lower=0.0> sigma1;
  real<lower=0.0> a;
  real<lower=0.0> b;
}

model {
  a ~ normal(0, 1.0);
  b ~ normal(0, 1.0);
  
  sigma1 ~ cauchy(0.0, 1.0);

  y ~ normal(a * t + b, sigma1);
}

//generated quantities {
//  vector[N] us;
//  
//  for (n in 1:N)
//    us[n] <- a * t[n] + b;    
//}
"""

sm = pystan.StanModel(model_code = model_code)
#sigma1 2.0e-4  6.7e-7 1.8e-5 1.7e-4 1.9e-4 2.0e-4 2.1e-4 2.4e-4  743.0    nan
#a      2.0e-8 2.5e-116.3e-10 1.9e-8 2.0e-8 2.0e-8 2.1e-8 2.1e-8  617.0    nan
#b        0.01  1.2e-5 3.0e-4   0.01   0.01   0.01   0.01   0.01  614.0    nan
#%%

fit = sm.sampling(data = {
  'N' : len(data),
  't' : data[:, 0],
  'y' : data[:, 1]
})
#%%

model_code = """
data {
  int<lower=1> N; // Number of single samples
  int<lower=1> T; // Number of timepoints
  //int<lower=1> K; // Number of mixture components
  vector<lower=0.0>[T] ts;
  vector[T] lslopes[N];
}

parameters {
  real a;
  real b;
  
  real u[T];
  real<lower=0.0> sigma[T];
}

model {
  for (t in 1:T)
    lslopes[:, t] ~ normal(u[t], sigma[t]);

  a * ts + b ~ normal(u, sigma);
}

//generated quantities {
//  vector[N] yhat1;
//  
//  for (n in 1:N)
//    yhat1[n] <- normal_rng(a * t[n] + b, sigma1);
//}
"""

sm2 = pystan.StanModel(model_code = model_code)

#%%

model_code = """
data {
  int<lower=1> N; // Number of single samples
  int<lower=1> T; // Number of timepoints
  vector<lower=0.0>[T] ts;
  vector[N] y[T];
}

parameters {
  real a;
  real b;
  
  real<lower=0.0> sigma;
}

model {
  for (t in 1:T)
    y[t] ~ normal(a * ts[t] + b, sigma);
}

//generated quantities {
//  vector[N] yhat1;
//  
//  for (n in 1:N)
//    yhat1[n] <- normal_rng(a * t[n] + b, sigma1);
//}
"""

sm3 = pystan.StanModel(model_code = model_code)

#%%

idxs = [0, 37, 73]

lslopes = numpy.log(slopes)

us = lslopes.mean(axis = 0)
stds = lslopes.std(axis = 0)

#%%
fit2 = sm2.sampling(data = {
  'N' : lslopes.shape[0],
  'T' : len(us),
  'ts' : numpy.log(Mpa),
  'lslopes' : lslopes
  })
#%%

fit3 = sm3.sampling(data = {
  'N' : slopes.shape[0],
  'T' : slopes.shape[1],
  'ts' : numpy.log(Mpa),
  'y' : numpy.log(slopes).transpose()
  })
#%%

r = fit3.extract()
r2 = fit2.extract()

seaborn.distplot(r['a'][2000:])
plt.show()

seaborn.distplot(r2['a'][2000:], bins = 'auto')
plt.show()
#%%

idxs = numpy.random.choice(range(2000), 20)
errs = []
for i in idxs:
    #errs.extend((data[:, 1] - r['yhat1'][2000 + i, :])[len(data) / 2:])
    errs.extend((data[:, 1] - r['yhat2'][2000 + i, :]))
seaborn.distplot(errs, fit = scipy.stats.norm, kde = False)
plt.show()
#%%

idxs = numpy.random.choice(range(2000), 20)
plt.plot(data[:, 0], data[:, 1], '*')
for i in idxs:
    plt.plot(data[:, 0], r['yhat1'][2000 + i, :], 'g', alpha = 0.1)
    plt.plot(data[:, 0], r['yhat2'][2000 + i, :], 'r', alpha = 0.1)
    plt.plot(data[:, 0], r['yhat3'][2000 + i, :], 'b', alpha = 0.1)

plt.ylim((0.0, 0.035))
plt.xlabel('Time')
plt.ylabel('e')
plt.title('Green linear fit, Red polynomial fit, Blue initial slope of polynomial')
plt.show()

for i in idxs:
    plt.loglog(r['a'][2000 + i, :], r['Mpahat'][2000 + i, :], 'r')#%%Mpa
plt.show()

for l in range(3):
    plt.hist(r['ti'][2000:, l], bins = 'auto')
    plt.title('Histogram of line intersections for section {0}'.format(l))
    plt.xlabel('Time of intersection')
    plt.ylabel('Count')
    plt.show()

for l in range(3):
    plt.plot(r['a'][2000:, l], r['C'][2000:, l] / r['p'][2000:, l] + r['em'][2000:, l], '*')
    plt.title('Comparison of maximum and minimum slopes for section {0}'.format(l))
    plt.xlabel('Minimum slopes (from linear fit)')
    plt.ylabel('Maximum slopes (from polynomial fit)')
    plt.show()

seaborn.distplot(r['n'][2000:])
plt.xlabel('Values of n')
plt.show()

labels = [['samples', 'plotted'][i in idxs] for i in range(2000)]

df = pandas.DataFrame({ 'C' : r['C'][2000:], 'p' : r['p'][2000:], 'em' : r['em'][2000:], 'e0' : r['e0'][2000:], 'labels' : labels })

ags = None
kags = None

def plot_scatter(*args, **kwargs):
    global ags
    global kags

    ags = args
    kags = kwargs

    plt.plot(*(list(ags) + ['*']), **kwargs)

g = seaborn.PairGrid(data = df, hue = 'labels', palette = reversed(seaborn.color_palette("Set1", 8)[0:2]))#
g = g.map_diag(plt.hist)
g = g.map_offdiag(plot_scatter)
plt.gcf().set_size_inches((12, 9))
plt.show()
#%%
model_code = """
data {
  int<lower=1> N; // Number of samples
  int<lower=1> L;
  vector<lower=0.0>[N] y[L];
  vector[L] Mpa;
}

parameters {
  real<lower=0.0> sigma;
  real a;
  real b;
}

model {
  for (l in 1:L)
  {
    y[l] ~ lognormal(a * log(Mpa[l]) + b, sigma);
  }
}

generated quantities {
  vector[L] yhat;

  for (l in 1:L)
    yhat[l] <- lognormal_rng(a * log(Mpa[l]) + b, sigma);
}
"""

sm2 = pystan.StanModel(model_code = model_code)
#%%
slopes = r['a'][:3000]

fit2 = sm2.sampling(data = {
  'y' : slopes.transpose(),
  'N' : slopes.shape[0],
  'L' : slopes.shape[1],
  'Mpa' : Mpa
})
#%%
r2 = fit2.extract()

idxs = numpy.random.choice(range(2000), 20)
for i in idxs:
    plt.plot(numpy.log(Mpa), numpy.log(r2['yhat'][2000 + i, :]), '-*')
plt.title('')
plt.xlabel('log(Mpa)')
plt.ylabel('log(minimum_slope)')
plt.show()

bp = plt.boxplot(numpy.log(slopes))
plt.setp(bp['boxes'], color='green')
plt.setp(bp['whiskers'], color='green')
plt.setp(bp['fliers'], marker='None')
bp = plt.boxplot(numpy.log(r2['yhat'][3500:, :]))
plt.setp(bp['boxes'], color='black')
plt.setp(bp['whiskers'], color='black')
plt.setp(bp['fliers'], marker='None')
plt.ylabel('Log(min slopes)')
plt.xlabel('These 1, 2, 3 are not spaced properly, but they represent the different MPas')
plt.show()

seaborn.distplot(r2['a'][2000:])
plt.xlabel('"a" in log(min_slope) ~ a * log(Mpa) + b')
plt.show()

seaborn.distplot(r2['b'][2000:])
plt.xlabel('"b" in log(min_slope) ~ a * log(Mpa) + b')
plt.show()

seaborn.distplot(r2['sigma'][2000:])
plt.show()
#%%
r['yhat1'][2000 + i, :164]
#%%r['yhat1'][2000 + i, :164]
#a[l] * t[s : se] + b[l]
a = numpy.concatenate([(r['a'][2000 + i, 0] * data1[:, 0] + r['b'][2000 + i, 0]) - data1[:, 1] for i in range(100, 110)])

for dist_name in ['norm', 'cauchy', 'laplace', 'gumbel_r']:#[ 'alpha', 'anglit', 'arcsine', 'beta', 'betaprime', 'bradford', 'burr', 'cauchy', 'chi', 'chi2', 'cosine', 'dgamma', 'dweibull', 'erlang', 'expon', 'exponweib', 'exponpow', 'f', 'fatiguelife', 'fisk', 'foldcauchy', 'foldnorm', 'frechet_r', 'frechet_l', 'genlogistic', 'genpareto', 'genexpon', 'genextreme', 'gausshyper', 'gamma', 'gengamma', 'genhalflogistic', 'gilbrat', 'gompertz', 'gumbel_r', 'gumbel_l', 'halfcauchy', 'halflogistic', 'halfnorm', 'hypsecant', 'invgamma', 'invgauss', 'invweibull', 'johnsonsb', 'johnsonsu', 'ksone', 'kstwobign', 'laplace', 'logistic', 'loggamma', 'loglaplace', 'lognorm', 'lomax', 'maxwell', 'mielke', 'nakagami', 'ncx2', 'ncf', 'nct', 'norm', 'pareto', 'pearson3', 'powerlaw', 'powerlognorm', 'powernorm', 'rdist', 'reciprocal', 'rayleigh', 'rice', 'recipinvgauss', 'semicircular', 't', 'triang', 'truncexpon', 'truncnorm', 'tukeylambda', 'uniform', 'vonmises', 'wald', 'weibull_min', 'weibull_max', 'wrapcauchy']:
    seaborn.distplot(a, fit = getattr(scipy.stats, dist_name), kde = False)
    plt.title(dist_name)
    plt.show()

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
  real<lower=0> e0;

  real<lower=0> a;
  real<lower=0> b;

  real<lower=0.0> sigma;
}

transformed parameters {
  vector[T] lp;

  lp <- rep_vector(log_unif, T);
  //for (t in 1:T)
  //  lp[t] <- normal_log(t, 51.0, 10.0);

  for (s in 1:T)
  {
    //lp[s] <- lp[s] + normal_log(y[1:s], C * (x[1:s] / p) ./ (1 + x[1:s] / p), sigma1);
    //lp[s] <- lp[s] + normal_log(y[s:], a * x[s:] + b, sigma2);

    for (t in 1:T)
    {
      if(t < s)
      {
        lp[s] <- lp[s] + normal_log(y[t], C * (x[t] / p) / (1 + x[t] / p) + e0, sigma);//C * x[t] + p
      }
      else
      {
        lp[s] <- lp[s] + normal_log(y[t], a * x[t] + b, sigma);
      }
    }
  }
}

model {
  C ~ uniform(0.0, 1.0);
  p ~ uniform(0.0, 10.0);
  e0 ~ uniform(0.0, 1.0);

  a ~ uniform(0.0, 1.0);
  b ~ uniform(0.0, 1.0);

  sigma ~ normal(0.0, 1.0);

  increment_log_prob(log_sum_exp(lp));
}

generated quantities {
  vector[T] yhat1;
  vector[T] yhat2;
  vector[T] yhat3;
  int<lower=1,upper=T> s;

  yhat1 <- C * (x / p) ./ (1 + x / p) + e0;
  yhat2 <- a * x + b;
  yhat3 <- C * x / p + e0;
  s <- categorical_rng(softmax(lp));
}
"""

sm = pystan.StanModel(model_code = model_code)


fit = sm.sampling(data = {
  'T' : len(data1),
  'x' : data1[:, 0],
  'y' : data1[:, 1]
  })

print fit
#%%
f = fit.extract()

idxs = numpy.random.choice(range(2000), 20)
for i in idxs:
    plt.plot(data1[:, 0] * 100000.0, f['yhat1'][2000 + i, :], 'g')
    plt.plot(data1[:, 0] * 100000.0, f['yhat3'][2000 + i, :], 'b')
    plt.plot(data1[:, 0] * 100000.0, f['yhat2'][2000 + i, :], 'r')

plt.ylim((0.0, 0.020))
plt.plot(data1[:, 0] * 100000.0, data1[:, 1], '*')
plt.xlabel('Time')
plt.ylabel('e')

ax2 = plt.gca().twinx()

lp = f['lp'][2000:]
for i in range(2000):
    lp[i] -= max(lp[i])
    lp[i] = numpy.exp(lp[i])
    lp[i] /= numpy.sum(lp[i])

ax2.plot(data1[:, 0] * 100000.0, numpy.mean(lp, axis = 0), 'c--o')
#ax2.hist(f['s'][2000:])
plt.title('Teal is the probability of switching from model 1 to model 2')
plt.ylabel('Probability (for teal line)')

plt.show()

print

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

labels = [['samples', 'plotted'][i in idxs] for i in range(2000)]

df = pandas.DataFrame({ 'a0' : r['C'][2000:], 'a1' : r['b'][2000:], 'a2' : r['a'][2000:], 'labels' : labels })

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
