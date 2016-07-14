#%%

import scipy
import os
import numpy
import matplotlib.pyplot as plt
import seaborn
import pystan
import pandas

os.chdir('/home/bbales2/old_paper_modeling')

data1 = scipy.io.loadmat('jackie/CreepInfo_75MPa_corr.mat')['CreepInfo_075corr'][::10]
data2 = scipy.io.loadmat('jackie/CreepInfo_90MPa_corr_NoBlip.mat')['CreepInfo_090corr_NoBlip'][::10]
data3 = scipy.io.loadmat('jackie/CreepInfo_105MPa_corr.mat')['CreepInfo_105corr'][::10]

#data1[:, 0] /= 100000.0
#data2[:, 0] /= 100000.0
#data3[:, 0] /= 100000.0

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
  vector<lower=0.0>[N] t;
  vector<lower=0.0>[N] y;
}

parameters {
  real<lower=0.0> sigma1;
  real<lower=0.0> a;
  real<lower=0.0> b;

  real<lower=0.0> C;
  real<lower=0.0> p;
  real<lower=0.0> e0;
  real<lower=0.0> em;
  real<lower=0.0> sigma2;
}

model {
  vector[N] yhat;

  a ~ normal(0, 1.0);
  b ~ normal(0, 1.0);

  C ~ normal(0, 1.0);
  p ~ normal(100000.0, 100000.0);
  e0 ~ normal(0, 1.0);
  em ~ normal(0, 1.0);

  sigma1 ~ cauchy(0.0, 1.0);
  sigma2 ~ cauchy(0.0, 1.0);

  y[N / 2:] ~ normal(a * t[N / 2:] + b, sigma1);
  y ~ normal(C * (t / p) ./ (1 + t / p) + em * t + e0, sigma2);
}

generated quantities {
  vector[N] yhat1;
  vector[N] yhat2;
  vector[N] yhat3;

  real ti;
  real yi;

  yhat1 <- a * t + b;
  yhat2 <- C * (t / p) ./ (1 + t / p) + em * t + e0;
  {
    real aTmp;
    real bTmp;

    aTmp <- (C / (p * pow(1 + t[1] / p, 2.0)) + em);
    bTmp <- yhat2[1] - aTmp * t[1];
    yhat3 <- aTmp * t + bTmp;

    ti <- -(bTmp - b) / (aTmp - a);
    yi <- a * ti * b;
  }
}
"""

sm = pystan.StanModel(model_code = model_code)

#%%

fit = sm.sampling(data = {
  'N' : len(data1),
  't' : data1[:, 0],
  'y' : data1[:, 1]
  })
#%%

r = fit.extract()

idxs = numpy.random.choice(range(2000), 20)
plt.plot(data1[:, 0], data1[:, 1], '*')
for i in idxs:
    plt.plot(data1[:, 0], r['yhat1'][2000 + i, :], 'g')
    plt.plot(data1[:, 0], r['yhat3'][2000 + i, :], 'b')
    plt.plot(data1[:, 0], r['yhat2'][2000 + i, :], 'r')

plt.ylim((0.0, 0.020))
plt.xlabel('Time')
plt.ylabel('e')
plt.title('Green linear fit, Red polynomial fit, Blue initial slope of polynomial')
plt.show()

plt.hist(r['ti'], bins = 'auto')
plt.title('Histogram of line intersections (2000 samples)')
plt.xlabel('Time of intersection')
plt.ylabel('Count')
plt.show()

plt.plot(r['a'][2000:], (r['C'][2000:] / (r['p'][2000:] * numpy.power(1 + data1[-1, 0] / r['p'][2000:], 2.0)) + r['em'][2000:]), '*')
plt.title('Comparison of linear slope from line and polynomial fit')
plt.xlabel('Slope from linear fit')
plt.ylabel('Slope from polynomial')
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
