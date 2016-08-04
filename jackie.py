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

data = numpy.concatenate((data1, data2, data3), axis = 0)

plt.plot(data[:, 0], data[:, 1])
plt.show()

plt.semilogy(data[:, 0], data[:, 1])
plt.show()

Ns = [len(data1), len(data2), len(data3)]
Ss = [1, len(data1), len(data1) + len(data2)]
Mpa = numpy.array([70.5, 90.0, 105.0])

#%%

model_code = """
data {
  int<lower=1> N; // Number of single samples
  int<lower=1> L;
  int<lower=1> Ss[L];
  int<lower=1> Ns[L];
  //int<lower=1> K; // Number of mixture components
  vector<lower=0.0>[N] t;
  vector<lower=0.0>[N] y;
}

parameters {
  real<lower=0.0> sigma1[L];
  vector<lower=0.0>[L] a;
  real<lower=0.0> b[L];

  real<lower=0.0> C[L];
  real<lower=0.0> p[L];
  real<lower=0.0> e0[L];
  real<lower=0.0> em[L];
  real<lower=0.0> sigma2[L];
}

model {
  vector[N] yhat;
  //real ps[K];

  a ~ normal(0, 1.0);
  b ~ normal(0, 1.0);

  C ~ normal(0, 1.0);
  p ~ normal(100000.0, 100000.0);
  e0 ~ normal(0, 1.0);
  em ~ normal(0, 1.0);

  sigma1 ~ cauchy(0.0, 1.0);
  sigma2 ~ cauchy(0.0, 1.0);

  for (l in 1:L)
  {
    vector[Ns[l]] trange;

    int s;
    int m;
    int se;

    s <- Ss[l];
    m <- Ss[l] + 1 * Ns[l] / 2;
    se <- Ss[l] + Ns[l] - 1;

    y[m : se] ~ normal(a[l] * t[m : se] + b[l], sigma1[l]);

    trange <- t[s : se] - t[s];
    y[s : se] ~ normal(C[l] * (trange / p[l]) ./ (1 + trange / p[l]) + em[l] * trange + e0[l], sigma2[l]);
  }
}

generated quantities {
  vector[N] yhat1;
  vector[N] yhat2;
  vector[N] yhat3;
  vector[N] yhat1m;
  vector[N] yhat2m;
  vector[N] yhat3m;
  vector[N] yhat1f[L];
  vector[N] yhat2f[L];

  real ti[L];
  real yi[L];

  for (l in 1:L)
  {
    int s;
    int se;
    vector[Ns[l]] trange;

    s <- Ss[l];
    se <- Ss[l] + Ns[l] - 1;

    for (j in s:se)
      yhat1[j] <- normal_rng(a[l] * t[j] + b[l], sigma1[l]);

    for (j in 1:N)
      yhat1f[l][j] <- normal_rng(a[l] * t[j] + b[l], sigma1[l]);

    yhat1m[s : se] <- a[l] * t[s : se] + b[l];

    trange <- t[s : se] - t[s];
    for (j in s:se)
      yhat2[j] <- normal_rng(C[l] * (trange[j - s + 1] / p[l]) ./ (1 + trange[j - s + 1] / p[l]) + em[l] * trange[j - s + 1] + e0[l], sigma2[l]);

    for (j in 1:N)
      yhat2f[l][j] <- normal_rng(C[l] * ((t[j] - t[s]) / p[l]) ./ (1 + (t[j] - t[s]) / p[l]) + em[l] * (t[j] - t[s]) + e0[l], sigma2[l]);

    yhat2m[s : se] <- C[l] * (trange / p[l]) ./ (1 + trange / p[l]) + em[l] * trange + e0[l];

    {
      real aTmp;
      real bTmp;

      aTmp <- C[l] / p[l] + em[l];
      bTmp <- yhat2[s];
      for (j in s:se)
        yhat3[j] <- normal_rng(aTmp * trange[j - s + 1] + bTmp, sigma2[l]);

      yhat3m[s : se] <- aTmp * trange + bTmp;

      ti[l] <- -(bTmp - aTmp * t[s] - b[l]) / (aTmp - a[l]);
      yi[l] <- a[l] * ti[l] + b[l];
    }
  }
}
"""

sm = pystan.StanModel(model_code = model_code)

#%%

fit = sm.sampling(data = {
  'L' : len(Ns),
  'Ss' : Ss,
  'Ns' : Ns,
  'N' : len(data),
  't' : data[:, 0],
  'y' : data[:, 1]
})

#%%

fit = sm.sampling(data = {
  'N' : len(data1),
  't' : data1[:, 0],
  'y' : data1[:, 1]
  })
#%%

r = fit.extract()

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
plt.gcf().set_size_inches((15, 9))
plt.show()
#%%

idxs = numpy.random.choice(range(2000), 20)
plt.plot(data[:, 0], data[:, 1], '*')
for i in idxs:
    plt.plot(data[:, 0], r['yhat1f'][2000 + i, 0, :], 'b', alpha = 0.1)
    plt.plot(data[:, 0], r['yhat2f'][2000 + i, 0, :], 'g', alpha = 0.1)

plt.ylim((0.0, 0.035))
plt.xlabel('Time')
plt.ylabel('e')
plt.title('Difference in slopes in first bit of data')
plt.gcf().set_size_inches((15, 9))
plt.show()
#%%
#for i in idxs:
#    plt.loglog(r['a'][2000 + i, :], r['Mpahat'][2000 + i, :], 'r')#%%Mpa
#plt.show()

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
#%%
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
for l in range(3):
    t = data[:, 0][Ss[l] + Ns[l] - 1] - data[:, 0][Ss[l]]
    plt.plot(r['a'][2000:, l], (r['C'][2000:, l] / r['p'][2000:, l]) / ((1 + t / r['p'][2000:, l])**2) + r['em'][2000:, l], '*')
    plt.title('Comparison of different minimum slopes for section {0}'.format(l))
    plt.xlabel('Minimum slopes (from linear fit)')
    plt.ylabel('Minimum slopes (from polynomial fit)')
    plt.show()
#%%
model_code = """
data {
  int<lower=1> N; // Number of samples
  int<lower=1> L;
  vector[N] y[L];
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
    y[l] ~ lognormal(a * Mpa[l] + b, sigma);
  }
}

generated quantities {
  vector[L] yhat;

  for (l in 1:L)
    yhat[l] <- lognormal_rng(a * Mpa[l] + b, sigma);
}
"""

sm2 = pystan.StanModel(model_code = model_code)

#%%
model_code1 = """
data {
  int<lower=1> N; // Number of single samples
  int<lower=1> L; // Number of timepoints
  vector[L] Mpa;
  vector[N] y[L];
}

parameters {
  real a;
  real b;

  real u[L];
  real<lower=0.0> sigma[L];
}

model {
  for (l in 1:L)
    y[l] ~ lognormal(u[l], sigma[l]);

  a * Mpa + b ~ normal(u, sigma);

  //for (l in 1:L)
  //{
  //  for(n in 1:N)
  //  {
  //    (log(y[l, n]) - b) / log(Mpa[l]) ~ normal(a, sigma);
  //  }
  //}
}
"""

sm3 = pystan.StanModel(model_code = model_code1)
#%%
slopes = numpy.zeros((1000, 3))

for l in range(3):
    t = data[:, 0][Ss[l] + Ns[l] - 1] - data[:, 0][Ss[l]]
    slopes[:, l] = (r['C'][3000:, l] / r['p'][3000:, l]) / ((1 + t / r['p'][3000:, l])**2) + r['em'][3000:, l]

print numpy.log(slopes).mean(axis = 0)
print numpy.log(slopes).std(axis = 0)
#%%
slopes = r['a'][3000:]

print numpy.log(slopes).mean(axis = 0)
print numpy.log(slopes).std(axis = 0)

lMpa = numpy.log(Mpa)
#%%
fit2 = sm2.sampling(data = {
  'y' : slopes.transpose(),
  'N' : slopes.shape[0],
  'L' : slopes.shape[1],
  'Mpa' : lMpa
})

r2 = fit2.extract()

seaborn.distplot(r2['a'][3000:])
plt.xlabel('"a" in log(min_slope) ~ a * log(Mpa) + b')
#plt.title('Slopes from the polynomial fit')
plt.show()

seaborn.distplot(r2['b'][3000:])
plt.xlabel('"b" in log(min_slope) ~ a * log(Mpa) + b')
plt.show()
#%%
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

idxs = numpy.random.choice(range(2000), 20)
for i in idxs:
    plt.plot(numpy.log(Mpa), numpy.log(r2['yhat'][2000 + i, :]), '-*')
plt.title('')
plt.xlabel('log(Mpa)')
plt.ylabel('log(minimum_slope)')
plt.show()

seaborn.distplot(r2['sigma'][2000:])
plt.show()
#%%
hist = numpy.zeros((100, 100))

a1 = r2['a'][2000:].mean()
b1 = r2['b'][2000:].mean()
u = a1 * numpy.log(Mpa) + b1
sigma = r2['sigma'][2000:].mean()

for i, b in enumerate(numpy.linspace(-19.0, -17.5, 100)):#min(r2['b'][2000:]), max(r2['b'][2000:])
    for j, a in enumerate(numpy.linspace(1.0, 5.0, 100)):
        hist[i, j] = numpy.exp(sum(numpy.log([scipy.stats.norm.pdf(v1, u1, sigma) for v1, u1 in zip(a * (numpy.log(Mpa) - numpy.log(Mpa[0])) + b, u)])))

plt.imshow(hist, cmap = plt.cm.jet, interpolation = 'NONE')
plt.colorbar()
plt.show()
#%%
avs = numpy.linspace(1.0, 5.0, 100)
ps = hist[:, :].sum(axis = 0)
ps /= sum(ps)

plt.plot(avs, ps2)
plt.plot(avs, ps)
plt.title('PDF of slopes')
plt.xlabel('"a" in log(min_slope) ~ a * log(Mpa) + b')
plt.legend(['Computed w/ linear fit', 'Polynomial fit'])
plt.show()
#%%
ps2 = numpy.array(ps)
#%%
seaborn.
#%%
r['yhat1'][2000 + i, :164]
#%%r['yhat1'][2000 + i, :164]
#a[l] * t[s : se] + b[l]
a = numpy.concatenate([((r['a'][2000 + i, 2] * data3[:, 0] + r['b'][2000 + i, 2]) - data3[:, 1])[len(data1) / 2:] for i in range(100, 110)])

for dist_name in ['norm']:#[ 'alpha', 'anglit', 'arcsine', 'beta', 'betaprime', 'bradford', 'burr', 'cauchy', 'chi', 'chi2', 'cosine', 'dgamma', 'dweibull', 'erlang', 'expon', 'exponweib', 'exponpow', 'f', 'fatiguelife', 'fisk', 'foldcauchy', 'foldnorm', 'frechet_r', 'frechet_l', 'genlogistic', 'genpareto', 'genexpon', 'genextreme', 'gausshyper', 'gamma', 'gengamma', 'genhalflogistic', 'gilbrat', 'gompertz', 'gumbel_r', 'gumbel_l', 'halfcauchy', 'halflogistic', 'halfnorm', 'hypsecant', 'invgamma', 'invgauss', 'invweibull', 'johnsonsb', 'johnsonsu', 'ksone', 'kstwobign', 'laplace', 'logistic', 'loggamma', 'loglaplace', 'lognorm', 'lomax', 'maxwell', 'mielke', 'nakagami', 'ncx2', 'ncf', 'nct', 'norm', 'pareto', 'pearson3', 'powerlaw', 'powerlognorm', 'powernorm', 'rdist', 'reciprocal', 'rayleigh', 'rice', 'recipinvgauss', 'semicircular', 't', 'triang', 'truncexpon', 'truncnorm', 'tukeylambda', 'uniform', 'vonmises', 'wald', 'weibull_min', 'weibull_max', 'wrapcauchy']:
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
