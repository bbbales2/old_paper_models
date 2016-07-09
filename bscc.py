#%%

import matplotlib.pyplot as plt
import numpy
import os
import seaborn
import pandas
import pystan

os.chdir('/home/bbales2/old_paper_modeling')

f = open('bscc_data.csv')
f.readline()

ts1 = []
ts2 = []
ys1 = []
ys2 = []

data = []
for line in f:
    idx, p2, p1, tr, A, Z = line.strip().split(',')

    if len(tr) == 0 or tr[0] == '<' or len(Z) == 0:
        continue

    if len(p1) > 0 and p1[0] != '<':
        ts2.append([float(p1), float(tr)])
        ys2.append([0.01, float(Z) / 100.0])

        data.append((float(p1), [0.01]))
    else:
        ts1.append([float(tr)])
        ys1.append(float(Z) / 100.0)

    data.append((float(tr), [float(Z) / 100.0]))

    print line

ts, ys = zip(*sorted(data, key = lambda x : x[0]))
f.close()
#%%

df = pandas.DataFrame({ 'trs' : trs, 'Zs' : Zs })
seaborn.jointplot(x = 'trs', y = 'Zs', data = df)

#%%

#y = (1 - numpy.exp(-t)) + b * t + c * numpy.exp(t - 10)

model_code = """
functions {
  real[] sho(real t,
             real[] y,
             real[] theta,
             real[] x_r,
             int[] x_i) {
    real dydt[1];
    dydt[1] <- (theta[1] / theta[2]) * exp(-t / theta[2]) + theta[3] + (theta[4] / theta[5]) * exp((t - theta[6]) / theta[5]);
    return dydt;
  }
}

data {
  int<lower=1> N1; // Number of single samples
  int<lower=1> N2; // Number of double samples
  real<lower=1> ts1[N1, 1];
  real<lower=1> ts2[N2, 2];
  real<lower=0.0> ys1[N1];
  real<lower=0.0> ys2[N2, 2];
  real t0;
}

transformed data {
  real x_r[0];
  int x_i[0];
}

parameters {
  real<lower=0.0> sigma;
  real<lower=0.0, upper=300.0> a;
  real<lower=0.0> b;
  real<lower=0.0> c;
  real<lower=0.0> d;
  real<lower=0.0> e;
  real<lower=0.0> f;
}

transformed parameters {
  real theta[6];

  theta[1] <- a;
  theta[2] <- b;
  theta[3] <- c;
  theta[4] <- d;
  theta[5] <- e;
  theta[6] <- f;
}

model {
  real y0[1];

  sigma ~ cauchy(0, 2.5);

  a ~ normal(0.0, 1.0);
  b ~ normal(100.0, 100.0);
  c ~ normal(0.0, 1.0);
  d ~ normal(0.0, 1.0);
  e ~ normal(3000.0, 1000.0);
  f ~ normal(3000.0, 1000.0);

  y0[1] <- 0.0;

  for (n in 1:N1)
  {
    real yhat[1, 1];

    yhat <- integrate_ode(sho, y0, t0, ts1[n], theta, x_r, x_i);

    ys1[n] ~ normal(yhat[1, 1], sigma);
  }

  for (n in 1:N2)
  {
    real yhat[2, 1];

    yhat <- integrate_ode(sho, y0, t0, ts2[n], theta, x_r, x_i);

    ys2[n, 1] ~ normal(yhat[1, 1], sigma);
    ys2[n, 2] ~ normal(yhat[2, 1], sigma);
  }
}

//generated quantities {
//  real yhat[T, 1];
//  real y0[1];
//
//  y0[1] <- 0.0;
//
//  yhat <- integrate_ode(sho, y0, t0, ts, theta, x_r, x_i);
//}
"""

sm = pystan.StanModel(model_code = model_code)
#%%

fit = sm.sampling(data = {
  'N1' : len(ts1),
  'N2' : len(ts2),
  'ts1' : ts1,
  'ts2' : ts2,
  'ys1' : ys1,
  'ys2' : ys2,
  't0' : 0.0
})
#%%

#y = (1 - numpy.exp(-t)) + b * t + c * numpy.exp(t - 10)

model_code = """
functions {
  real[] sho(real t,
             real[] y,
             real[] theta,
             real[] x_r,
             int[] x_i) {
    real dydt[1];
    dydt[1] <- (theta[1] / theta[2]) * exp(-t / theta[2]) + theta[3] + (theta[4] / theta[5]) * exp((t - theta[6]) / theta[5]);
    return dydt;
  }
}

data {
  int<lower=1> N; // Number of single samples
  real<lower=1> ts[N];
  real<lower=0.0> ys[N, 1];
  real t0;
}

transformed data {
  real x_r[0];
  int x_i[0];
}

parameters {
  real<lower=0.0> sigma;
  real<lower=0.0, upper=300.0> a;
  real<lower=0.0> b;
  real<lower=0.0> c;
  real<lower=0.0> d;
  real<lower=0.0> e;
  real<lower=0.0> f;
}

transformed parameters {
  real theta[6];

  theta[1] <- a;
  theta[2] <- b;
  theta[3] <- c;
  theta[4] <- d;
  theta[5] <- e;
  theta[6] <- f;
}

model {
  real yhat[N, 1];
  real y0[1];

  sigma ~ cauchy(0, 2.5);

  a ~ normal(0.0, 1.0);
  b ~ normal(100.0, 100.0);
  c ~ normal(0.0, 1.0);
  d ~ normal(0.0, 1.0);
  e ~ normal(3000.0, 1000.0);
  f ~ normal(3000.0, 1000.0);

  y0[1] <- 0.0;

  yhat <- integrate_ode(sho, y0, t0, ts, theta, x_r, x_i);

  for (n in 1:N)
  {
    ys[n] ~ normal(yhat[n, 1], sigma);
  }
}

//generated quantities {
//  real yhat[T, 1];
//  real y0[1];
//
//  y0[1] <- 0.0;
//
//  yhat <- integrate_ode(sho, y0, t0, ts, theta, x_r, x_i);
//}
"""

sm = pystan.StanModel(model_code = model_code)
#%%

fit = sm.sampling(data = {
  'N' : len(ts),
  'ts' : ts,
  'ys' : ys,
  't0' : 0.0
})

#%%

t = numpy.linspace(0, 10, 100)

y = (1 - numpy.exp(-t)) + 0.15 * t + numpy.exp(t - 10)

plt.plot(t, y)
plt.show()